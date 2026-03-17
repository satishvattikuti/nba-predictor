"""
NBA Predictor - XGBoost Models (Winner + Spread)
"""

import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, mean_absolute_error, mean_squared_error
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import TEAM_DISPLAY


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get feature columns (home_*/away_* + elo_diff, except metadata)."""
    exclude = {"home_team", "away_team", "home_win", "home_pts", "away_pts", "home_margin"}
    cols = [c for c in df.columns
            if (c.startswith("home_") or c.startswith("away_")) and c not in exclude]
    if "elo_diff" in df.columns:
        cols.append("elo_diff")
    return cols


def train_models(feature_matrix: pd.DataFrame) -> tuple:
    """Train winner classifier and spread regressor."""
    config = load_config()
    df = feature_matrix.copy()

    # Time-based split: train on everything except last 30 days, validate on last 30 days
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE")
    cutoff = df["GAME_DATE"].max() - pd.Timedelta(days=30)
    train_df = df[df["GAME_DATE"] <= cutoff]
    val_df = df[df["GAME_DATE"] > cutoff]

    feature_cols = get_feature_columns(df)
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(train_df)} games, Val: {len(val_df)} games")

    X_train = train_df[feature_cols].astype(float)
    X_val = val_df[feature_cols].astype(float)

    # ── Model 1: Winner (Classifier) ──
    print("\n  Training Winner Model...")
    y_train_win = train_df["home_win"].astype(int)
    y_val_win = val_df["home_win"].astype(int)

    winner_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        min_child_weight=5,
        gamma=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    winner_model.fit(X_train, y_train_win, eval_set=[(X_val, y_val_win)], verbose=False)

    y_pred_proba = winner_model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    winner_metrics = {
        "accuracy": float(accuracy_score(y_val_win, y_pred)),
        "auc_roc": float(roc_auc_score(y_val_win, y_pred_proba)),
        "log_loss": float(log_loss(y_val_win, y_pred_proba)),
        "home_win_rate_actual": float(y_val_win.mean()),
    }

    print(f"    Accuracy: {winner_metrics['accuracy']:.3f}")
    print(f"    AUC-ROC:  {winner_metrics['auc_roc']:.3f}")
    print(f"    Log Loss: {winner_metrics['log_loss']:.3f}")

    # ── Model 2: Spread (Regressor) ──
    print("\n  Training Spread Model...")
    y_train_margin = train_df["home_margin"].astype(float)
    y_val_margin = val_df["home_margin"].astype(float)

    spread_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=2.0,
        min_child_weight=5,
        gamma=0.1,
        objective="reg:squarederror",
        eval_metric="mae",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    spread_model.fit(X_train, y_train_margin, eval_set=[(X_val, y_val_margin)], verbose=False)

    y_pred_margin = spread_model.predict(X_val)

    spread_metrics = {
        "mae": float(mean_absolute_error(y_val_margin, y_pred_margin)),
        "rmse": float(np.sqrt(mean_squared_error(y_val_margin, y_pred_margin))),
        "avg_actual_margin": float(y_val_margin.mean()),
        "avg_predicted_margin": float(y_pred_margin.mean()),
    }

    print(f"    MAE:  {spread_metrics['mae']:.2f} points")
    print(f"    RMSE: {spread_metrics['rmse']:.2f} points")

    # Combined metrics
    metrics = {
        "winner": winner_metrics,
        "spread": spread_metrics,
        "n_features": len(feature_cols),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
    }

    return winner_model, spread_model, feature_cols, metrics


def save_models(winner_model, spread_model, feature_cols: list[str], metrics: dict):
    """Save both models and metadata."""
    config = load_config()

    joblib.dump(winner_model, PROJECT_ROOT / config["winner_model_path"])
    joblib.dump(spread_model, PROJECT_ROOT / config["spread_model_path"])

    with open(PROJECT_ROOT / config["feature_cols_path"], "w") as f:
        json.dump(feature_cols, f, indent=2)

    with open(PROJECT_ROOT / config["metrics_path"], "w") as f:
        json.dump(metrics, f, indent=2)

    print("  Models and metadata saved.")


def load_models() -> tuple:
    """Load both models, feature cols, and metrics."""
    config = load_config()

    winner_model = joblib.load(PROJECT_ROOT / config["winner_model_path"])
    spread_model = joblib.load(PROJECT_ROOT / config["spread_model_path"])

    with open(PROJECT_ROOT / config["feature_cols_path"]) as f:
        feature_cols = json.load(f)

    with open(PROJECT_ROOT / config["metrics_path"]) as f:
        metrics = json.load(f)

    return winner_model, spread_model, feature_cols, metrics


def predict_today(today_features: pd.DataFrame, injury_impact: dict = None) -> pd.DataFrame:
    """Run both models on today's games, with optional injury adjustments."""
    winner_model, spread_model, feature_cols, _ = load_models()

    # Align columns
    X = pd.DataFrame(columns=feature_cols)
    for col in feature_cols:
        X[col] = today_features[col].values if col in today_features.columns else 0.0
    X = X.astype(float)

    # Winner predictions
    win_proba = winner_model.predict_proba(X)[:, 1]

    # Spread predictions
    pred_margin = spread_model.predict(X)

    results = pd.DataFrame({
        "home_team": today_features["home_team"].values,
        "away_team": today_features["away_team"].values,
        "home_win_prob": win_proba,
        "away_win_prob": 1 - win_proba,
        "predicted_margin": pred_margin,
        "is_demo": today_features.get("is_demo", True).values,
    })

    # ── Injury adjustments ──
    # Since we can't train on historical injury data (not available),
    # we apply a post-model adjustment based on missing player impact.
    # Rule: each 10% of missing minutes shifts the spread ~2 points against that team.
    # Capped at 30% impact to avoid overreacting to teams resting many bench players.
    INJURY_PTS_PER_10PCT = 2.0
    MAX_IMPACT_PCT = 0.50  # cap at 50% — beyond this, diminishing returns
    results["home_win_prob"] = results["home_win_prob"].astype(float)
    results["away_win_prob"] = results["away_win_prob"].astype(float)
    results["home_injury_pct"] = 0.0
    results["away_injury_pct"] = 0.0
    results["injury_adj"] = 0.0

    if injury_impact:
        for idx, row in results.iterrows():
            h_impact = injury_impact.get(row["home_team"], {})
            a_impact = injury_impact.get(row["away_team"], {})
            h_miss = min(h_impact.get("missing_min_pct", 0.0), MAX_IMPACT_PCT)
            a_miss = min(a_impact.get("missing_min_pct", 0.0), MAX_IMPACT_PCT)

            results.at[idx, "home_injury_pct"] = h_miss
            results.at[idx, "away_injury_pct"] = a_miss

            # Positive adjustment = favors home (away is more hurt)
            # Negative adjustment = favors away (home is more hurt)
            adj = (a_miss - h_miss) * 10 * INJURY_PTS_PER_10PCT
            results.at[idx, "injury_adj"] = adj

        # Apply adjustment to margin and recalculate win prob
        results["predicted_margin"] = results["predicted_margin"] + results["injury_adj"]

        # Adjust win probability: shift logit by injury differential
        for idx, row in results.iterrows():
            if row["injury_adj"] != 0:
                # Convert prob to logit, shift, convert back
                p = np.clip(row["home_win_prob"], 0.01, 0.99)
                logit = np.log(p / (1 - p))
                # ~0.1 logit shift per point of margin adjustment
                logit += row["injury_adj"] * 0.1
                new_p = 1 / (1 + np.exp(-logit))
                results.at[idx, "home_win_prob"] = new_p
                results.at[idx, "away_win_prob"] = 1 - new_p

    results["predicted_winner"] = np.where(
        results["home_win_prob"] >= 0.5,
        results["home_team"],
        results["away_team"],
    )
    results["confidence"] = np.abs(results["home_win_prob"] - 0.5) * 2

    # Display names
    results["home_display"] = results["home_team"].map(TEAM_DISPLAY)
    results["away_display"] = results["away_team"].map(TEAM_DISPLAY)
    results["winner_display"] = results["predicted_winner"].map(TEAM_DISPLAY)
    results["matchup"] = results["away_display"] + " @ " + results["home_display"]

    return results


# ── Main ─────────────────────────────────────────────────────────────

def run_training_pipeline():
    """Main entry point."""
    config = load_config()
    cache_dir = PROJECT_ROOT / config["cache_dir"]

    print("NBA Predictor - Model Training")
    print("=" * 50)

    matrix_path = cache_dir / "feature_matrix.parquet"
    if not matrix_path.exists():
        print("\nFeature matrix not found. Building...")
        from src.features import build_game_feature_matrix_multi

        seasons = config["seasons"]
        all_logs = []
        all_advs = []
        for season in seasons:
            log_path = cache_dir / f"game_logs_{season.replace('-', '_')}.parquet"
            adv_path = cache_dir / f"team_advanced_{season.replace('-', '_')}.parquet"
            if log_path.exists():
                all_logs.append(pd.read_parquet(log_path))
            if adv_path.exists():
                all_advs.append(pd.read_parquet(adv_path))

        build_game_feature_matrix_multi(all_logs, all_advs)

    matrix = pd.read_parquet(matrix_path)
    print(f"\nFeature matrix: {matrix.shape[0]} games, {matrix.shape[1]} columns")

    print("\nTraining models...")
    winner_model, spread_model, feature_cols, metrics = train_models(matrix)

    print("\nSaving...")
    save_models(winner_model, spread_model, feature_cols, metrics)

    print("\nDone!")


if __name__ == "__main__":
    run_training_pipeline()
