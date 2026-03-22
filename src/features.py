"""
NBA Predictor - Feature Engineering
Rolling 10-game features with anti-leakage protection.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import normalize_team


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ── Rolling features ─────────────────────────────────────────────────

def compute_rolling_features(game_logs: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Compute rolling features per team. shift(1) prevents data leakage.
    """
    df = game_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

    team_col = "TEAM_ABBREVIATION"

    # Basic scoring
    for col in ["PTS", "REB", "AST", "TOV", "STL", "BLK"]:
        if col in df.columns:
            df[f"roll_{col.lower()}"] = (
                df.groupby(team_col)[col]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
            )

    # Points allowed (opponent points) - need to compute from matchup data
    # For now, use game-level stats
    if "PLUS_MINUS" in df.columns and "PTS" in df.columns:
        df["PTS_ALLOWED"] = df["PTS"] - df["PLUS_MINUS"]
        df["roll_pts_allowed"] = (
            df.groupby(team_col)["PTS_ALLOWED"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
        )

    # Win percentage
    df["roll_win_pct"] = (
        df.groupby(team_col)["WIN"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
    )

    # eFG% = (FGM + 0.5 * FG3M) / FGA
    if all(c in df.columns for c in ["FGM", "FG3M", "FGA"]):
        df["eFG_PCT"] = (df["FGM"] + 0.5 * df["FG3M"]) / df["FGA"].replace(0, np.nan)
        df["roll_efg"] = (
            df.groupby(team_col)["eFG_PCT"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
        )

    # TS% = PTS / (2 * (FGA + 0.44 * FTA))
    if all(c in df.columns for c in ["PTS", "FGA", "FTA"]):
        denom = 2 * (df["FGA"] + 0.44 * df["FTA"])
        df["TS_PCT"] = df["PTS"] / denom.replace(0, np.nan)
        df["roll_ts"] = (
            df.groupby(team_col)["TS_PCT"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
        )

    # Turnover rate = TOV / (FGA + 0.44 * FTA + TOV)
    if all(c in df.columns for c in ["TOV", "FGA", "FTA"]):
        denom = df["FGA"] + 0.44 * df["FTA"] + df["TOV"]
        df["TOV_RATE"] = df["TOV"] / denom.replace(0, np.nan)
        df["roll_tov_rate"] = (
            df.groupby(team_col)["TOV_RATE"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
        )

    # Rebound rate (simple: REB / (REB + OPP_REB approximate))
    if "REB" in df.columns:
        df["roll_reb"] = (
            df.groupby(team_col)["REB"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
        )

    # 3PT rate and %
    if "FG3A" in df.columns and "FGA" in df.columns:
        df["FG3_RATE"] = df["FG3A"] / df["FGA"].replace(0, np.nan)
        df["roll_fg3_rate"] = (
            df.groupby(team_col)["FG3_RATE"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
        )

    if "FG3_PCT" in df.columns:
        df["roll_fg3_pct"] = (
            df.groupby(team_col)["FG3_PCT"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
        )

    # Plus/minus (net rating proxy)
    if "PLUS_MINUS" in df.columns:
        df["roll_plus_minus"] = (
            df.groupby(team_col)["PLUS_MINUS"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
        )

    # Win/loss streak
    df["streak"] = (
        df.groupby(team_col)["WIN"]
        .transform(lambda x: _compute_streak(x).shift(1))
    )

    # Season win% (expanding — captures overall team caliber)
    df["season_win_pct"] = (
        df.groupby(team_col)["WIN"]
        .transform(lambda x: x.shift(1).expanding(min_periods=10).mean())
    )

    # Net rating (rolling pts scored - pts allowed per game)
    if "PLUS_MINUS" in df.columns:
        df["roll_net_rtg"] = (
            df.groupby(team_col)["PLUS_MINUS"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=5).mean())
        )

    # Margin volatility (std of plus_minus — captures blowout-prone teams)
    if "PLUS_MINUS" in df.columns:
        df["margin_volatility"] = (
            df.groupby(team_col)["PLUS_MINUS"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=5).std())
        )

    # Momentum (recent 5 vs prior 5 plus_minus avg)
    if "PLUS_MINUS" in df.columns:
        recent_5 = df.groupby(team_col)["PLUS_MINUS"].transform(
            lambda x: x.shift(1).rolling(5, min_periods=3).mean()
        )
        prior_5 = df.groupby(team_col)["PLUS_MINUS"].transform(
            lambda x: x.shift(6).rolling(5, min_periods=3).mean()
        )
        df["momentum"] = recent_5 - prior_5

    return df


def _compute_streak(series: pd.Series) -> pd.Series:
    """Compute current win/loss streak. Positive = wins, negative = losses."""
    streaks = []
    current = 0
    for val in series:
        if pd.isna(val):
            streaks.append(0)
            continue
        if val == 1:
            current = max(1, current + 1)
        else:
            current = min(-1, current - 1)
        streaks.append(current)
    return pd.Series(streaks, index=series.index)


def compute_elo(game_logs: pd.DataFrame, k: float = 20, home_adv: float = 100) -> pd.DataFrame:
    """Compute Elo ratings for each team. Pre-game Elo only (no leakage)."""
    df = game_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE").reset_index(drop=True)

    # Initialize Elo per season
    elo = {}

    # We need to process game-by-game (each game appears twice: home + away row)
    # Build a game-level view first
    home_rows = df[df["is_home"] == 1][["GAME_DATE", "TEAM_ABBREVIATION", "OPP", "WIN"]].copy()
    home_rows.columns = ["GAME_DATE", "home_team", "away_team", "home_win"]
    home_rows = home_rows.drop_duplicates(subset=["GAME_DATE", "home_team", "away_team"])
    home_rows = home_rows.sort_values("GAME_DATE").reset_index(drop=True)

    # Track pre-game Elo for each game
    game_elos = {}  # (date, home, away) -> (home_elo, away_elo)

    # Detect season boundaries by large gaps (>60 days)
    prev_date = None
    for _, game in home_rows.iterrows():
        gd = game["GAME_DATE"]
        home = game["home_team"]
        away = game["away_team"]
        hw = game["home_win"]

        # Reset Elo at season boundary
        if prev_date is not None and (gd - prev_date).days > 60:
            elo = {t: 1500 for t in elo}
        prev_date = gd

        # Initialize new teams
        if home not in elo:
            elo[home] = 1500
        if away not in elo:
            elo[away] = 1500

        # Store pre-game Elo
        h_elo = elo[home]
        a_elo = elo[away]
        game_elos[(str(gd), home, away)] = (h_elo, a_elo)

        # Expected scores
        exp_home = 1 / (1 + 10 ** ((a_elo - h_elo - home_adv) / 400))
        actual_home = float(hw) if pd.notna(hw) else 0.5

        # Update
        elo[home] = h_elo + k * (actual_home - exp_home)
        elo[away] = a_elo + k * ((1 - actual_home) - (1 - exp_home))

    # Map Elo back to each row
    df["elo"] = np.nan
    for idx, row in df.iterrows():
        gd = str(row["GAME_DATE"])
        if row["is_home"] == 1:
            key = (gd, row["TEAM_ABBREVIATION"], row["OPP"])
            if key in game_elos:
                df.at[idx, "elo"] = game_elos[key][0]
        else:
            key = (gd, row["OPP"], row["TEAM_ABBREVIATION"])
            if key in game_elos:
                df.at[idx, "elo"] = game_elos[key][1]

    return df, elo  # Return current elo dict for today's predictions


def compute_venue_splits(game_logs: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Compute rolling stats split by home/away venue. Uses shift(1) for anti-leakage."""
    df = game_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)

    team_col = "TEAM_ABBREVIATION"

    for col, name in [("PTS", "venue_pts"), ("WIN", "venue_win_pct"), ("PLUS_MINUS", "venue_plus_minus")]:
        if col not in df.columns:
            continue
        # Rolling average within home-only and away-only games per team
        df[name] = (
            df.groupby([team_col, "is_home"])[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
        )

    return df


def compute_rest_features(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Compute days rest and back-to-back flag."""
    df = game_logs.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])

    df["prev_date"] = df.groupby("TEAM_ABBREVIATION")["GAME_DATE"].shift(1)
    df["days_rest"] = (df["GAME_DATE"] - df["prev_date"]).dt.days
    df["days_rest"] = df["days_rest"].fillna(3).clip(0, 7)
    df["is_b2b"] = (df["days_rest"] <= 1).astype(int)

    return df


# ── Build game matrix ────────────────────────────────────────────────

def build_game_feature_matrix_multi(log_list: list, adv_list: list) -> pd.DataFrame:
    """Build feature matrix from multiple seasons."""
    # Merge advanced stats into each season's logs before combining
    merged = []
    for i, logs in enumerate(log_list):
        adv = adv_list[i] if i < len(adv_list) else None
        m = _merge_adv_single(logs.copy(), adv)
        m["season_idx"] = i
        merged.append(m)

    all_logs = pd.concat(merged, ignore_index=True)
    return _build_matrix_from_logs(all_logs)


def build_game_feature_matrix(train_logs: pd.DataFrame,
                               val_logs: pd.DataFrame,
                               train_adv: pd.DataFrame = None,
                               val_adv: pd.DataFrame = None) -> pd.DataFrame:
    """Build feature matrix from two seasons (legacy)."""
    train_logs = _merge_adv_single(train_logs.copy(), train_adv)
    val_logs = _merge_adv_single(val_logs.copy(), val_adv)
    train_logs["season"] = "train"
    val_logs["season"] = "val"
    all_logs = pd.concat([train_logs, val_logs], ignore_index=True)
    return _build_matrix_from_logs(all_logs)


def _build_matrix_from_logs(all_logs: pd.DataFrame) -> pd.DataFrame:
    """Core matrix building from combined game logs."""
    config = load_config()
    window = config["rolling_window"]

    print("  Computing rolling features...")
    all_logs = compute_rolling_features(all_logs, window)

    print("  Computing rest features...")
    all_logs = compute_rest_features(all_logs)

    print("  Computing venue splits...")
    all_logs = compute_venue_splits(all_logs, window)

    print("  Computing Elo ratings...")
    all_logs, _current_elo = compute_elo(all_logs)

    # Feature columns
    roll_cols = [c for c in all_logs.columns if c.startswith("roll_")]
    venue_cols = [c for c in all_logs.columns if c.startswith("venue_")]
    extra_cols = ["days_rest", "is_b2b", "streak", "elo",
                  "season_win_pct", "margin_volatility", "momentum"]
    adv_cols = [c for c in all_logs.columns if c.startswith("adv_")]
    team_features = roll_cols + venue_cols + extra_cols + adv_cols
    team_features = [f for f in team_features if f in all_logs.columns]

    print(f"  Team features: {len(team_features)}")

    # Build home/away matrix
    print("  Building home/away game matrix...")
    home_df = all_logs[all_logs["is_home"] == 1].copy()
    away_df = all_logs[all_logs["is_home"] == 0].copy()

    # Rename for home
    home_rename = {f: f"home_{f}" for f in team_features}
    home_rename["TEAM_ABBREVIATION"] = "home_team"
    home_rename["WIN"] = "home_win"
    home_rename["PTS"] = "home_pts"
    base_cols = ["GAME_DATE", "TEAM_ABBREVIATION", "OPP", "WIN", "PTS"]
    home_slim = home_df[
        base_cols + team_features
    ].rename(columns=home_rename)
    home_slim = home_slim.rename(columns={"OPP": "away_team"})

    # Away
    away_rename = {f: f"away_{f}" for f in team_features}
    away_rename["TEAM_ABBREVIATION"] = "away_team_check"
    away_rename["PTS"] = "away_pts"
    away_slim = away_df[
        ["GAME_DATE", "TEAM_ABBREVIATION", "OPP", "PTS"] + team_features
    ].rename(columns=away_rename)
    away_slim = away_slim.rename(columns={"OPP": "home_team_check"})

    # Merge
    matrix = home_slim.merge(
        away_slim,
        left_on=["GAME_DATE", "home_team", "away_team"],
        right_on=["GAME_DATE", "home_team_check", "away_team_check"],
        how="inner",
    )
    matrix = matrix.drop(columns=["home_team_check", "away_team_check"], errors="ignore")

    # Elo differential
    if "home_elo" in matrix.columns and "away_elo" in matrix.columns:
        matrix["elo_diff"] = matrix["home_elo"] - matrix["away_elo"]

    # Team quality differentials — key for big spread games
    if "home_season_win_pct" in matrix.columns and "away_season_win_pct" in matrix.columns:
        matrix["season_win_diff"] = matrix["home_season_win_pct"] - matrix["away_season_win_pct"]
    if "home_roll_net_rtg" in matrix.columns and "away_roll_net_rtg" in matrix.columns:
        matrix["net_rtg_diff"] = matrix["home_roll_net_rtg"] - matrix["away_roll_net_rtg"]

    # Home margin (for spread model)
    matrix["home_margin"] = matrix["home_pts"] - matrix["away_pts"]

    # Drop rows with NaN features
    feat_cols = [c for c in matrix.columns
                 if (c.startswith("home_") or c.startswith("away_"))
                 and c not in ("home_team", "away_team", "home_win", "home_pts",
                               "away_pts", "home_margin")]
    before = len(matrix)
    matrix = matrix.dropna(subset=feat_cols)
    print(f"  Dropped {before - len(matrix)} rows with NaN ({len(matrix)} remaining)")

    # Save
    cache_dir = PROJECT_ROOT / config["cache_dir"]
    matrix.to_parquet(cache_dir / "feature_matrix.parquet", index=False)
    print(f"  Saved feature matrix: {matrix.shape}")

    return matrix


def _merge_adv_single(logs: pd.DataFrame, adv_df: pd.DataFrame) -> pd.DataFrame:
    """Merge season-level advanced stats onto a single season's game logs."""
    if adv_df is None or adv_df.empty:
        return logs

    adv_col_map = {
        "E_OFF_RATING": "adv_off_rtg",
        "E_DEF_RATING": "adv_def_rtg",
        "E_NET_RATING": "adv_net_rtg",
        "E_PACE": "adv_pace",
        "E_AST_RATIO": "adv_ast_ratio",
        "E_REB_PCT": "adv_reb_pct",
    }

    available = {k: v for k, v in adv_col_map.items() if k in adv_df.columns}
    if not available:
        return logs

    adv = adv_df.copy()
    if "TEAM_ABBREVIATION" not in adv.columns and "TEAM_NAME" in adv.columns:
        adv["TEAM_ABBREVIATION"] = adv["TEAM_NAME"].apply(
            lambda x: normalize_team(str(x))
        )
    if "TEAM_ABBREVIATION" not in adv.columns:
        return logs

    slim = adv[["TEAM_ABBREVIATION"] + list(available.keys())].rename(columns=available)
    return logs.merge(slim, on="TEAM_ABBREVIATION", how="left")


# ── Today's features ─────────────────────────────────────────────────

def build_today_features(today_schedule: pd.DataFrame,
                          game_logs: pd.DataFrame,
                          adv_stats: pd.DataFrame = None) -> pd.DataFrame:
    """Build features for today's games using the SAME functions as training.

    This eliminates train/serve skew by reusing compute_rolling_features(),
    compute_rest_features(), compute_venue_splits(), and compute_elo().
    """
    config = load_config()
    window = config["rolling_window"]

    df = game_logs.copy()
    df = _merge_adv_single(df, adv_stats)

    # Reuse the exact same feature computation pipeline as training
    df = compute_rolling_features(df, window)
    df = compute_rest_features(df)
    df = compute_venue_splits(df, window)
    df, current_elo = compute_elo(df)

    # Identify feature columns (same logic as _build_matrix_from_logs)
    roll_cols = [c for c in df.columns if c.startswith("roll_")]
    venue_cols = [c for c in df.columns if c.startswith("venue_")]
    extra_cols = ["days_rest", "is_b2b", "streak", "elo",
                  "season_win_pct", "margin_volatility", "momentum"]
    adv_cols = [c for c in df.columns if c.startswith("adv_")]
    team_features = roll_cols + venue_cols + extra_cols + adv_cols
    team_features = [f for f in team_features if f in df.columns]

    # Get last row per team (already has shift(1) applied — no leakage)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    latest = df.groupby("TEAM_ABBREVIATION").last()

    # Build rows for each game
    rows = []
    for _, game in today_schedule.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        row = {"home_team": home, "away_team": away, "is_demo": game.get("is_demo", False)}

        for prefix, team in [("home", home), ("away", away)]:
            if team in latest.index:
                team_row = latest.loc[team]
                for feat in team_features:
                    val = team_row.get(feat, np.nan)
                    row[f"{prefix}_{feat}"] = val
            else:
                for feat in team_features:
                    row[f"{prefix}_{feat}"] = 0.0

        # Elo ratings
        home_elo = current_elo.get(home, 1500)
        away_elo = current_elo.get(away, 1500)
        row["home_elo"] = home_elo
        row["away_elo"] = away_elo
        row["elo_diff"] = home_elo - away_elo

        # Team quality differentials
        if f"home_season_win_pct" in row and f"away_season_win_pct" in row:
            h_swp = row.get("home_season_win_pct", 0.5) or 0.5
            a_swp = row.get("away_season_win_pct", 0.5) or 0.5
            row["season_win_diff"] = h_swp - a_swp
        if f"home_roll_net_rtg" in row and f"away_roll_net_rtg" in row:
            h_nrt = row.get("home_roll_net_rtg", 0) or 0
            a_nrt = row.get("away_roll_net_rtg", 0) or 0
            row["net_rtg_diff"] = h_nrt - a_nrt

        rows.append(row)

    return pd.DataFrame(rows)


# Need this import for build_today_features
from src.ingest import _NBA_API_ABBREV_FIXES
