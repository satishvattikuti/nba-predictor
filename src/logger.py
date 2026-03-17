"""
NBA Predictor - Prediction Logger
Append-only CSV log of daily predictions with result backfill.
"""

from datetime import date
from pathlib import Path

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "data" / "prediction_log.csv"

LOG_COLUMNS = [
    "prediction_date", "home_team", "away_team", "matchup",
    "home_win_prob", "away_win_prob", "predicted_winner", "confidence",
    "predicted_margin", "home_ml", "away_ml", "spread", "over_under",
    "home_implied_prob", "away_implied_prob", "edge", "spread_diff",
    "is_demo",
    "actual_winner", "home_score", "away_score", "actual_margin", "pick_correct",
    "spread_lean", "spread_cover_correct",
]


def log_predictions(predictions: pd.DataFrame) -> None:
    """Append today's predictions to the log. Skips duplicates."""
    today = str(date.today())

    # Check for existing entries
    if LOG_PATH.exists():
        existing = pd.read_csv(LOG_PATH)
        already_logged = set(
            existing.loc[existing["prediction_date"] == today, "matchup"].values
        )
    else:
        existing = pd.DataFrame(columns=LOG_COLUMNS)
        already_logged = set()

    # Build rows to append
    new_rows = []
    for _, row in predictions.iterrows():
        matchup = row.get("matchup", f"{row['away_team']} @ {row['home_team']}")
        if matchup in already_logged:
            continue
        new_rows.append({
            "prediction_date": today,
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "matchup": matchup,
            "home_win_prob": round(row.get("home_win_prob", 0), 4),
            "away_win_prob": round(row.get("away_win_prob", 0), 4),
            "predicted_winner": row.get("predicted_winner"),
            "confidence": round(row.get("confidence", 0), 4),
            "predicted_margin": round(row.get("predicted_margin", 0), 2),
            "home_ml": row.get("home_ml"),
            "away_ml": row.get("away_ml"),
            "spread": row.get("spread"),
            "over_under": row.get("over_under"),
            "home_implied_prob": round(row["home_implied_prob"], 4) if pd.notna(row.get("home_implied_prob")) else None,
            "away_implied_prob": round(row["away_implied_prob"], 4) if pd.notna(row.get("away_implied_prob")) else None,
            "edge": round(row["edge"], 2) if pd.notna(row.get("edge")) else None,
            "spread_diff": round(row["spread_diff"], 2) if pd.notna(row.get("spread_diff")) else None,
            "is_demo": row.get("is_demo", False),
            "spread_lean": ("Home" if row["spread_diff"] > 0 else "Away") if pd.notna(row.get("spread_diff")) else None,
        })

    if not new_rows:
        return

    new_df = pd.DataFrame(new_rows, columns=LOG_COLUMNS)
    header = not LOG_PATH.exists()
    new_df.to_csv(LOG_PATH, mode="a", header=header, index=False)


def load_prediction_log() -> pd.DataFrame:
    """Load the full prediction log with results backfilled."""
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(LOG_PATH)
    df["prediction_date"] = pd.to_datetime(df["prediction_date"])

    # Ensure result columns exist (for logs created before results feature)
    for col in ["actual_winner", "home_score", "away_score", "actual_margin",
                 "pick_correct", "spread_lean", "spread_cover_correct"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Backfill results for dates with missing results (including today's finished games)
    needs_results = df[df["actual_winner"].isna() & (df["prediction_date"].dt.date <= date.today())]
    if not needs_results.empty:
        dates_to_fill = needs_results["prediction_date"].dt.date.unique()
        for d in dates_to_fill:
            _backfill_results_for_date(df, d)
        # Save updated log
        df.to_csv(LOG_PATH, index=False)

    return df


_ESPN_FIX = {"WSH": "WAS", "GS": "GSW", "SA": "SAS", "NY": "NYK",
             "NO": "NOP", "UTAH": "UTA", "PHO": "PHX"}


def _backfill_results_for_date(df: pd.DataFrame, game_date) -> None:
    """Fetch final scores from ESPN for a given date and update df in place."""
    try:
        d_str = pd.Timestamp(game_date).strftime("%Y%m%d")
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={d_str}"
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return

    results = {}
    for event in data.get("events", []):
        comp = event["competitions"][0]
        status = comp.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue

        game = {}
        for c in comp.get("competitors", []):
            abbr = c["team"]["abbreviation"]
            abbr = _ESPN_FIX.get(abbr, abbr)
            game[c["homeAway"]] = {"abbr": abbr, "score": int(c.get("score", 0))}

        if "home" in game and "away" in game:
            home = game["home"]["abbr"]
            away = game["away"]["abbr"]
            home_score = game["home"]["score"]
            away_score = game["away"]["score"]
            winner = home if home_score > away_score else away
            results[(home, away)] = {
                "actual_winner": winner,
                "home_score": home_score,
                "away_score": away_score,
                "actual_margin": home_score - away_score,
            }

    mask = df["prediction_date"].dt.date == game_date
    for idx in df[mask].index:
        key = (df.at[idx, "home_team"], df.at[idx, "away_team"])
        if key in results:
            r = results[key]
            df.at[idx, "actual_winner"] = r["actual_winner"]
            df.at[idx, "home_score"] = r["home_score"]
            df.at[idx, "away_score"] = r["away_score"]
            df.at[idx, "actual_margin"] = r["actual_margin"]
            df.at[idx, "pick_correct"] = 1 if df.at[idx, "predicted_winner"] == r["actual_winner"] else 0

            # Spread cover: model lean vs actual
            spread = df.at[idx, "spread"]
            spread_diff = df.at[idx, "spread_diff"]
            if pd.notna(spread) and pd.notna(spread_diff):
                # spread_diff > 0 means model leans home cover, < 0 means away cover
                lean = "Home" if spread_diff > 0 else "Away"
                # home covers if actual_margin > -spread (i.e. home beats the spread)
                home_covered = r["actual_margin"] > -spread
                cover_hit = (lean == "Home" and home_covered) or (lean == "Away" and not home_covered)
                df.at[idx, "spread_lean"] = lean
                df.at[idx, "spread_cover_correct"] = 1 if cover_hit else 0
