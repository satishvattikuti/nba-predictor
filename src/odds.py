"""
NBA Predictor - ESPN Odds (Free, no API key)
Fetches moneyline, spreads, and totals from ESPN's scoreboard API.
"""

import sys
from pathlib import Path
from datetime import datetime

import httpx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import normalize_team

# ESPN abbreviation fixes
_ESPN_ABBREV_MAP = {
    "WSH": "WAS",
    "GS": "GSW",
    "SA": "SAS",
    "NY": "NYK",
    "NO": "NOP",
    "UTAH": "UTA",
    "PHO": "PHX",
    "BKN": "BKN",
}

ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"


def _american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds is None or odds == 0:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def fetch_espn_odds() -> pd.DataFrame:
    """
    Fetch today's NBA odds from ESPN's free scoreboard API.
    Returns DataFrame with: home_team, away_team, home_ml, away_ml,
    spread, over_under, home_implied_prob, away_implied_prob.
    """
    try:
        resp = httpx.get(ESPN_SCOREBOARD_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ESPN odds fetch failed: {e}")
        return pd.DataFrame()

    rows = []
    events = data.get("events", [])

    for event in events:
        competitions = event.get("competitions", [])
        if not competitions:
            continue

        comp = competitions[0]

        # Get teams
        competitors = comp.get("competitors", [])
        home_info = None
        away_info = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home_info = c
            elif c.get("homeAway") == "away":
                away_info = c

        if not home_info or not away_info:
            continue

        home_abbr = home_info.get("team", {}).get("abbreviation", "")
        away_abbr = away_info.get("team", {}).get("abbreviation", "")

        # Normalize
        home_abbr = _ESPN_ABBREV_MAP.get(home_abbr, home_abbr)
        away_abbr = _ESPN_ABBREV_MAP.get(away_abbr, away_abbr)

        # Get odds
        odds_list = comp.get("odds", [])
        if not odds_list:
            # Still include game without odds
            rows.append({
                "home_team": home_abbr,
                "away_team": away_abbr,
                "home_ml": None,
                "away_ml": None,
                "spread": None,
                "over_under": None,
                "home_implied_prob": None,
                "away_implied_prob": None,
            })
            continue

        odds = odds_list[0]  # First bookmaker (usually consensus/DraftKings)

        home_ml = None
        away_ml = None
        spread = None
        over_under = None

        # Moneyline
        home_odds = odds.get("homeTeamOdds", {})
        away_odds = odds.get("awayTeamOdds", {})
        home_ml = home_odds.get("moneyLine")
        away_ml = away_odds.get("moneyLine")

        # Spread (from home team perspective)
        spread_val = odds.get("spread")
        if spread_val is not None:
            try:
                spread = float(spread_val)
            except (ValueError, TypeError):
                spread = None

        # Over/Under
        ou_val = odds.get("overUnder")
        if ou_val is not None:
            try:
                over_under = float(ou_val)
            except (ValueError, TypeError):
                over_under = None

        # Implied probabilities from moneyline
        home_implied = _american_to_implied(home_ml) if home_ml else None
        away_implied = _american_to_implied(away_ml) if away_ml else None

        rows.append({
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_ml": home_ml,
            "away_ml": away_ml,
            "spread": spread,
            "over_under": over_under,
            "home_implied_prob": home_implied,
            "away_implied_prob": away_implied,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        # Cache
        cache_path = PROJECT_ROOT / "data" / "espn_odds.csv"
        df.to_csv(cache_path, index=False)

    return df


def get_odds_for_game(odds_df: pd.DataFrame, home_team: str, away_team: str) -> dict:
    """Get odds for a specific game matchup."""
    if odds_df.empty:
        return {}

    match = odds_df[
        (odds_df["home_team"] == home_team) & (odds_df["away_team"] == away_team)
    ]

    if match.empty:
        return {}

    row = match.iloc[0]
    return row.to_dict()


if __name__ == "__main__":
    print("Fetching ESPN NBA odds...")
    df = fetch_espn_odds()
    if df.empty:
        print("No odds available.")
    else:
        print(f"Got odds for {len(df)} games:")
        print(df.to_string(index=False))
