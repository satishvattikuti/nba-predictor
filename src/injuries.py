"""
NBA Predictor - Injury / Player Availability
Fetches injury reports from ESPN + player season averages from nba_api.
Computes missing-minutes % and missing-points % per team.
"""

import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import normalize_team

ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"

# ESPN abbreviation fixes (same as odds.py)
_ESPN_ABBREV_MAP = {
    "WSH": "WAS", "GS": "GSW", "SA": "SAS", "NY": "NYK",
    "NO": "NOP", "UTAH": "UTA", "PHO": "PHX",
}

TEAM_MINUTES_PER_GAME = 240  # 5 players * 48 min


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ── ESPN Injury Report ──────────────────────────────────────────────

def fetch_injury_report() -> pd.DataFrame:
    """
    Fetch current NBA injury report from ESPN.
    Returns DataFrame with: player, team, status, fantasy_status.
    """
    try:
        resp = httpx.get(ESPN_INJURIES_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  ESPN injury fetch failed: {e}")
        return pd.DataFrame()

    rows = []
    for team_entry in data.get("injuries", []):
        # Get team abbreviation
        team_name = team_entry.get("displayName", "")
        try:
            team_abbr = normalize_team(team_name)
        except ValueError:
            continue

        for inj in team_entry.get("injuries", []):
            athlete = inj.get("athlete", {})
            details = inj.get("details", {})
            fs_raw = details.get("fantasyStatus", "")
            # fantasyStatus can be a dict {"description": "OUT", "abbreviation": "OUT"}
            if isinstance(fs_raw, dict):
                fantasy_status = fs_raw.get("abbreviation", "")
            else:
                fantasy_status = str(fs_raw)

            # Only care about OUT and OFS (out for season)
            if fantasy_status not in ("OUT", "OFS"):
                continue

            rows.append({
                "player": athlete.get("displayName", ""),
                "team": team_abbr,
                "status": inj.get("status", ""),
                "fantasy_status": fantasy_status,
                "injury_type": details.get("type", ""),
                "detail": details.get("detail", ""),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        cache_path = PROJECT_ROOT / "data" / "injury_report.csv"
        df.to_csv(cache_path, index=False)
        print(f"  Injury report: {len(df)} players OUT across {df['team'].nunique()} teams")
    else:
        print("  No injured players found (OUT/OFS)")

    return df


# ── Player Season Averages ──────────────────────────────────────────

def fetch_player_averages(season: str = None) -> pd.DataFrame:
    """
    Fetch per-game averages for all players this season via nba_api.
    Returns DataFrame with: PLAYER_NAME, TEAM_ABBREVIATION, MIN, PTS, GP.
    """
    if season is None:
        config = load_config()
        season = config["current_season"]

    cache_path = PROJECT_ROOT / "data" / "player_averages.parquet"

    # Cache for 12 hours
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < 12:
            print(f"  Loading cached player averages")
            return pd.read_parquet(cache_path)

    print(f"  Fetching player averages for {season}...")
    from nba_api.stats.endpoints import leaguedashplayerstats

    time.sleep(0.6)
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed="PerGame",
        season_type_all_star="Regular Season",
    )
    df = stats.get_data_frames()[0]
    time.sleep(0.6)

    # Normalize team abbreviations
    from src.ingest import _NBA_API_ABBREV_FIXES
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].apply(
        lambda x: _NBA_API_ABBREV_FIXES.get(x, x)
    )

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  Player averages: {len(df)} players")

    return df


# ── Compute Team Impact ─────────────────────────────────────────────

def _classify_role(mpg: float) -> str:
    """Classify player role by minutes per game."""
    if mpg >= 25:
        return "Starter"
    elif mpg >= 15:
        return "Rotation"
    return "Bench"


def _normalize_name(name: str) -> str:
    """Normalize player name for matching (lowercase, strip suffixes)."""
    name = name.strip().lower()
    for suffix in [" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    return name


def compute_team_injury_impact(injury_df: pd.DataFrame,
                                player_avg_df: pd.DataFrame) -> dict:
    """
    Compute missing minutes % and points % per team.
    Returns dict: {team_abbr: {"missing_min_pct": float, "missing_pts_pct": float, "n_out": int}}
    """
    if injury_df.empty or player_avg_df.empty:
        return {}

    # Build player lookup: normalized_name -> (TEAM, MIN, PTS)
    player_lookup = {}
    for _, row in player_avg_df.iterrows():
        key = _normalize_name(row["PLAYER_NAME"])
        player_lookup[key] = {
            "team": row["TEAM_ABBREVIATION"],
            "min": float(row.get("MIN", 0)),
            "pts": float(row.get("PTS", 0)),
            "gp": int(row.get("GP", 0)),
        }

    # Team total points (sum of all player PPG on the team)
    team_total_pts = player_avg_df.groupby("TEAM_ABBREVIATION")["PTS"].sum().to_dict()

    impact = {}
    for _, inj in injury_df.iterrows():
        team = inj["team"]
        player_name = _normalize_name(inj["player"])

        # Look up player stats
        pinfo = player_lookup.get(player_name)

        # If exact match fails, try partial match within same team
        if pinfo is None:
            for pname, pdata in player_lookup.items():
                if pdata["team"] == team and (pname in player_name or player_name in pname):
                    pinfo = pdata
                    break

        if pinfo is None or pinfo["gp"] < 5:
            continue

        if team not in impact:
            impact[team] = {
                "missing_min": 0.0, "missing_pts": 0.0,
                "n_out": 0, "starters_out": 0, "rotation_out": 0,
                "key_players": [],
            }

        role = _classify_role(pinfo["min"])
        impact[team]["missing_min"] += pinfo["min"]
        impact[team]["missing_pts"] += pinfo["pts"]
        impact[team]["n_out"] += 1
        if role == "Starter":
            impact[team]["starters_out"] += 1
        elif role == "Rotation":
            impact[team]["rotation_out"] += 1
        if role in ("Starter", "Rotation"):
            impact[team]["key_players"].append({
                "name": inj["player"], "role": role,
                "mpg": pinfo["min"], "ppg": pinfo["pts"],
            })

    # Convert to percentages
    result = {}
    for team, data in impact.items():
        total_pts = team_total_pts.get(team, 115.0)  # fallback
        result[team] = {
            "missing_min_pct": min(data["missing_min"] / TEAM_MINUTES_PER_GAME, 1.0),
            "missing_pts_pct": min(data["missing_pts"] / total_pts, 1.0) if total_pts > 0 else 0.0,
            "n_out": data["n_out"],
            "starters_out": data["starters_out"],
            "rotation_out": data["rotation_out"],
            "missing_min": data["missing_min"],
            "missing_pts": data["missing_pts"],
            "key_players": sorted(data["key_players"], key=lambda x: -x["mpg"]),
        }

    return result


# ── Main entry point ────────────────────────────────────────────────

def get_injury_features() -> dict:
    """
    Full pipeline: fetch injuries + player averages, compute impact per team.
    Returns dict: {team_abbr: {"missing_min_pct": float, "missing_pts_pct": float, ...}}
    """
    print("  Fetching injury report...")
    injuries = fetch_injury_report()
    if injuries.empty:
        return {}

    print("  Fetching player averages...")
    averages = fetch_player_averages()
    if averages.empty:
        return {}

    return compute_team_injury_impact(injuries, averages)


if __name__ == "__main__":
    print("NBA Predictor - Injury Impact")
    print("=" * 50)
    impact = get_injury_features()
    if not impact:
        print("No injury data available.")
    else:
        for team, data in sorted(impact.items(), key=lambda x: -x[1]["missing_min_pct"]):
            print(f"  {team}: {data['n_out']} OUT, "
                  f"{data['missing_min_pct']:.1%} min missing, "
                  f"{data['missing_pts_pct']:.1%} pts missing "
                  f"({data['missing_min']:.1f} MPG, {data['missing_pts']:.1f} PPG)")
