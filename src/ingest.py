"""
NBA Predictor - Data Ingestion
Pulls game logs and team stats from nba_api, today's schedule from live endpoints.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_config() -> dict:
    with open(PROJECT_ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


# ── Team mappings ────────────────────────────────────────────────────

TEAM_ABBREVS = [
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN",
    "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA",
    "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
    "POR", "SAC", "SAS", "TOR", "UTA", "WAS",
]

TEAM_FULL_NAMES = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

TEAM_DISPLAY = {
    "ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets",
    "CHI": "Bulls", "CLE": "Cavaliers", "DAL": "Mavericks", "DEN": "Nuggets",
    "DET": "Pistons", "GSW": "Warriors", "HOU": "Rockets", "IND": "Pacers",
    "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies", "MIA": "Heat",
    "MIL": "Bucks", "MIN": "Timberwolves", "NOP": "Pelicans", "NYK": "Knicks",
    "OKC": "Thunder", "ORL": "Magic", "PHI": "76ers", "PHX": "Suns",
    "POR": "Trail Blazers", "SAC": "Kings", "SAS": "Spurs", "TOR": "Raptors",
    "UTA": "Jazz", "WAS": "Wizards",
}

# nba_api uses TEAM_ID -> abbreviation
_NBA_API_ABBREV_FIXES = {
    "PHO": "PHX",  # nba_api sometimes uses PHO for Phoenix
    "GS": "GSW",
    "SA": "SAS",
    "NY": "NYK",
    "NO": "NOP",
    "UTAH": "UTA",
    "WSH": "WAS",
    "BKN": "BKN",
    "NJN": "BKN",
    "CHA": "CHA",
    "CHO": "CHA",
}

# Full name -> abbreviation (for nba_api which returns full names)
_FULL_NAME_TO_ABBREV = {}
for abbr, full in TEAM_FULL_NAMES.items():
    _FULL_NAME_TO_ABBREV[full] = abbr
    # Also map nickname only
    nickname = full.split()[-1] if len(full.split()) > 1 else full
    _FULL_NAME_TO_ABBREV[nickname] = abbr
# Special cases
_FULL_NAME_TO_ABBREV["Trail Blazers"] = "POR"
_FULL_NAME_TO_ABBREV["76ers"] = "PHI"


def normalize_team(name: str) -> str:
    """Map any team name/abbreviation to canonical 3-letter code."""
    name = name.strip()
    if name in TEAM_ABBREVS:
        return name
    if name in _NBA_API_ABBREV_FIXES:
        return _NBA_API_ABBREV_FIXES[name]
    if name in _FULL_NAME_TO_ABBREV:
        return _FULL_NAME_TO_ABBREV[name]
    # Substring match
    for full, abbr in _FULL_NAME_TO_ABBREV.items():
        if name in full or full in name:
            return abbr
    raise ValueError(f"Unknown NBA team: {name}")


# ── Caching ──────────────────────────────────────────────────────────

def _load_or_fetch(cache_path: Path, fetch_fn, max_age_hours: int = 24):
    """Load from cache if fresh, otherwise fetch."""
    if cache_path.exists():
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours < max_age_hours:
            print(f"  Loading cached: {cache_path.name}")
            if cache_path.suffix == ".parquet":
                return pd.read_parquet(cache_path)
            return pd.read_csv(cache_path)

    print(f"  Fetching: {cache_path.name}")
    df = fetch_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.suffix == ".parquet":
        df.to_parquet(cache_path, index=False)
    else:
        df.to_csv(cache_path, index=False)
    return df


# ── Data fetchers ────────────────────────────────────────────────────

def _season_to_nba_api(season_str: str) -> str:
    """Convert '2023-24' to nba_api format '2023-24'."""
    return season_str


def fetch_team_game_logs(season: str, cache_dir: Path) -> pd.DataFrame:
    """Fetch team game logs for a season using LeagueGameLog."""
    cache_path = cache_dir / f"game_logs_{season.replace('-', '_')}.parquet"
    return _load_or_fetch(cache_path, lambda: _fetch_game_logs(season), max_age_hours=12)


def _fetch_game_logs(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import LeagueGameLog

    print(f"    Fetching game logs for {season}...")
    time.sleep(0.6)
    result = LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T",
    )
    df = result.get_data_frames()[0]
    time.sleep(0.6)

    # Standardize columns
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Parse matchup: e.g., "BOS vs. NYK" (home) or "BOS @ NYK" (away)
    df["is_home"] = df["MATCHUP"].apply(lambda x: 1 if "vs." in str(x) else 0)

    # Extract opponent
    df["OPP"] = df["MATCHUP"].apply(
        lambda x: str(x).split("vs.")[-1].strip() if "vs." in str(x)
        else str(x).split("@")[-1].strip()
    )

    # Normalize team abbreviations
    df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].apply(
        lambda x: _NBA_API_ABBREV_FIXES.get(x, x)
    )
    df["OPP"] = df["OPP"].apply(
        lambda x: _NBA_API_ABBREV_FIXES.get(x.strip(), x.strip())
    )

    # Win flag
    df["WIN"] = (df["WL"] == "W").astype(int)

    return df


def fetch_advanced_team_stats(season: str, cache_dir: Path) -> pd.DataFrame:
    """Fetch team advanced stats (off/def rating, pace, etc.)."""
    cache_path = cache_dir / f"team_advanced_{season.replace('-', '_')}.parquet"
    return _load_or_fetch(cache_path, lambda: _fetch_advanced(season), max_age_hours=24)


def _fetch_advanced(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import TeamEstimatedMetrics

    print(f"    Fetching advanced stats for {season}...")
    time.sleep(0.6)
    result = TeamEstimatedMetrics(season=season)
    df = result.get_data_frames()[0]
    time.sleep(0.6)

    # Normalize team abbreviations
    if "TEAM_ABBREVIATION" in df.columns:
        df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].apply(
            lambda x: _NBA_API_ABBREV_FIXES.get(x, x)
        )

    return df


def fetch_today_schedule(cache_dir: Path) -> pd.DataFrame:
    """Fetch today's NBA games."""
    cache_path = cache_dir / "today_schedule.csv"

    # Always refresh today's schedule
    try:
        return _load_or_fetch(cache_path, _fetch_today_games, max_age_hours=1)
    except Exception as e:
        print(f"  Error fetching today's schedule: {e}")
        return _build_demo_schedule(cache_path)


def _fetch_today_games() -> pd.DataFrame:
    """Fetch today's games using ESPN scoreboard (more reliable than nba_api live)."""
    import httpx

    print("    Fetching today's NBA scoreboard from ESPN...")
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    resp = httpx.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    _espn_fix = {
        "WSH": "WAS", "GS": "GSW", "SA": "SAS", "NY": "NYK",
        "NO": "NOP", "UTAH": "UTA", "PHO": "PHX",
    }

    rows = []
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        home_info = away_info = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home_info = c
            elif c.get("homeAway") == "away":
                away_info = c

        if not home_info or not away_info:
            continue

        home_abbr = home_info.get("team", {}).get("abbreviation", "")
        away_abbr = away_info.get("team", {}).get("abbreviation", "")
        home_abbr = _espn_fix.get(home_abbr, home_abbr)
        away_abbr = _espn_fix.get(away_abbr, away_abbr)

        rows.append({
            "game_id": event.get("id", ""),
            "home_team": home_abbr,
            "away_team": away_abbr,
            "home_name": home_info.get("team", {}).get("displayName", ""),
            "away_name": away_info.get("team", {}).get("displayName", ""),
            "status": comp.get("status", {}).get("type", {}).get("shortDetail", ""),
            "is_demo": False,
        })

    if not rows:
        print("    No games today. Using demo schedule.")
        return _build_demo_schedule_df()

    return pd.DataFrame(rows)


def _build_demo_schedule(cache_path: Path) -> pd.DataFrame:
    df = _build_demo_schedule_df()
    df.to_csv(cache_path, index=False)
    return df


def _build_demo_schedule_df() -> pd.DataFrame:
    """Sample games for demo mode."""
    return pd.DataFrame([
        {"game_id": "demo1", "home_team": "BOS", "away_team": "LAL", "home_name": "Celtics", "away_name": "Lakers", "status": "Demo", "is_demo": True},
        {"game_id": "demo2", "home_team": "GSW", "away_team": "MIL", "home_name": "Warriors", "away_name": "Bucks", "status": "Demo", "is_demo": True},
        {"game_id": "demo3", "home_team": "DEN", "away_team": "PHX", "home_name": "Nuggets", "away_name": "Suns", "status": "Demo", "is_demo": True},
        {"game_id": "demo4", "home_team": "NYK", "away_team": "PHI", "home_name": "Knicks", "away_name": "76ers", "status": "Demo", "is_demo": True},
        {"game_id": "demo5", "home_team": "DAL", "away_team": "MIA", "home_name": "Mavericks", "away_name": "Heat", "status": "Demo", "is_demo": True},
        {"game_id": "demo6", "home_team": "CLE", "away_team": "OKC", "home_name": "Cavaliers", "away_name": "Thunder", "status": "Demo", "is_demo": True},
    ])


# ── Main ─────────────────────────────────────────────────────────────

def run_ingestion():
    """Main entry point."""
    config = load_config()
    cache_dir = PROJECT_ROOT / config["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("NBA Predictor - Data Ingestion")
    print("=" * 50)

    seasons = config["seasons"]
    for i, season in enumerate(seasons):
        print(f"\n{i+1}. Fetching {season} game logs...")
        logs = fetch_team_game_logs(season, cache_dir)
        print(f"   Got {len(logs)} entries")

        print(f"   Fetching {season} advanced stats...")
        adv = fetch_advanced_team_stats(season, cache_dir)
        print(f"   Got {len(adv)} teams")

    # Today's schedule
    print(f"\n{len(seasons)+1}. Fetching today's schedule...")
    today = fetch_today_schedule(cache_dir)
    print(f"   Got {len(today)} games")
    if today["is_demo"].all():
        print("   (DEMO MODE)")

    print(f"\nIngestion complete! Data cached in: {cache_dir}")


if __name__ == "__main__":
    run_ingestion()
