"""
Daily retrain script.
Fetches latest game results, rebuilds features, retrains models.

Usage:
    python retrain.py

Schedule with cron (e.g., every day at 10am):
    crontab -e
    0 10 * * * cd /Users/satishvattikuti/Desktop/aiprojects/nba-predictor && /Users/satishvattikuti/Desktop/aiprojects/nba_predictor/.venv/bin/python retrain.py >> data/retrain.log 2>&1
"""

import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.ingest import fetch_team_game_logs, fetch_advanced_team_stats, load_config
from src.features import build_game_feature_matrix_multi
from src.model import train_models, save_models


def main():
    print(f"\n{'='*50}")
    print(f"NBA Predictor - Daily Retrain")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")

    config = load_config()
    cache_dir = PROJECT_ROOT / config["cache_dir"]
    seasons = config["seasons"]
    current = config["current_season"]

    # Step 1: Fetch latest data (delete current season cache to force refresh)
    print("\n1. Fetching latest game data...")
    all_logs = []
    all_advs = []
    for season in seasons:
        log_path = cache_dir / f"game_logs_{season.replace('-', '_')}.parquet"
        adv_path = cache_dir / f"team_advanced_{season.replace('-', '_')}.parquet"

        # Force refresh current season
        if season == current:
            if log_path.exists():
                log_path.unlink()
            if adv_path.exists():
                adv_path.unlink()

        logs = fetch_team_game_logs(season, cache_dir)
        adv = fetch_advanced_team_stats(season, cache_dir)
        if logs is not None and not logs.empty:
            all_logs.append(logs)
        if adv is not None and not adv.empty:
            all_advs.append(adv)

    total_games = sum(len(l) for l in all_logs)
    print(f"   Total game log rows: {total_games}")

    # Step 2: Rebuild feature matrix
    print("\n2. Building feature matrix...")
    matrix_path = cache_dir / "feature_matrix.parquet"
    if matrix_path.exists():
        matrix_path.unlink()
    build_game_feature_matrix_multi(all_logs, all_advs)
    matrix = pd.read_parquet(matrix_path)
    print(f"   Matrix: {matrix.shape[0]} games, {matrix.shape[1]} columns")

    # Step 3: Retrain models
    print("\n3. Training models...")
    winner_model, spread_model, feature_cols, metrics = train_models(matrix)
    save_models(winner_model, spread_model, feature_cols, metrics)

    # Summary
    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Accuracy: {metrics['winner']['accuracy']:.1%}")
    print(f"  AUC-ROC:  {metrics['winner']['auc_roc']:.3f}")
    print(f"  MAE:      {metrics['spread']['mae']:.1f} pts")
    print(f"  Features: {metrics['n_features']}")
    print(f"  Train:    {metrics['train_size']} games")
    print(f"  Val:      {metrics['val_size']} games")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
