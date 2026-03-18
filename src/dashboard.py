"""
NBA Predictor - Streamlit Dashboard
Two tabs: Game Winner and Spread predictions.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yaml

from src.model import load_models, predict_today
from src.features import build_today_features
from src.odds import fetch_espn_odds
from src.injuries import get_injury_features
from src.ingest import TEAM_DISPLAY, load_config
from src.logger import log_predictions, load_prediction_log


# ── Page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="NBA Model Testing",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #1d428a, #c8102e);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .demo-banner {
        background: linear-gradient(135deg, #fff3cd, #ffeeba);
        border: 1px solid #ffc107; border-radius: 10px;
        padding: 12px 20px; margin-bottom: 20px;
    }
    .edge-positive {
        background: #d4edda; border-radius: 8px; padding: 12px;
        border-left: 4px solid #28a745; margin-bottom: 8px;
    }
    .edge-negative {
        background: #f8d7da; border-radius: 8px; padding: 12px;
        border-left: 4px solid #dc3545; margin-bottom: 8px;
    }

    /* ── Responsive ── */
    @media (max-width: 768px) {
        .main-header { font-size: 1.6rem; }
        /* Stack Streamlit columns vertically on mobile */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        /* Compact metrics */
        [data-testid="stMetric"] {
            padding: 8px 0;
        }
        /* Smaller tab text */
        [data-testid="stTabs"] button {
            font-size: 0.85rem;
            padding: 8px 10px;
        }
        /* Scrollable tables */
        [data-testid="stDataFrame"] {
            overflow-x: auto;
        }
        /* Cards */
        .edge-positive, .edge-negative {
            padding: 10px;
            font-size: 0.9rem;
        }
        .demo-banner {
            padding: 8px 12px;
            font-size: 0.9rem;
        }
    }
    @media (max-width: 480px) {
        .main-header { font-size: 1.3rem; }
        [data-testid="stTabs"] button {
            font-size: 0.75rem;
            padding: 6px 6px;
        }
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────

@st.cache_data(ttl=1800)
def load_all_data():
    from src.ingest import fetch_team_game_logs, fetch_advanced_team_stats, fetch_today_schedule

    config = load_config()
    cache_dir = PROJECT_ROOT / config["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    winner_model, spread_model, feature_cols, metrics = load_models()

    # Load current season game logs (fetch if missing)
    current = config["current_season"]
    current_key = current.replace("-", "_")
    log_path = cache_dir / f"game_logs_{current_key}.parquet"
    if not log_path.exists():
        fetch_team_game_logs(current, cache_dir)
    game_logs = pd.read_parquet(log_path)

    adv_path = cache_dir / f"team_advanced_{current_key}.parquet"
    if not adv_path.exists():
        fetch_advanced_team_stats(current, cache_dir)
    adv_stats = pd.read_parquet(adv_path) if adv_path.exists() else None

    today_schedule = fetch_today_schedule(cache_dir)

    # Build features for today
    today_features = build_today_features(today_schedule, game_logs, adv_stats)

    # Injury impact
    try:
        injury_impact = get_injury_features()
    except Exception as e:
        print(f"  Injury data unavailable: {e}")
        injury_impact = {}

    # Predictions (injury data is display-only, not adjusting model outputs)
    predictions = predict_today(today_features)

    # ESPN odds
    odds = fetch_espn_odds()

    # Merge odds into predictions
    if not odds.empty:
        predictions = predictions.merge(
            odds[["home_team", "away_team", "home_ml", "away_ml", "spread",
                  "over_under", "home_implied_prob", "away_implied_prob"]],
            on=["home_team", "away_team"],
            how="left",
        )
    else:
        predictions["home_ml"] = None
        predictions["away_ml"] = None
        predictions["spread"] = None
        predictions["over_under"] = None
        predictions["home_implied_prob"] = None
        predictions["away_implied_prob"] = None

    # Backfill missing spreads from prediction log (ESPN drops odds after tipoff)
    log_path = PROJECT_ROOT / config["cache_dir"] / "prediction_log.csv"
    if log_path.exists():
        from datetime import date
        log = pd.read_csv(log_path)
        today_log = log[log["prediction_date"] == str(date.today())]
        if not today_log.empty:
            for idx, row in predictions.iterrows():
                if pd.isna(row.get("spread")):
                    match = today_log[
                        (today_log["home_team"] == row["home_team"]) &
                        (today_log["away_team"] == row["away_team"]) &
                        (today_log["spread"].notna())
                    ]
                    if not match.empty:
                        predictions.at[idx, "spread"] = match.iloc[0]["spread"]

    # Compute edges
    predictions["edge"] = predictions.apply(
        lambda r: (r["home_win_prob"] - r["home_implied_prob"]) * 100
        if pd.notna(r.get("home_implied_prob")) else None,
        axis=1,
    )
    predictions["spread_diff"] = predictions.apply(
        lambda r: r["predicted_margin"] - (-r["spread"])
        if pd.notna(r.get("spread")) else None,
        axis=1,
    )

    # Log predictions
    log_predictions(predictions)

    return winner_model, spread_model, feature_cols, metrics, predictions, injury_impact


# ── Retrain ──────────────────────────────────────────────────────

def _run_retrain():
    """Fetch fresh game logs, rebuild feature matrix, retrain models."""
    from src.ingest import fetch_team_game_logs, fetch_advanced_team_stats
    from src.features import build_game_feature_matrix_multi
    from src.model import train_models, save_models

    config = load_config()
    cache_dir = PROJECT_ROOT / config["cache_dir"]
    seasons = config["seasons"]

    with st.spinner("Fetching latest game logs..."):
        all_logs = []
        all_advs = []
        for season in seasons:
            log_path = cache_dir / f"game_logs_{season.replace('-', '_')}.parquet"
            adv_path = cache_dir / f"team_advanced_{season.replace('-', '_')}.parquet"

            # Force refresh current season by deleting cache
            if season == config["current_season"]:
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

    with st.spinner("Building feature matrix..."):
        # Delete old matrix
        matrix_path = cache_dir / "feature_matrix.parquet"
        if matrix_path.exists():
            matrix_path.unlink()
        build_game_feature_matrix_multi(all_logs, all_advs)
        matrix = pd.read_parquet(matrix_path)

    with st.spinner("Training models..."):
        winner_model, spread_model, feature_cols, metrics = train_models(matrix)
        save_models(winner_model, spread_model, feature_cols, metrics)

    st.cache_data.clear()
    st.success(
        f"Retrained! {metrics['n_features']} features, "
        f"{metrics.get('winner', {}).get('accuracy', 0):.1%} accuracy, "
        f"{metrics.get('spread', {}).get('mae', 0):.1f} MAE"
    )
    st.rerun()


# ── Sidebar ──────────────────────────────────────────────────────────

def render_sidebar(metrics: dict):
    with st.sidebar:
        st.markdown("## 🏀 NBA Model Testing")
        st.markdown("---")

        st.markdown("### Winner Model")
        wm = metrics.get("winner", {})
        st.metric("Accuracy", f"{wm.get('accuracy', 0):.1%}")
        st.metric("AUC-ROC", f"{wm.get('auc_roc', 0):.3f}")

        st.markdown("### Spread Model")
        sm = metrics.get("spread", {})
        st.metric("MAE", f"{sm.get('mae', 0):.1f} pts")
        st.metric("RMSE", f"{sm.get('rmse', 0):.1f} pts")

        st.markdown("---")
        st.markdown(f"**Training:** {metrics.get('train_size', 'N/A')} games")
        st.markdown(f"**Validation:** {metrics.get('val_size', 'N/A')} games")
        st.markdown(f"**Features:** {metrics.get('n_features', 'N/A')}")

        st.markdown("---")
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### Retrain Model")
        st.caption("Fetches latest game logs, rebuilds features, and retrains both models.")
        if st.button("🧠 Retrain Now"):
            _run_retrain()


# ── Tab 1: Game Winner ──────────────────────────────────────────────

def render_winner_tab(predictions: pd.DataFrame, metrics: dict):
    # Edges callout
    has_odds = predictions["home_implied_prob"].notna().any()
    if has_odds:
        edges = predictions[predictions["edge"].abs() > 5].sort_values("edge", key=abs, ascending=False)
        if not edges.empty:
            st.markdown("#### 🎯 Value Picks (Edge > 5%)")
            cols = st.columns(min(len(edges), 4))
            for i, (_, row) in enumerate(edges.head(4).iterrows()):
                with cols[i]:
                    edge_val = row["edge"]
                    css_class = "edge-positive" if edge_val > 0 else "edge-negative"
                    team = row["home_display"] if edge_val > 0 else row["away_display"]
                    st.markdown(
                        f'<div class="{css_class}">'
                        f'<strong>{team}</strong><br>'
                        f'<small>{row["matchup"]}</small><br>'
                        f'Edge: <strong>{edge_val:+.1f}%</strong>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown("")

    # Table
    st.markdown("#### Game Predictions")
    display_df = predictions[["away_display", "home_display", "home_win_prob", "away_win_prob"]].copy()
    display_df.columns = ["Away", "Home", "Home Win %", "Away Win %"]
    display_df["Home Win %"] = pd.to_numeric(display_df["Home Win %"], errors="coerce").mul(100).round(1)
    display_df["Away Win %"] = pd.to_numeric(display_df["Away Win %"], errors="coerce").mul(100).round(1)

    if has_odds:
        display_df["Vegas Home %"] = (predictions["home_implied_prob"] * 100).round(1)
        display_df["Edge"] = predictions["edge"].round(1)

    st.dataframe(
        display_df,
        column_config={
            "Home Win %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
            "Away Win %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
        },
        hide_index=True,
        use_container_width=True,
    )

    # Bar chart
    st.markdown("#### Win Probability Chart")
    df_sorted = predictions.sort_values("home_win_prob", ascending=True)
    colors = [
        "#28a745" if p > 0.6 else "#dc3545" if p < 0.4 else "#ffc107"
        for p in df_sorted["home_win_prob"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sorted["matchup"], x=df_sorted["home_win_prob"],
        orientation="h", marker_color=colors,
        text=[f"{p:.0%}" for p in df_sorted["home_win_prob"]],
        textposition="auto",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="rgba(0,0,0,0.3)")
    fig.update_layout(
        xaxis_title="Home Win Probability", xaxis_range=[0, 1], xaxis_tickformat=".0%",
        height=max(300, len(df_sorted) * 55),
        margin=dict(l=10, r=10, t=10, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 2: Spread ───────────────────────────────────────────────────

def render_spread_tab(predictions: pd.DataFrame, metrics: dict, injury_impact: dict = None):
    has_spread = predictions["spread"].notna().any()

    # Injury alert
    if injury_impact:
        game_teams = set(predictions["home_team"].tolist() + predictions["away_team"].tolist())
        hurt_teams = {t: d for t, d in injury_impact.items()
                      if t in game_teams and d["missing_min_pct"] > 0.15}
        if hurt_teams:
            st.markdown("#### 🏥 Injury Situation (Today's Games)")
            cols = st.columns(min(len(hurt_teams), 4))
            for i, (team, data) in enumerate(
                sorted(hurt_teams.items(), key=lambda x: -x[1]["missing_min_pct"])[:4]
            ):
                with cols[i]:
                    display_name = TEAM_DISPLAY.get(team, team)
                    starters = data.get("starters_out", 0)
                    rotation = data.get("rotation_out", 0)
                    bench = data["n_out"] - starters - rotation
                    role_line = []
                    if starters:
                        role_line.append(f"{starters} starter{'s' if starters > 1 else ''}")
                    if rotation:
                        role_line.append(f"{rotation} rotation")
                    if bench:
                        role_line.append(f"{bench} bench")
                    role_str = ", ".join(role_line)
                    # Key player names
                    key_names = [p["name"].split()[-1] for p in data.get("key_players", [])[:3]]
                    names_str = ", ".join(key_names) if key_names else ""
                    st.markdown(
                        f'<div class="edge-negative">'
                        f'<strong>{display_name}</strong><br>'
                        f'{data["n_out"]} OUT ({role_str})<br>'
                        f'{data["missing_pts"]:.1f} PPG missing<br>'
                        f'<small>{names_str}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown("")

    # Spread differences callout
    if has_spread:
        big_diffs = predictions[predictions["spread_diff"].abs() > 3].sort_values(
            "spread_diff", key=abs, ascending=False
        )
        if not big_diffs.empty:
            st.markdown("#### 🔥 Spread Mismatches (Diff > 3 pts)")
            cols = st.columns(min(len(big_diffs), 4))
            for i, (_, row) in enumerate(big_diffs.head(4).iterrows()):
                with cols[i]:
                    diff = row["spread_diff"]
                    lean = row["home_display"] if diff > 0 else row["away_display"]
                    css_class = "edge-positive" if diff > 0 else "edge-negative"
                    # Model agreement: spread lean vs winner lean
                    spread_leans_home = diff > 0
                    winner_leans_home = row["home_win_prob"] > 0.5
                    agree = spread_leans_home == winner_leans_home
                    agree_icon = "✅" if agree else "⚠️"
                    agree_text = "Both models agree" if agree else "Models disagree"
                    st.markdown(
                        f'<div class="{css_class}">'
                        f'Lean <strong>{lean}</strong> {agree_icon}<br>'
                        f'<small>{row["matchup"]}</small><br>'
                        f'Vegas: {row["spread"]:+.1f} | Model: {row["predicted_margin"]:+.1f}<br>'
                        f'<small>{agree_text}</small>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            st.markdown("")

    # Table
    st.markdown("#### Spread Predictions")
    spread_df = predictions[["away_display", "home_display", "predicted_margin"]].copy()
    spread_df.columns = ["Away", "Home", "Predicted Margin"]
    spread_df["Predicted Margin"] = spread_df["Predicted Margin"].round(1)

    if has_spread:
        spread_df["Vegas Spread"] = pd.to_numeric(predictions["spread"], errors="coerce").round(1)
        spread_df["Difference"] = pd.to_numeric(predictions["spread_diff"], errors="coerce").round(1)
        spread_df["Lean"] = predictions.apply(
            lambda r: f"{'Home' if r['spread_diff'] > 0 else 'Away'} Cover"
            if pd.notna(r.get("spread_diff")) else "-",
            axis=1,
        ).values
        # Model agreement column
        def _agreement(r):
            if pd.isna(r.get("spread_diff")):
                return "-"
            spread_home = r["spread_diff"] > 0
            winner_home = r["home_win_prob"] > 0.5
            return "✅ Agree" if spread_home == winner_home else "⚠️ Split"
        spread_df["Models"] = predictions.apply(_agreement, axis=1).values

    st.dataframe(spread_df, hide_index=True, use_container_width=True)

    # Per-game injury details
    if injury_impact:
        inj_cache = PROJECT_ROOT / "data" / "injury_report.csv"
        player_cache = PROJECT_ROOT / "data" / "player_averages.parquet"
        if inj_cache.exists() and player_cache.exists():
            inj_df = pd.read_csv(inj_cache)
            player_df = pd.read_parquet(player_cache)
            game_teams = set(predictions["home_team"].tolist() + predictions["away_team"].tolist())
            game_injuries = inj_df[inj_df["team"].isin(game_teams)].copy()
            if not game_injuries.empty:
                # Merge player averages
                player_lookup = player_df.set_index(["PLAYER_NAME", "TEAM_ABBREVIATION"])[["MIN", "PTS", "GP"]].to_dict("index")
                rows = []
                for _, inj in game_injuries.iterrows():
                    key = (inj["player"], inj["team"])
                    pstats = player_lookup.get(key, {})
                    mpg = pstats.get('MIN', 0)
                    role = "Starter" if mpg >= 25 else "Rotation" if mpg >= 15 else "Bench"
                    rows.append({
                        "Team": TEAM_DISPLAY.get(inj["team"], inj["team"]),
                        "Player": inj["player"],
                        "Role": role,
                        "MPG": f"{mpg:.1f}",
                        "PPG": f"{pstats.get('PTS', 0):.1f}",
                        "Injury": f"{inj.get('injury_type', '')} - {inj.get('detail', '')}",
                    })
                if rows:
                    with st.expander("🏥 Injured Players (Today's Games)", expanded=False):
                        st.dataframe(
                            pd.DataFrame(rows).sort_values(["Team", "MPG"], ascending=[True, False]),
                            hide_index=True, use_container_width=True,
                        )

    # Margin chart
    st.markdown("#### Predicted Margin (+ = Home Favored)")
    df_sorted = predictions.sort_values("predicted_margin")

    colors = ["#1d428a" if m > 0 else "#c8102e" for m in df_sorted["predicted_margin"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sorted["matchup"], x=df_sorted["predicted_margin"],
        orientation="h", marker_color=colors,
        text=[f"{m:+.1f}" for m in df_sorted["predicted_margin"]],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="rgba(0,0,0,0.3)")
    fig.update_layout(
        xaxis_title="Predicted Point Margin",
        height=max(300, len(df_sorted) * 55),
        margin=dict(l=10, r=40, t=10, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 3: History ─────────────────────────────────────────────────

def render_history_tab():
    log = load_prediction_log()

    if log.empty:
        st.info("No prediction history yet. Predictions are logged each time the dashboard loads.")
        return

    # Date filter
    dates = sorted(log["prediction_date"].dt.date.unique(), reverse=True)
    selected = st.multiselect("Filter by date", dates, default=dates[:7])
    if selected:
        log = log[log["prediction_date"].dt.date.isin(selected)]

    if log.empty:
        st.warning("No predictions for selected dates.")
        return

    # Only games with spread data
    spread_log = log[log["spread"].notna()].copy()

    if spread_log.empty:
        st.warning("No spread picks yet. Spreads are captured when the dashboard loads before tipoff.")
        return

    # Summary metrics
    graded = spread_log[spread_log["spread_cover_correct"].notna()]
    wins = int(graded["spread_cover_correct"].sum()) if len(graded) else 0
    losses = len(graded) - wins
    pending = len(spread_log) - len(graded)

    c1, c2 = st.columns(2)
    if len(graded) > 0:
        c1.metric("Spread Record", f"{wins}-{losses} ({wins / len(graded):.0%})")
    else:
        c1.metric("Spread Record", "No results yet")
    c2.metric("Pending", pending)

    # Table - spread cover only
    display = spread_log[["prediction_date", "matchup"]].copy()
    display["Spread"] = spread_log["spread"].round(1)
    display["Lean"] = spread_log["spread_lean"]
    display["Edge (pts)"] = spread_log["spread_diff"].abs().round(1)

    # Confidence tier: flag extreme spreads where model is unreliable
    def _confidence_tier(row):
        spread_abs = abs(row["spread"]) if pd.notna(row["spread"]) else 0
        edge_abs = abs(row["spread_diff"]) if pd.notna(row["spread_diff"]) else 0
        if spread_abs > 15 and edge_abs > 11.5:
            return "⚠️"
        return ""
    display["Flag"] = spread_log.apply(_confidence_tier, axis=1)

    display["Score"] = spread_log.apply(
        lambda r: f"{int(r['away_score'])}-{int(r['home_score'])}"
        if pd.notna(r.get("home_score")) else "—", axis=1,
    )
    display["Cover"] = spread_log["spread_cover_correct"].map(
        {1.0: "✅", 0.0: "❌"}
    ).fillna("⏳")

    display.columns = ["Date", "Matchup", "Spread", "Lean", "Edge (pts)", "Flag", "Score", "Cover"]
    display["Date"] = display["Date"].dt.strftime("%Y-%m-%d")

    st.dataframe(display, hide_index=True, use_container_width=True)


# ── Model stats ──────────────────────────────────────────────────────

def render_model_stats(metrics: dict, winner_model, feature_cols):
    st.markdown("### Model Performance")

    st.markdown("#### Winner Model (Validation)")
    wm = metrics.get("winner", {})
    w1, w2, w3 = st.columns(3)
    w1.metric("Accuracy", f"{wm.get('accuracy', 0):.1%}",
              delta=f"{(wm.get('accuracy', 0) - 0.58) * 100:+.1f}% vs home baseline")
    w2.metric("AUC-ROC", f"{wm.get('auc_roc', 0):.3f}")
    w3.metric("Log Loss", f"{wm.get('log_loss', 0):.3f}")

    st.markdown("#### Spread Model (Validation)")
    sm = metrics.get("spread", {})
    s1, s2 = st.columns(2)
    s1.metric("MAE", f"{sm.get('mae', 0):.1f} pts")
    s2.metric("RMSE", f"{sm.get('rmse', 0):.1f} pts")

    # Feature importance
    st.markdown("#### Top Features")
    importance = winner_model.feature_importances_
    feat_imp = pd.DataFrame({
        "feature": feature_cols, "importance": importance,
    }).sort_values("importance", ascending=False).head(15)
    feat_imp["display"] = feat_imp["feature"].str.replace("_", " ").str.title()

    fig = px.bar(
        feat_imp.sort_values("importance", ascending=True),
        x="importance", y="display", orientation="h",
        color="importance", color_continuous_scale="Blues",
    )
    fig.update_layout(
        showlegend=False, coloraxis_showscale=False,
        xaxis_title="Importance", yaxis_title="",
        height=420, margin=dict(l=10, r=10, t=10, b=40),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    st.markdown('<p class="main-header">🏀 NBA Model Testing</p>', unsafe_allow_html=True)
    st.caption("XGBoost winner & spread predictions with ESPN odds")

    try:
        winner_model, spread_model, feature_cols, metrics, predictions, injury_impact = load_all_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Run `python src/ingest.py` and `python src/model.py` first.")
        return

    render_sidebar(metrics)

    # Demo banner
    if predictions["is_demo"].all():
        st.markdown(
            '<div class="demo-banner">'
            '⚠️ <strong>DEMO MODE</strong> — No live games. Showing sample matchups.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Game Winner", "📏 Spread", "📜 History", "📊 Model Stats"])

    with tab1:
        render_winner_tab(predictions, metrics)

    with tab2:
        render_spread_tab(predictions, metrics, injury_impact)

    with tab3:
        render_history_tab()

    with tab4:
        render_model_stats(metrics, winner_model, feature_cols)


if __name__ == "__main__":
    main()
