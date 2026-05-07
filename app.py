"""ATP Tennis Match Predictor — Streamlit dashboard.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

DATA_DIR  = Path("data/processed")
MODEL_DIR = DATA_DIR / "models"

st.set_page_config(
    page_title="ATP Tennis Predictor",
    page_icon="🎾",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached loaders — built once, reused across interactions
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading match data and building tracker…")
def load_data_and_tracker():
    import features as feat_mod
    raw_df = pd.read_parquet(DATA_DIR / "raw.parquet")
    raw_df = raw_df.sort_values("date").reset_index(drop=True)
    tracker  = feat_mod.build_tracker(raw_df)
    players  = feat_mod.known_players(raw_df)
    max_rank = float(pd.concat([raw_df["w_rank"], raw_df["l_rank"]]).dropna().max())
    return raw_df, tracker, players, max_rank


@st.cache_data(ttl=3600, show_spinner="Fetching live ATP rankings…")
def load_live_rank_map(players: tuple) -> dict:
    import atp_scraper
    return atp_scraper.get_live_rank_map(list(players))


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    import json
    import lightgbm as lgb
    booster          = lgb.Booster(model_file=str(MODEL_DIR / "model.lgb"))
    booster_no_odds  = lgb.Booster(model_file=str(MODEL_DIR / "model_no_odds.lgb"))
    feature_cols         = json.loads((MODEL_DIR / "feature_cols.json").read_text())
    feature_cols_no_odds = json.loads((MODEL_DIR / "feature_cols_no_odds.json").read_text())
    return booster, booster_no_odds, feature_cols, feature_cols_no_odds


# ---------------------------------------------------------------------------
# Kelly criterion helpers
# ---------------------------------------------------------------------------

def kelly_fraction(prob: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    return (b * prob - (1 - prob)) / b


def build_kelly_table(prob: float, offered_odds: float) -> pd.DataFrame:
    f  = kelly_fraction(prob, offered_odds)
    ev = (offered_odds - 1) * prob - (1 - prob)
    return pd.DataFrame([
        {"Strategy": "Full Kelly",         "Fraction": f"{f:.4f}",    "Stake / ₹100 bankroll": f"{f*100:.2f}",    "EV per unit": f"{ev:+.4f}", "Note": "Very volatile"},
        {"Strategy": "Half Kelly",         "Fraction": f"{f/2:.4f}",  "Stake / ₹100 bankroll": f"{f/2*100:.2f}",  "EV per unit": f"{ev:+.4f}", "Note": "Moderate risk"},
        {"Strategy": "Quarter Kelly ★",   "Fraction": f"{f/4:.4f}",  "Stake / ₹100 bankroll": f"{f/4*100:.2f}",  "EV per unit": f"{ev:+.4f}", "Note": "RECOMMENDED"},
    ]), f


# ---------------------------------------------------------------------------
# SHAP waterfall chart
# ---------------------------------------------------------------------------

def shap_waterfall(shap_df: pd.DataFrame, p1: str, prob: float, n: int = 15) -> go.Figure:
    top = shap_df.head(n).copy().sort_values("shap_value")  # ascending → positive bars point right

    def fmt_val(v):
        if pd.isna(v):
            return "NaN"
        return f"{v:.3g}"

    labels = [f"{row['feature']}  =  {fmt_val(row['feature_value'])}" for _, row in top.iterrows()]
    colors = ["#27ae60" if v > 0 else "#e74c3c" for v in top["shap_value"]]

    fig = go.Figure(go.Bar(
        x=top["shap_value"],
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in top["shap_value"]],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        title=dict(text=f"Why the model predicts <b>{prob:.1%}</b> for {p1}", font_size=15),
        xaxis_title="SHAP contribution (log-odds; green = helps P1, red = hurts P1)",
        yaxis_title="",
        height=520,
        margin=dict(l=10, r=60, t=50, b=20),
        plot_bgcolor="white",
        xaxis=dict(zeroline=True, zerolinecolor="#aaa", zerolinewidth=1.5, gridcolor="#eee"),
        yaxis=dict(tickfont=dict(size=12)),
    )
    return fig


# ---------------------------------------------------------------------------
# Player stat comparison
# ---------------------------------------------------------------------------

STAT_ROWS = [
    # (label,            p1_key,                  p2_key,                 higher_is_better)
    ("Elo",              "w_elo",                  "l_elo",                True),
    ("Surface Elo",      "w_elo_surf",             "l_elo_surf",           True),
    ("ATP Rank",         "w_rank",                 "l_rank",               False),
    ("Current streak",   "w_current_streak",       "l_current_streak",     True),
    ("Wins in last 5",   "w_streak_5",             "l_streak_5",           True),
    ("30d win rate",     "w_win_rate_30d",          "l_win_rate_30d",       True),
    ("90d surf win %",   "w_win_rate_surf_90d",     "l_win_rate_surf_90d",  True),
    ("Days since last",  "w_days_since_last",       "l_days_since_last",    False),
    ("H2H win rate",     "h2h_win_rate_w",          None,                   True),
]


def player_comparison_df(feat: dict, p1: str, p2: str) -> pd.DataFrame:
    def fmt(v, pct=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if pct:
            return f"{v:.1%}"
        return f"{v:.1f}"

    rows = []
    for label, p1_key, p2_key, higher_better in STAT_ROWS:
        v1 = feat.get(p1_key, np.nan)
        # H2H is stored from p1 perspective only; p2 side = 1 - v1
        if p2_key is None:
            v2 = 1.0 - v1 if pd.notna(v1) else np.nan
        else:
            v2 = feat.get(p2_key, np.nan)

        is_pct = "rate" in label.lower() or "h2h" in label.lower()
        s1, s2 = fmt(v1, is_pct), fmt(v2, is_pct)

        if pd.notna(v1) and pd.notna(v2):
            arrow = "→" if abs(v1 - v2) < 1e-6 else ("↑" if (higher_better and v1 > v2) or (not higher_better and v1 < v2) else "↓")
        else:
            arrow = "—"

        rows.append({"Stat": label, p1: s1, "": arrow, p2: s2})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.title("ATP Tennis Match Predictor")
    st.caption(
        "LightGBM model trained on 2000–2026 ATP data from tennis-data.co.uk.  "
        "No-odds CV AUC: **0.740** · With-odds CV AUC: **0.763** · Test holdout AUC: **0.753**"
    )

    # Load cached resources
    raw_df, tracker, players, max_rank = load_data_and_tracker()
    booster_odds, booster_no_odds, feature_cols, feature_cols_no_odds = load_models()

    # -----------------------------------------------------------------------
    # Sidebar — match inputs
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.header("Match Setup")

        default_p1 = players.index("Sinner J.") if "Sinner J." in players else 0
        default_p2 = players.index("Zverev A.") if "Zverev A." in players else 1

        p1 = st.selectbox("Player 1", players, index=default_p1)
        p2 = st.selectbox("Player 2", players, index=default_p2)
        surface = st.selectbox("Surface", ["Hard", "Clay", "Grass", "Carpet"])


        st.subheader("Bookmaker odds (optional)")
        with st.expander("Enter available odds"):
            st.caption("Leave blank if odds are unavailable — the no-odds model will be used.")
            b365w = st.number_input("Bet365 — P1 odds", min_value=1.01, value=None, placeholder="e.g. 1.65", format="%.2f")
            b365l = st.number_input("Bet365 — P2 odds", min_value=1.01, value=None, placeholder="e.g. 2.30", format="%.2f")
            psw   = st.number_input("Pinnacle — P1 odds", min_value=1.01, value=None, placeholder="optional", format="%.2f")
            psl   = st.number_input("Pinnacle — P2 odds", min_value=1.01, value=None, placeholder="optional", format="%.2f")
            maxw  = st.number_input("Max — P1 odds", min_value=1.01, value=None, placeholder="optional", format="%.2f")
            maxl  = st.number_input("Max — P2 odds", min_value=1.01, value=None, placeholder="optional", format="%.2f")

        run_btn = st.button("Run Prediction", type="primary", use_container_width=True)

    # -----------------------------------------------------------------------
    # Guard: wait for button press
    # -----------------------------------------------------------------------
    if not run_btn:
        st.info("Configure the match in the sidebar and click **Run Prediction**.")
        return

    if p1 == p2:
        st.error("Player 1 and Player 2 must be different.")
        return

    # -----------------------------------------------------------------------
    # Compute features
    # -----------------------------------------------------------------------
    import features as feat_mod
    import model as model_mod

    with st.spinner("Computing features…"):
        feat = feat_mod.get_live_features(tracker, p1, p2, surface)

        live_ranks = load_live_rank_map(tuple(players))
        p1_rank = float(live_ranks[p1]) if p1 in live_ranks else model_mod._get_last_rank(raw_df, p1)
        p2_rank = float(live_ranks[p2]) if p2 in live_ranks else model_mod._get_last_rank(raw_df, p2)
        feat.update(model_mod._compute_rank_features(p1_rank, p2_rank, max_rank))

        # Tournament context features (previously always NaN at inference)
        feat["is_grand_slam"] = 0.0
        feat["is_best_of_5"]  = 0.0
        feat["round_num"]     = float("nan")
        feat["surface_code"]  = float(feat_mod.SURFACE_MAP.get(surface, float("nan")))

        odds: dict = {}
        if b365w and b365l:
            odds["b365w"] = b365w
            odds["b365l"] = b365l
        if psw and psl:
            odds["psw"] = psw
            odds["psl"] = psl
        if maxw and maxl:
            odds["maxw"] = maxw
            odds["maxl"] = maxl

        # No-odds model (always run)
        X_no    = pd.DataFrame([{col: feat.get(col, np.nan) for col in feature_cols_no_odds}])
        prob_p1_no = float(booster_no_odds.predict(X_no)[0])

        if odds:
            feat.update(model_mod._compute_odds_features(odds))
            X_odds      = pd.DataFrame([{col: feat.get(col, np.nan) for col in feature_cols}])
            prob_p1_w   = float(booster_odds.predict(X_odds)[0])
            # Ensemble: 65% with-odds + 35% no-odds (same blend as predict_upcoming.py)
            prob_p1     = 0.65 * prob_p1_w + 0.35 * prob_p1_no
            booster     = booster_odds
            used_cols   = feature_cols
            model_label = f"Ensemble (65% with-odds {prob_p1_w:.1%} + 35% no-odds {prob_p1_no:.1%})"
        else:
            prob_p1     = prob_p1_no
            booster     = booster_no_odds
            used_cols   = feature_cols_no_odds
            model_label = "No-odds model"

        prob_p2 = 1.0 - prob_p1
        X       = pd.DataFrame([{col: feat.get(col, np.nan) for col in used_cols}])

    # -----------------------------------------------------------------------
    # Section 1 — Prediction
    # -----------------------------------------------------------------------
    st.header("Prediction")
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.metric(f"{p1} wins", f"{prob_p1:.1%}")
    with c2:
        st.metric(f"{p2} wins", f"{prob_p2:.1%}")
    with c3:
        st.info(f"Model: **{model_label}**")
        if psw and psl:
            raw_w, raw_l = 1 / psw, 1 / psl
            pin_prob = raw_w / (raw_w + raw_l)
            edge_pp  = (prob_p1 - pin_prob) * 100
            st.caption(f"Pinnacle implies **{pin_prob:.1%}** for {p1} — model edge: **{edge_pp:+.1f}pp**")

    # -----------------------------------------------------------------------
    # Section 2 — Kelly Criterion (only when odds entered)
    # -----------------------------------------------------------------------
    # Use Pinnacle odds for Kelly if available (sharp book), else fall back to Bet365
    kelly_odds_p1 = psw if psw else b365w
    kelly_odds_p2 = psl if psl else b365l

    if kelly_odds_p1 and kelly_odds_p2:
        st.divider()
        st.header("Kelly Criterion")
        odds_source = "Pinnacle" if psw else "Bet365"
        st.caption(f"Using **{odds_source}** odds for Kelly calculation.")

        f1  = kelly_fraction(prob_p1, kelly_odds_p1)
        f2  = kelly_fraction(prob_p2, kelly_odds_p2)
        ev1 = (kelly_odds_p1 - 1) * prob_p1 - (1 - prob_p1)
        ev2 = (kelly_odds_p2 - 1) * prob_p2 - (1 - prob_p2)
        # Edge vs vig-free probability (margin-removed), same as predict_upcoming.py
        _rw, _rl = 1 / kelly_odds_p1, 1 / kelly_odds_p2
        _tot = _rw + _rl
        vf_p1, vf_p2 = _rw / _tot, _rl / _tot
        e1  = prob_p1 - vf_p1
        e2  = prob_p2 - vf_p2

        # Decision engine thresholds (matching backtest.py)
        MIN_EDGE_APP = 0.03
        MIN_CONF_APP = 0.70
        MIN_ODDS_APP = 1.20

        def _qualifies(prob, odds, edge_val):
            return edge_val >= MIN_EDGE_APP and prob >= MIN_CONF_APP and odds >= MIN_ODDS_APP

        if f1 <= 0 and f2 <= 0:
            st.warning(
                f"**No edge on either player.** "
                f"Model says {p1}: {prob_p1:.1%}, Pinnacle vig-free: {vf_p1:.1%}. "
                f"Model says {p2}: {prob_p2:.1%}, Pinnacle vig-free: {vf_p2:.1%}."
            )
        else:
            bet_player = p1 if f1 >= f2 else p2
            bet_prob   = prob_p1 if f1 >= f2 else prob_p2
            bet_odds   = kelly_odds_p1 if f1 >= f2 else kelly_odds_p2
            bet_edge   = e1 if f1 >= f2 else e2
            bet_f      = max(f1, f2)

            qualifies = _qualifies(bet_prob, bet_odds, bet_edge)

            k1, k2, k3 = st.columns(3)
            with k1:
                st.metric("Edge vs sharp odds", f"{bet_edge*100:+.2f}pp")
            with k2:
                st.metric("Expected value (per unit)", f"{(ev1 if f1 >= f2 else ev2):+.4f}")
            with k3:
                st.metric("Quarter Kelly stake", f"{bet_f/4*100:.2f}% of bankroll")

            if qualifies:
                st.success(
                    f"**BET {bet_player}** — edge {bet_edge:.1%} ≥ 3%, confidence {bet_prob:.1%} ≥ 70%, "
                    f"odds {bet_odds:.2f} ≥ 1.20. Quarter Kelly: **{bet_f/4*100:.2f}%** of bankroll."
                )
            else:
                reasons = []
                if bet_edge < MIN_EDGE_APP: reasons.append(f"edge {bet_edge:.1%} < 3%")
                if bet_prob < MIN_CONF_APP: reasons.append(f"confidence {bet_prob:.1%} < 70%")
                if bet_odds < MIN_ODDS_APP: reasons.append(f"odds {bet_odds:.2f} < 1.20")
                st.warning(f"**SKIP** — {', '.join(reasons)}.")

            kelly_df, _ = build_kelly_table(bet_prob, bet_odds)
            st.dataframe(kelly_df, hide_index=True, use_container_width=True)

            st.caption(
                "★ **Quarter Kelly is the recommended stake.** "
                "Full Kelly is mathematically optimal but one wrong calibration destroys your bankroll."
            )

    # -----------------------------------------------------------------------
    # Section 3 — Explainable AI
    # -----------------------------------------------------------------------
    st.divider()
    st.header("Explainable AI")

    tab1, tab2 = st.tabs(["SHAP Feature Contributions", "Player Comparison"])

    with tab1:
        shap_df = model_mod.explain_prediction(booster, X, used_cols)
        fig = shap_waterfall(shap_df, p1, prob_p1)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Each bar shows a single feature's contribution to the prediction in log-odds space. "
            "Green bars push the probability higher for **P1**; red bars push it lower. "
            "The feature value is shown next to the feature name."
        )

        with st.expander("Full SHAP table (all features)"):
            display = shap_df.copy()
            display["shap_value"]    = display["shap_value"].round(6)
            display["feature_value"] = display["feature_value"].apply(
                lambda v: f"{v:.5g}" if pd.notna(v) else "NaN"
            )
            st.dataframe(display, hide_index=True, use_container_width=True)

    with tab2:
        st.subheader(f"{p1}  vs  {p2}  —  {surface}")
        comp_df = player_comparison_df(feat, p1, p2)
        st.dataframe(comp_df, hide_index=True, use_container_width=True)
        st.caption(
            "↑ = P1 has the better value · ↓ = P2 has the better value · → = equal. "
            "For ATP Rank and Days since last, lower is better."
        )


if __name__ == "__main__":
    main()
