"""Run model predictions on today's upcoming ATP singles matches.

Usage:
    uv run python predict_upcoming.py
    uv run python predict_upcoming.py --days 3
    uv run python predict_upcoming.py --date 2026-04-30
"""

import argparse
import datetime
import json
import os
import sys
import unicodedata
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent / "src"))

DATA_DIR  = Path("data/processed")
MODEL_DIR = DATA_DIR / "models"

# Betting decision engine thresholds — overridden at startup from
# bet_thresholds.json if present (produced by `python main.py tune-thresholds`).
# Key finding from test-holdout backtest: MIN_ODDS=1.40 is the primary filter;
# below 1.40 the market is too efficient and Pinnacle vig eats any edge.
MIN_EDGE        = 0.03   # must have ≥ 3% edge over vig-free Pinnacle prob
MIN_CONFIDENCE  = 0.70   # model probability must be ≥ 70%
MIN_ODDS        = 1.40   # ← was 1.20; below 1.40 ROI is consistently negative
KELLY_FRACTION  = 0.25   # quarter Kelly (conservative, recommended for live use)


def _load_thresholds() -> tuple[float, float, float]:
    """Read MIN_EDGE/MIN_CONFIDENCE/MIN_ODDS from bet_thresholds.json if available."""
    path = MODEL_DIR / "bet_thresholds.json"
    if path.exists():
        try:
            t = json.loads(path.read_text())
            return (float(t.get("min_edge",      MIN_EDGE)),
                    float(t.get("min_conf",       MIN_CONFIDENCE)),
                    float(t.get("min_odds",       MIN_ODDS)))
        except Exception:
            pass
    return MIN_EDGE, MIN_CONFIDENCE, MIN_ODDS

# Ensemble weight: with-odds vs no-odds. Reads from ensemble_weight.json if a
# grid-search has tuned it; otherwise falls back to the default the model was
# validated at. Keep this consistent with src/model.py predict_ensemble().
DEFAULT_ENSEMBLE_W = 0.65

def _load_ensemble_weight() -> float:
    path = MODEL_DIR / "ensemble_weight.json"
    if path.exists():
        try:
            return float(json.loads(path.read_text()).get("weight_odds", DEFAULT_ENSEMBLE_W))
        except Exception:
            pass
    return DEFAULT_ENSEMBLE_W

SURFACE_MAP = {
    "red clay":    "Clay",
    "clay":        "Clay",
    "hard":        "Hard",
    "grass":       "Grass",
    "carpet":      "Carpet",
    "indoor hard": "Hard",
    "outdoor hard":"Hard",
}

# The Odds API sport keys for ATP tournaments
_ATP_SPORT_KEYS = [
    "tennis_atp_french_open",
    "tennis_atp_wimbledon",
    "tennis_atp_us_open",
    "tennis_atp_australian_open",
    "tennis_atp_madrid_open",
    "tennis_atp_rome",
    "tennis_atp_indian_wells",
    "tennis_atp_miami_open",
    "tennis_atp_canadian_open",
    "tennis_atp_cincinnati",
    "tennis_atp",
]


def normalise_surface(raw: str) -> str:
    return SURFACE_MAP.get(str(raw).strip().lower(), "Hard")


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


# ── Live Rankings ────────────────────────────────────────────────────────────

def fetch_live_ranks(known: list[str], top_n: int = 200) -> tuple[dict[str, int], bool]:
    """Return ({internal_name: live_rank}, live_ok). live_ok is False if the
    fetch failed and the caller will be using stale historical ranks."""
    from atp_scraper import get_live_rank_map
    try:
        print("Fetching live ATP rankings...")
        rank_map = get_live_rank_map(known, top_n=top_n)
        print(f"  Got {len(rank_map)} live ranks.")
        if not rank_map:
            print("  ⚠️  WARNING: live ranking API returned 0 players — predictions will use stale historical ranks.")
            return {}, False
        return rank_map, True
    except Exception as e:
        print(f"  ⚠️  WARNING: live rankings fetch FAILED ({e}). Predictions will use stale historical ranks.")
        return {}, False


# ── Odds API ─────────────────────────────────────────────────────────────────

def fetch_live_odds(api_key: str) -> dict[tuple[str, str], dict]:
    """
    Return odds keyed by (home_team_lower, away_team_lower) from The Odds API.
    Values: {psw, psl} (Pinnacle decimal odds, winner/loser from home perspective).
    """
    if not api_key:
        return {}

    base = "https://api.the-odds-api.com/v4"
    all_odds: dict[tuple[str, str], dict] = {}

    # Find active ATP sport keys
    try:
        r = requests.get(f"{base}/sports", params={"apiKey": api_key}, timeout=10)
        r.raise_for_status()
        active_keys = [s["key"] for s in r.json() if s.get("active") and "tennis_atp" in s["key"]]
    except Exception as e:
        print(f"  Warning: could not fetch sport list ({e}).")
        return {}

    print(f"  Active ATP sport keys: {active_keys}")

    for sport_key in active_keys:
        try:
            r = requests.get(
                f"{base}/sports/{sport_key}/odds",
                params={
                    "apiKey":      api_key,
                    "regions":     "eu",
                    "markets":     "h2h",
                    "oddsFormat":  "decimal",
                    "bookmakers":  "pinnacle,bet365",
                },
                timeout=10,
            )
            r.raise_for_status()
            for match in r.json():
                home = match["home_team"].lower()
                away = match["away_team"].lower()
                entry = all_odds.get((home, away), {})
                for bk in match.get("bookmakers", []):
                    for mkt in bk.get("markets", []):
                        if mkt["key"] != "h2h":
                            continue
                        outcome_map = {o["name"].lower(): o["price"] for o in mkt["outcomes"]}
                        if bk["key"] == "pinnacle":
                            entry["psw"] = outcome_map.get(home)
                            entry["psl"] = outcome_map.get(away)
                        elif bk["key"] == "bet365":
                            entry["b365w"] = outcome_map.get(home)
                            entry["b365l"] = outcome_map.get(away)
                all_odds[(home, away)] = entry
        except Exception as e:
            print(f"  Warning: odds fetch failed for {sport_key} ({e}).")

    print(f"  Fetched odds for {len(all_odds)} matches.")
    return all_odds


def _match_odds(odds_map: dict, p1_raw: str, p2_raw: str) -> dict | None:
    """Try to find odds for a player pair using last-name fuzzy matching."""
    p1_last = strip_accents(p1_raw.split()[-1]).lower()
    p2_last = strip_accents(p2_raw.split()[-1]).lower()

    for (home, away), odds in odds_map.items():
        home_last = home.split()[-1]
        away_last = away.split()[-1]
        if p1_last in home_last and p2_last in away_last:
            return {
                "psw": odds.get("psw"), "psl": odds.get("psl"),
                "b365w": odds.get("b365w"), "b365l": odds.get("b365l"),
            }
        if p2_last in home_last and p1_last in away_last:
            # Swap so p1=home perspective flipped
            return {
                "psw": odds.get("psl"),   "psl": odds.get("psw"),
                "b365w": odds.get("b365l"), "b365l": odds.get("b365w"),
            }
    return None


# ── Kelly Criterion ───────────────────────────────────────────────────────────

def kelly(prob: float, decimal_odds: float, fraction: float = KELLY_FRACTION) -> float:
    """Quarter Kelly by default. Pass fraction=1.0 for full Kelly."""
    b = decimal_odds - 1.0
    q = 1.0 - prob
    k = (b * prob - q) / b
    return round(k * fraction, 4)


def _vig_free_prob(odds_w: float, odds_l: float) -> tuple[float, float]:
    """Return vig-free (margin-removed) probabilities from decimal odds pair."""
    raw_w, raw_l = 1 / odds_w, 1 / odds_l
    total = raw_w + raw_l
    return raw_w / total, raw_l / total


def _kelly_fractions(prob: float, decimal_odds: float) -> dict:
    """Compute full, half, and quarter Kelly fractions (floored at 0)."""
    b = decimal_odds - 1.0
    q = 1.0 - prob
    full = max((b * prob - q) / b, 0.0)
    return {"full": round(full, 4), "half": round(full / 2, 4), "quarter": round(full / 4, 4)}


def _bet_rec(prob_p1: float, prob_p2: float, p1_odds: float | None, p2_odds: float | None,
             p1_name: str, p2_name: str) -> dict:
    """Apply decision engine. Returns a dict with kelly fractions and recommendation."""
    empty = {
        "kelly_p1": {"full": 0.0, "half": 0.0, "quarter": 0.0},
        "kelly_p2": {"full": 0.0, "half": 0.0, "quarter": 0.0},
        "rec": "No odds available",
        "qualifies": False,
        "bet_on": None,
        "edge": 0.0,
    }
    if not p1_odds or not p2_odds:
        return empty

    vf_p1, vf_p2 = _vig_free_prob(p1_odds, p2_odds)
    e1 = prob_p1 - vf_p1
    e2 = prob_p2 - vf_p2

    kf1 = _kelly_fractions(prob_p1, p1_odds)
    kf2 = _kelly_fractions(prob_p2, p2_odds)

    def _qualifies(prob: float, odds: float, edge_val: float) -> bool:
        return edge_val >= MIN_EDGE and prob >= MIN_CONFIDENCE and odds >= MIN_ODDS

    q1 = _qualifies(prob_p1, p1_odds, e1)
    q2 = _qualifies(prob_p2, p2_odds, e2)

    # Break-even win rate: the minimum win% needed to profit at these odds
    be1 = 1.0 / p1_odds if p1_odds > 0 else 1.0
    be2 = 1.0 / p2_odds if p2_odds > 0 else 1.0

    if q1 and kf1["quarter"] >= kf2["quarter"]:
        margin1 = prob_p1 - be1
        rec = (f"BET {p1_name} — "
               f"QKelly {kf1['quarter']:.2%} | HKelly {kf1['half']:.2%} | "
               f"FKelly {kf1['full']:.2%}  "
               f"(edge vs Pinnacle {e1:+.1%} | margin over break-even {margin1:+.1%})")
        return {"kelly_p1": kf1, "kelly_p2": kf2, "rec": rec,
                "qualifies": True, "bet_on": "p1", "edge": e1, "break_even": be1}
    elif q2 and kf2["quarter"] > kf1["quarter"]:
        margin2 = prob_p2 - be2
        rec = (f"BET {p2_name} — "
               f"QKelly {kf2['quarter']:.2%} | HKelly {kf2['half']:.2%} | "
               f"FKelly {kf2['full']:.2%}  "
               f"(edge vs Pinnacle {e2:+.1%} | margin over break-even {margin2:+.1%})")
        return {"kelly_p1": kf1, "kelly_p2": kf2, "rec": rec,
                "qualifies": True, "bet_on": "p2", "edge": e2, "break_even": be2}
    else:
        reasons = []
        best_prob = max(prob_p1, prob_p2)
        best_edge = max(e1, e2)
        best_odds = p1_odds if prob_p1 >= prob_p2 else p2_odds
        if best_edge < MIN_EDGE:
            reasons.append(f"edge {best_edge:+.1%} < {MIN_EDGE:.0%}")
        if best_prob < MIN_CONFIDENCE:
            reasons.append(f"conf {best_prob:.1%} < {MIN_CONFIDENCE:.0%}")
        if best_odds < MIN_ODDS:
            reasons.append(f"odds {best_odds:.2f} < {MIN_ODDS:.2f} (short-price filter)")
        rec = "SKIP — " + ", ".join(reasons) if reasons else "SKIP — no qualifying edge"
        return {"kelly_p1": kf1, "kelly_p2": kf2, "rec": rec,
                "qualifies": False, "bet_on": None, "edge": max(e1, e2), "break_even": None}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict upcoming ATP matches")
    parser.add_argument("--days",     type=int,   default=2,    help="Days ahead to fetch (default 2)")
    parser.add_argument("--date",     type=str,   default=None, help="Filter to specific date e.g. 2026-04-30")
    parser.add_argument("--excel",    action="store_true",      help="Save output as Excel (.xlsx)")
    parser.add_argument("--no-odds",  action="store_true",      help="Skip odds API (no-odds model only)")
    parser.add_argument("--bankroll", type=float, default=100,  help="Bankroll in $ for stake calculation (default $100)")
    parser.add_argument("--allow-stale-ranks", action="store_true",
                        help="Proceed even if live ATP rank fetch fails (uses historical ranks).")
    parser.add_argument("--log-unknown", type=str, default="logs/unknown_players.csv",
                        help="CSV file to append unknown-player encounters to.")
    args = parser.parse_args()

    # ── 1. Fetch upcoming matches ─────────────────────────────────────────────
    print(f"Fetching upcoming ATP matches (next {args.days} days)...")
    from atp_scraper import fetch_atp_upcoming, match_to_internal

    try:
        upcoming = fetch_atp_upcoming(days_ahead=args.days)
    except Exception as e:
        print(f"Error fetching matches: {e}")
        return

    singles = upcoming[~upcoming["tournament"].str.contains("Doubles", case=False, na=False)].copy()
    singles = singles[singles["status"] == "Not started"].copy()

    if args.date:
        target = datetime.date.fromisoformat(args.date)
        singles = singles[singles["date"] == target].reset_index(drop=True)

    singles = (
        singles
        .sort_values("date")
        .drop_duplicates(subset=["player1", "player2"], keep="first")
        .reset_index(drop=True)
    )

    if singles.empty:
        print("No upcoming singles matches found.")
        return

    print(f"\nFound {len(singles)} upcoming singles matches.\n")

    # ── 2. Load model + historical data ──────────────────────────────────────
    print("Loading model and historical data...")
    import features as feat_mod
    import model as model_mod

    raw_df   = pd.read_parquet(DATA_DIR / "raw.parquet")
    raw_df   = raw_df.sort_values("date").reset_index(drop=True)
    tracker  = feat_mod.build_tracker(raw_df)
    known    = feat_mod.known_players(raw_df)
    max_rank = float(pd.concat([raw_df["w_rank"], raw_df["l_rank"]]).dropna().max())

    # No-odds model (always loaded)
    booster_no_odds   = lgb.Booster(model_file=str(MODEL_DIR / "model_no_odds.lgb"))
    feat_cols_no_odds = json.loads((MODEL_DIR / "feature_cols_no_odds.json").read_text())

    # Odds model (loaded if available)
    odds_model_path = MODEL_DIR / "model.lgb"
    if odds_model_path.exists() and not args.no_odds:
        booster_odds   = lgb.Booster(model_file=str(odds_model_path))
        feat_cols_odds = json.loads((MODEL_DIR / "feature_cols.json").read_text())
    else:
        booster_odds = None
        feat_cols_odds = []

    print("Model loaded.\n")

    # ── 3. Live rankings ──────────────────────────────────────────────────────
    live_ranks, live_ranks_ok = fetch_live_ranks(known)
    if not live_ranks_ok and not args.allow_stale_ranks:
        print("\nAborting: refusing to predict with stale historical ranks. "
              "Pass --allow-stale-ranks to override (predictions will be marked).")
        return

    # ── 4. Live odds ──────────────────────────────────────────────────────────
    api_key = os.getenv("ODDS_API_KEY", "")
    live_odds_map: dict = {}
    if not args.no_odds:
        if not api_key:
            print("ODDS_API_KEY not set — skipping live odds (no-odds model only).")
        else:
            print("Fetching live odds from The Odds API...")
            live_odds_map = fetch_live_odds(api_key)
            print()

    # ── 4b. Ensemble weight, thresholds + optional probability calibrator ─────
    ENSEMBLE_W = _load_ensemble_weight()
    _edge, _conf, _odds = _load_thresholds()
    # Override module-level constants so _bet_rec() picks them up
    global MIN_EDGE, MIN_CONFIDENCE, MIN_ODDS
    MIN_EDGE, MIN_CONFIDENCE, MIN_ODDS = _edge, _conf, _odds
    print(f"Ensemble weight (with-odds): {ENSEMBLE_W:.3f}")
    print(f"Thresholds: edge≥{MIN_EDGE:.0%}  conf≥{MIN_CONFIDENCE:.0%}  odds≥{MIN_ODDS:.2f}")
    calibrator = None
    cal_path = MODEL_DIR / "calibrator.json"
    if cal_path.exists():
        try:
            from calibration import load_calibrator
            calibrator = load_calibrator(cal_path)
            print(f"Loaded probability calibrator: {cal_path.name}")
        except Exception as e:
            print(f"  Warning: could not load calibrator ({e}); using raw probabilities.")

    # Track players we couldn't resolve so user knows coverage gaps
    unknown_players: list[tuple[str, str]] = []

    # ── 5. Predict each match ─────────────────────────────────────────────────
    results = []

    for _, row in singles.iterrows():
        p1_raw  = row["player1"]
        p2_raw  = row["player2"]
        surface = normalise_surface(row["surface"])
        tourn   = row["tournament"]
        rnd     = row["round"]
        date    = row["date"]

        p1 = match_to_internal(p1_raw, known) or match_to_internal(strip_accents(p1_raw), known)
        p2 = match_to_internal(p2_raw, known) or match_to_internal(strip_accents(p2_raw), known)

        if p1 is None or p2 is None:
            unknown = [x for x, y in [(p1_raw, p1), (p2_raw, p2)] if y is None]
            for u in unknown:
                unknown_players.append((str(date), u))
            results.append({
                "Date": str(date), "Tournament": tourn, "Round": rnd, "Surface": surface,
                "Player 1": p1_raw, "Player 2": p2_raw,
                "P1 Win % (no odds)": "—", "P2 Win % (no odds)": "—",
                "P1 Win % (odds)": "—",    "P2 Win % (odds)": "—",
                "P1 Rank (live)": "—",     "P2 Rank (live)": "—",
                "Pinnacle P1 Odds": "—",   "Pinnacle P2 Odds": "—",
                "Bet365 P1 Odds": "—",     "Bet365 P2 Odds": "—",
                "Kelly P1 (fraction)": "—", "Kelly P2 (fraction)": "—",
                f"$ Stake (bankroll ${args.bankroll:.0f})": "—",
                "Favourite": "—",
                "Bet Recommendation": f"Unknown player: {', '.join(unknown)}",
                "Actual Winner": "",
            })
            continue

        feat = feat_mod.get_live_features(tracker, p1, p2, surface)

        # Tournament context features — must be set at inference to match training
        feat["surface_code"]  = float(feat_mod.SURFACE_MAP.get(surface, float("nan")))
        feat["is_grand_slam"] = 1.0 if any(gs.lower() in str(tourn).lower() for gs in feat_mod.GRAND_SLAMS) else 0.0
        feat["round_num"]     = float(feat_mod.ROUND_MAP.get(str(rnd).strip(), float("nan")))
        # Grand Slam men's singles are best of 5; all other ATP events are best of 3
        feat["is_best_of_5"]  = feat["is_grand_slam"]

        # Use live rank if available, else fall back to historical
        p1_rank = live_ranks.get(p1) or model_mod._get_last_rank(raw_df, p1)
        p2_rank = live_ranks.get(p2) or model_mod._get_last_rank(raw_df, p2)
        feat.update(model_mod._compute_rank_features(p1_rank, p2_rank, max_rank))

        # ── No-odds prediction ────────────────────────────────────────────────
        X_no = pd.DataFrame([{col: feat.get(col, np.nan) for col in feat_cols_no_odds}])
        prob_p1_no = float(booster_no_odds.predict(X_no)[0])
        prob_p2_no = 1.0 - prob_p1_no

        # ── Odds prediction + Kelly ───────────────────────────────────────────
        prob_p1_odds = prob_p2_odds = None
        prob_p1_ensemble = prob_p2_ensemble = None
        p1_dec_odds = p2_dec_odds = "—"
        b365_p1_odds = b365_p2_odds = "—"
        bet_info = {"rec": "No odds available", "qualifies": False, "bet_on": None,
                    "kelly_p1": {"quarter": 0.0, "half": 0.0, "full": 0.0},
                    "kelly_p2": {"quarter": 0.0, "half": 0.0, "full": 0.0}}

        match_odds = _match_odds(live_odds_map, p1_raw, p2_raw) if live_odds_map else None

        if match_odds and booster_odds:
            odds_feats = model_mod._compute_odds_features({
                "psw":   match_odds.get("psw"),
                "psl":   match_odds.get("psl"),
                "b365w": match_odds.get("b365w"),
                "b365l": match_odds.get("b365l"),
            })
            if match_odds.get("b365w"):
                b365_p1_odds = match_odds["b365w"]
                b365_p2_odds = match_odds["b365l"]
            feat_with_odds = {**feat, **odds_feats}
            X_odds = pd.DataFrame([{col: feat_with_odds.get(col, np.nan) for col in feat_cols_odds}])
            prob_p1_odds = float(booster_odds.predict(X_odds)[0])
            prob_p2_odds = 1.0 - prob_p1_odds

            # Ensemble blend. Weight is read from ensemble_weight.json (auto-tuned
            # by `python main.py tune-ensemble`), defaulting to DEFAULT_ENSEMBLE_W.
            prob_p1_ensemble = ENSEMBLE_W * prob_p1_odds + (1 - ENSEMBLE_W) * prob_p1_no
            if calibrator is not None:
                prob_p1_ensemble = float(calibrator.transform([prob_p1_ensemble])[0])
            prob_p2_ensemble = 1.0 - prob_p1_ensemble

            p1_dec_odds = match_odds["psw"]
            p2_dec_odds = match_odds["psl"]

            bet_info = _bet_rec(
                prob_p1_ensemble, prob_p2_ensemble, p1_dec_odds, p2_dec_odds, p1, p2
            )

            # CLV: stamp the opening Pinnacle price the moment we recommend a bet.
            # Later, `python -m clv_close` (run near match start) records the
            # closing price and computes CLV.
            if bet_info["qualifies"]:
                try:
                    import clv
                    clv.record_open(p1, p2, prob_p1_ensemble,
                                    p1_dec_odds, p2_dec_odds, date=date)
                except Exception as e:
                    print(f"  Warning: CLV open-record failed: {e}")

        # Favourite: use ensemble when available, else no-odds
        _fav_prob_p1 = prob_p1_ensemble if prob_p1_ensemble is not None else prob_p1_no
        favourite = p1 if _fav_prob_p1 >= 0.5 else p2
        conf      = max(_fav_prob_p1, 1.0 - _fav_prob_p1)

        # Dollar stakes for all three Kelly sizes (show for bet_on side, else "—")
        bankroll = args.bankroll

        def _stake(kf: dict, side: str) -> dict:
            """Return dict of $ stakes for quarter/half/full kelly on the given side."""
            if not bet_info["qualifies"] or bet_info["bet_on"] != side:
                return {"q": "—", "h": "—", "f": "—"}
            return {
                "q": f"${kf['quarter'] * bankroll:.2f}",
                "h": f"${kf['half']    * bankroll:.2f}",
                "f": f"${kf['full']    * bankroll:.2f}",
            }

        s1 = _stake(bet_info["kelly_p1"], "p1")
        s2 = _stake(bet_info["kelly_p2"], "p2")

        # Pick the winning side's stakes for the unified stake columns
        if bet_info["bet_on"] == "p1":
            sq, sh, sf = s1["q"], s1["h"], s1["f"]
        elif bet_info["bet_on"] == "p2":
            sq, sh, sf = s2["q"], s2["h"], s2["f"]
        else:
            sq = sh = sf = "—"

        kp1 = bet_info["kelly_p1"]
        kp2 = bet_info["kelly_p2"]

        results.append({
            "Date":                     str(date),
            "Tournament":               tourn,
            "Round":                    rnd,
            "Surface":                  surface,
            "Player 1":                 p1,
            "Player 2":                 p2,
            "P1 Rank (live)":           int(p1_rank) if pd.notna(p1_rank) else "—",
            "P2 Rank (live)":           int(p2_rank) if pd.notna(p2_rank) else "—",
            "P1 Win % (no odds)":       f"{prob_p1_no:.1%}",
            "P2 Win % (no odds)":       f"{prob_p2_no:.1%}",
            "P1 Win % (odds)":          f"{prob_p1_odds:.1%}" if prob_p1_odds is not None else "—",
            "P2 Win % (odds)":          f"{prob_p2_odds:.1%}" if prob_p2_odds is not None else "—",
            "P1 Win % (ensemble)":      f"{prob_p1_ensemble:.1%}" if prob_p1_ensemble is not None else "—",
            "P2 Win % (ensemble)":      f"{prob_p2_ensemble:.1%}" if prob_p2_ensemble is not None else "—",
            "Pinnacle P1 Odds":         p1_dec_odds,
            "Pinnacle P2 Odds":         p2_dec_odds,
            "Bet365 P1 Odds":           b365_p1_odds,
            "Bet365 P2 Odds":           b365_p2_odds,
            "Full Kelly P1 (frac)":     f"{kp1['full']:.4f}"    if kp1["full"]    > 0 else "—",
            "Half Kelly P1 (frac)":     f"{kp1['half']:.4f}"    if kp1["half"]    > 0 else "—",
            "Qrtr Kelly P1 (frac)":     f"{kp1['quarter']:.4f}" if kp1["quarter"] > 0 else "—",
            "Full Kelly P2 (frac)":     f"{kp2['full']:.4f}"    if kp2["full"]    > 0 else "—",
            "Half Kelly P2 (frac)":     f"{kp2['half']:.4f}"    if kp2["half"]    > 0 else "—",
            "Qrtr Kelly P2 (frac)":     f"{kp2['quarter']:.4f}" if kp2["quarter"] > 0 else "—",
            f"$ Quarter Kelly (${bankroll:.0f})": sq,
            f"$ Half Kelly    (${bankroll:.0f})": sh,
            f"$ Full Kelly    (${bankroll:.0f})": sf,
            "Favourite":                f"{favourite} ({conf:.1%})",
            "Bet Recommendation":       bet_info["rec"],
            "Actual Winner":            "",
        })

    # ── 6. Print results ──────────────────────────────────────────────────────
    df_out = pd.DataFrame(results)
    br = args.bankroll
    display_cols = [
        "Date", "Tournament", "Round", "Surface",
        "Player 1", "Player 2",
        "P1 Rank (live)", "P2 Rank (live)",
        "P1 Win % (ensemble)", "P2 Win % (ensemble)",
        "Pinnacle P1 Odds", "Pinnacle P2 Odds",
        "Qrtr Kelly P1 (frac)", "Half Kelly P1 (frac)", "Full Kelly P1 (frac)",
        "Qrtr Kelly P2 (frac)", "Half Kelly P2 (frac)", "Full Kelly P2 (frac)",
        f"$ Quarter Kelly (${br:.0f})",
        f"$ Half Kelly    (${br:.0f})",
        f"$ Full Kelly    (${br:.0f})",
        "Favourite", "Bet Recommendation",
    ]
    display_cols = [c for c in display_cols if c in df_out.columns]
    print(df_out[display_cols].to_string(index=False))

    # ── 7. Save output ────────────────────────────────────────────────────────
    if args.excel:
        date_tag = args.date or "upcoming"
        out_path = Path(f"predictions_{date_tag}.xlsx")
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Predictions")
            # Bankroll summary sheet
            bankroll = args.bankroll
            bet_rows = df_out[df_out["Bet Recommendation"].str.startswith("BET", na=False)]

            def _sum_col(col):
                if col not in bet_rows.columns:
                    return 0.0
                return sum(float(v.replace("$", "")) for v in bet_rows[col]
                           if isinstance(v, str) and v.startswith("$"))

            qk_col = f"$ Quarter Kelly (${bankroll:.0f})"
            hk_col = f"$ Half Kelly    (${bankroll:.0f})"
            fk_col = f"$ Full Kelly    (${bankroll:.0f})"
            total_q = _sum_col(qk_col)
            total_h = _sum_col(hk_col)
            total_f = _sum_col(fk_col)

            summary = pd.DataFrame([
                {"Item": "Starting bankroll",          "Value": f"${bankroll:.2f}"},
                {"Item": "Qualifying bets",             "Value": len(bet_rows)},
                {"Item": "Total staked — Quarter Kelly","Value": f"${total_q:.2f}"},
                {"Item": "Total staked — Half Kelly",   "Value": f"${total_h:.2f}"},
                {"Item": "Total staked — Full Kelly",   "Value": f"${total_f:.2f}"},
                {"Item": "Remaining (Quarter Kelly)",   "Value": f"${bankroll - total_q:.2f}"},
                {"Item": "Rules",                       "Value": "Edge ≥ 3%, Confidence ≥ 70%, Odds ≥ 1.20"},
            ])
            summary.to_excel(writer, index=False, sheet_name="Bankroll Summary")
        print(f"\nPredictions saved → {out_path}")
    else:
        out_path = Path("predictions_log.csv")
        if out_path.exists():
            existing = pd.read_csv(out_path)
            # Preserve existing Actual Winner entries
            merged = pd.concat([existing, df_out], ignore_index=True)
            merged = merged.drop_duplicates(subset=["Date", "Player 1", "Player 2"], keep="first")
            df_out = merged
        df_out.to_csv(out_path, index=False)
        print(f"\nPredictions saved → {out_path}")

    print("\nFill in the 'Actual Winner' column after matches finish to track accuracy.")

    # ── 8. Append unknown-player encounters to log ───────────────────────────
    if unknown_players:
        log_path = Path(args.log_unknown)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_df = pd.DataFrame(unknown_players, columns=["date", "raw_name"])
        if log_path.exists():
            log_df = pd.concat([pd.read_csv(log_path), log_df], ignore_index=True)
            log_df = log_df.drop_duplicates(subset=["date", "raw_name"], keep="first")
        log_df.to_csv(log_path, index=False)
        print(f"\n⚠️  {len(unknown_players)} unknown player encounter(s) appended to {log_path}")


if __name__ == "__main__":
    main()
