"""Stage 3: LightGBM model training, evaluation, and prediction."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbose": -1,
}

NON_FEATURE_COLS = {
    "date", "winner", "loser", "p1", "p2", "surface", "round",
    "best_of", "tournament", "label", "year", "court",
}

# Odds-derived columns — excluded from the no-odds model
ODDS_COLS = {
    "implied_prob_w", "implied_prob_l", "overround",
    "pin_prob_w", "pin_prob_l", "b365_vs_pin",
    "max_prob_w", "max_prob_l",
}

# Confirmed zero-gain features (permutation + split + gain analysis on 2023-2024 holdout).
# The model never uses these; removing them keeps the feature space clean and avoids
# NaN-at-inference issues for features that are hard to populate at prediction time.
ZERO_GAIN_COLS = {
    "round_num", "is_grand_slam", "is_best_of_5", "surface_code",
    "w_streak_5", "l_streak_5",
    "rank_pct_w", "rank_pct_l",
    "h2h_recent_2y_w", "h2h_surface_w",
}

# ── Temporal split boundaries (date-based) ────────────────────────────────────
#
#   Training   2000-01-01 → 2023-12-31   core temporal CV folds
#   Validation 2024-01-01 → 2025-06-30   early-stopping reference for final model
#   Test       2025-07-01 → 2026-04-30   holdout — never seen during training
#
TRAIN_END  = pd.Timestamp("2023-12-31")
VAL_START  = pd.Timestamp("2024-01-01")
VAL_END    = pd.Timestamp("2025-06-30")
TEST_START = pd.Timestamp("2025-07-01")
TEST_END   = pd.Timestamp("2026-04-30")


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS and c not in ZERO_GAIN_COLS]


def _get_no_odds_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in NON_FEATURE_COLS and c not in ODDS_COLS and c not in ZERO_GAIN_COLS]


def _run_cv(X: pd.DataFrame, y: pd.Series, feature_cols: list[str],
             lgb_params: dict, n_splits: int, label: str) -> int:
    """Run temporal CV, print fold metrics, return median best_iteration."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_losses, cv_aucs, best_iterations = [], [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        booster = lgb.train(
            lgb_params, dtrain, num_boost_round=1000,
            valid_sets=[dval], callbacks=callbacks,
        )

        preds     = booster.predict(X_val)
        fold_loss = log_loss(y_val, preds)
        fold_auc  = roc_auc_score(y_val, preds)
        cv_losses.append(fold_loss)
        cv_aucs.append(fold_auc)
        best_iterations.append(booster.best_iteration)
        print(f"  [{label}] Fold {fold+1}/{n_splits}: log-loss={fold_loss:.4f}  AUC={fold_auc:.4f}  best_iter={booster.best_iteration}")

    print(f"\n  [{label}] CV mean log-loss : {np.mean(cv_losses):.4f} ± {np.std(cv_losses):.4f}")
    print(f"  [{label}] CV mean AUC (OOS): {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    return int(np.median(best_iterations)) or 200


def _train_final(X_core: pd.DataFrame, y_core: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series,
                  feature_cols: list[str], lgb_params: dict,
                  n_splits: int, label: str) -> lgb.Booster:
    """Run CV on core data, then train the final model on core+val using median best_iter.

    Two-stage final training:
      Stage 1 — find best_iter via early-stopping on val year.
      Stage 2 — retrain on core+val with that fixed round count (no early stopping).
    """
    # CV to get a rough estimate and validate architecture
    X_all = pd.concat([X_core, X_val])
    y_all = pd.concat([y_core, y_val])
    cv_best_iter = _run_cv(X_all, y_all, feature_cols, lgb_params, n_splits, label)
    print(f"  [{label}] CV median best_iteration: {cv_best_iter}")

    # Stage 1: early-stopping pass using the val year to get precise best_iter
    dtrain_s1 = lgb.Dataset(X_core, label=y_core, feature_name=feature_cols)
    dval_s1   = lgb.Dataset(X_val,  label=y_val,  reference=dtrain_s1)
    callbacks_s1 = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    booster_s1 = lgb.train(
        lgb_params, dtrain_s1, num_boost_round=1000,
        valid_sets=[dval_s1], callbacks=callbacks_s1,
    )
    es_best_iter = booster_s1.best_iteration
    print(f"  [{label}] Early-stopping best_iteration (on val year): {es_best_iter}")

    # Stage 2: retrain on core+val with fixed round count
    dtrain_final = lgb.Dataset(X_all, label=y_all, feature_name=feature_cols)
    final = lgb.train(lgb_params, dtrain_final, num_boost_round=es_best_iter)
    return final


def train(
    features_parquet: Path,
    out_dir: Path,
    n_splits: int = 5,
    params: dict | None = None,
    holdout_years: list[int] | None = None,
) -> lgb.Booster:
    """Train LightGBM models (with-odds and no-odds) with proper temporal splits.

    Data zones enforced automatically:
      Core training  : date <= TRAIN_END      (temporal CV folds)
      Validation     : VAL_START to VAL_END   (early-stopping reference)
      Test / holdout : TEST_START to TEST_END (never seen during training)

    Args:
        holdout_years: Ignored (kept for CLI compatibility). Test set is always
                       TEST_START–TEST_END. Pass holdout_years=[] only if you
                       want a production model trained on core+val+test.
    """
    df = pd.read_parquet(features_parquet)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Exclude test set — never touches training
    before = len(df)
    if holdout_years is None or holdout_years:
        # Default: exclude test period
        df_train_pool = df[df["date"] < TEST_START].reset_index(drop=True)
        print(f"Test holdout excluded ({TEST_START.date()}→{TEST_END.date()}): "
              f"{before - len(df_train_pool):,} rows removed.")
    else:
        # holdout_years=[] → production model, include everything up to TEST_END
        df_train_pool = df[df["date"] <= TEST_END].reset_index(drop=True)
        print("Production mode: training on ALL data up to TEST_END.")

    # Split into core-train and validation window
    core_df = df_train_pool[df_train_pool["date"] <= TRAIN_END].reset_index(drop=True)
    val_df  = df_train_pool[(df_train_pool["date"] >= VAL_START) &
                             (df_train_pool["date"] <= VAL_END)].reset_index(drop=True)

    print(f"\nData split:")
    print(f"  Core training  : {core_df['date'].min().date()}–{core_df['date'].max().date()}  ({len(core_df):,} rows)")
    if not val_df.empty:
        print(f"  Validation     : {VAL_START.date()}–{VAL_END.date()}  ({len(val_df):,} rows)")
    print(f"  Test (holdout) : {TEST_START.date()}–{TEST_END.date()}")

    lgb_params      = {**DEFAULT_PARAMS, **(params or {})}
    feature_cols    = _get_feature_cols(df)
    feature_cols_no = _get_no_odds_feature_cols(df)

    X_core    = core_df[feature_cols].astype(float)
    X_val_f   = val_df[feature_cols].astype(float)  if not val_df.empty else pd.DataFrame(columns=feature_cols)
    y_core    = core_df["label"].astype(int)
    y_val_s   = val_df["label"].astype(int)   if not val_df.empty else pd.Series(dtype=int)

    X_core_no = core_df[feature_cols_no].astype(float)
    X_val_no  = val_df[feature_cols_no].astype(float)  if not val_df.empty else pd.DataFrame(columns=feature_cols_no)

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── With-odds model ───────────────────────────────────────────────────────
    print("\n── With-odds model ──────────────────────────────────────────────")
    if val_df.empty:
        # No separate val year (e.g., production mode without holdout) — use CV only
        X_all = X_core
        y_all = y_core
        best_iter = _run_cv(X_all, y_all, feature_cols, lgb_params, n_splits, "with_odds")
        dtrain_full = lgb.Dataset(X_all, label=y_all, feature_name=feature_cols)
        final_booster = lgb.train(lgb_params, dtrain_full, num_boost_round=best_iter)
    else:
        final_booster = _train_final(X_core, y_core, X_val_f, y_val_s, feature_cols, lgb_params, n_splits, "with_odds")

    model_path = out_dir / "model.lgb"
    final_booster.save_model(str(model_path))
    (out_dir / "feature_cols.json").write_text(json.dumps(feature_cols))
    print(f"  Saved → {model_path}")

    # ── No-odds model ─────────────────────────────────────────────────────────
    print("\n── No-odds model ────────────────────────────────────────────────")
    if val_df.empty:
        X_all_no = X_core_no
        y_all_no = y_core
        best_iter_no = _run_cv(X_all_no, y_all_no, feature_cols_no, lgb_params, n_splits, "no_odds")
        dtrain_no = lgb.Dataset(X_all_no, label=y_all_no, feature_name=feature_cols_no)
        final_no  = lgb.train(lgb_params, dtrain_no, num_boost_round=best_iter_no)
    else:
        final_no = _train_final(X_core_no, y_core, X_val_no, y_val_s, feature_cols_no, lgb_params, n_splits, "no_odds")

    model_no_path = out_dir / "model_no_odds.lgb"
    final_no.save_model(str(model_no_path))
    (out_dir / "feature_cols_no_odds.json").write_text(json.dumps(feature_cols_no))
    print(f"  Saved → {model_no_path}")

    # ── Rank → Elo calibration table (for cold-start inference) ──────────────
    print("\n── Rank→Elo cold-start calibration ─────────────────────────────")
    from features import build_rank_elo_table
    rank_elo_poly = build_rank_elo_table(df)
    np.save(str(out_dir / "rank_elo_coeffs.npy"), rank_elo_poly.coeffs)
    # Quick sanity: rank 1 ≈ 2200, rank 100 ≈ 1600, rank 500 ≈ 1350
    for test_rank in (1, 10, 50, 100, 250, 500):
        from features import rank_to_elo
        print(f"  Rank {test_rank:>4}: estimated Elo = {rank_to_elo(test_rank, rank_elo_poly):.0f}")
    print(f"  Saved → {out_dir / 'rank_elo_coeffs.npy'}")

    return final_booster


def evaluate(
    model_dir: Path,
    features_parquet: Path,
    test_years: list[int] | None = None,
) -> dict:
    """Evaluate on the holdout test set (true OOS — eval model trained on non-test data).

    Retrains a dedicated evaluation model on core+val only (never on test period),
    so the reported metrics are honest out-of-sample numbers.

    Date-based split mirrors train():
      Core  : date <= TRAIN_END
      Val   : VAL_START to VAL_END   (early-stopping reference)
      Test  : TEST_START to TEST_END (evaluated here)

    test_years is kept for CLI compatibility but ignored — test range is always
    TEST_START→TEST_END.
    """
    feature_cols = json.loads((model_dir / "feature_cols.json").read_text())

    df = pd.read_parquet(features_parquet)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    test = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()
    if test.empty:
        print(f"No data found for test period {TEST_START.date()}–{TEST_END.date()}")
        return {}

    core_df = df[df["date"] <= TRAIN_END].reset_index(drop=True)
    val_df  = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].reset_index(drop=True)
    train_df = pd.concat([core_df, val_df]).sort_values("date").reset_index(drop=True)

    if train_df.empty:
        print("No training data outside test period — cannot evaluate OOS.")
        return {}

    print(f"\nEval model split:")
    print(f"  Core  : {core_df['date'].min().date()}–{core_df['date'].max().date()}  ({len(core_df):,} rows)")
    print(f"  Val   : {VAL_START.date()}–{VAL_END.date()}  ({len(val_df):,} rows)")
    print(f"  Test  : {TEST_START.date()}–{TEST_END.date()}  ({len(test):,} rows)")

    X_core = core_df[feature_cols].astype(float)
    y_core = core_df["label"].astype(int)
    X_val  = val_df[feature_cols].astype(float)
    y_val  = val_df["label"].astype(int)

    dtrain_s1  = lgb.Dataset(X_core, label=y_core, feature_name=feature_cols)
    dval_s1    = lgb.Dataset(X_val,  label=y_val,  reference=dtrain_s1)
    callbacks  = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]

    print(f"\nStage 1 — early-stopping on val period {VAL_START.date()}–{VAL_END.date()}…")
    booster_s1 = lgb.train(DEFAULT_PARAMS, dtrain_s1, num_boost_round=1000,
                            valid_sets=[dval_s1], callbacks=callbacks)
    best_iter = booster_s1.best_iteration
    print(f"  Best iteration: {best_iter}")

    print(f"Stage 2 — retraining on core+val ({len(train_df):,} rows) for {best_iter} rounds…")
    X_all = train_df[feature_cols].astype(float)
    y_all = train_df["label"].astype(int)
    dtrain_final = lgb.Dataset(X_all, label=y_all, feature_name=feature_cols)
    eval_booster = lgb.train(DEFAULT_PARAMS, dtrain_final, num_boost_round=best_iter)

    X_test = test[feature_cols].astype(float)
    y_test = test["label"].astype(int)
    probs  = eval_booster.predict(X_test)

    ll    = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    auc   = roc_auc_score(y_test, probs)

    print(f"\nEvaluation on {TEST_START.date()}–{TEST_END.date()}:")
    print(f"  Rows        : {len(test):,}")
    print(f"  ROC-AUC     : {auc:.4f}")
    print(f"  Log-loss    : {ll:.4f}")
    print(f"  Brier score : {brier:.4f}")

    # Flat-stake betting simulation (p1 bets only for quick signal check)
    threshold = 0.55
    bets = test.copy()
    bets["prob"] = probs
    bets = bets[bets["prob"] > threshold]
    if not bets.empty and "avgw" in bets.columns:
        bets = bets.dropna(subset=["avgw"])
        profits = np.where(bets["label"] == 1, bets["avgw"] - 1, -1.0)
        roi = profits.mean() * 100
        print(f"\n  Flat-stake ROI (prob>{threshold}, avg odds): {roi:.2f}%  ({len(bets):,} bets)")
    else:
        roi = None

    return {"log_loss": ll, "brier": brier, "auc": auc, "roi": roi}


def _get_last_rank(raw_df: pd.DataFrame, player: str) -> float:
    """Return the most recently recorded ATP rank for player, or NaN."""
    as_winner = raw_df[raw_df["winner"] == player][["date", "w_rank"]].rename(columns={"w_rank": "rank"})
    as_loser  = raw_df[raw_df["loser"]  == player][["date", "l_rank"]].rename(columns={"l_rank": "rank"})
    both = pd.concat([as_winner, as_loser]).dropna(subset=["rank"])
    if both.empty:
        return np.nan
    return float(both.sort_values("date").iloc[-1]["rank"])


def _compute_rank_features(p1_rank: float, p2_rank: float, max_rank: float) -> dict:
    """Compute rank feature group given the two players' ranks."""
    rec: dict = {"w_rank": p1_rank, "l_rank": p2_rank}
    if pd.notna(p1_rank) and pd.notna(p2_rank) and p1_rank > 0 and p2_rank > 0:
        rec["rank_diff"]     = p2_rank - p1_rank
        rec["log_rank_w"]    = np.log1p(p1_rank)
        rec["log_rank_l"]    = np.log1p(p2_rank)
        rec["log_rank_diff"] = np.log1p(p2_rank) - np.log1p(p1_rank)
        rec["rank_pct_w"]    = p1_rank / max_rank if max_rank > 0 else np.nan
        rec["rank_pct_l"]    = p2_rank / max_rank if max_rank > 0 else np.nan
    else:
        for c in ["rank_diff", "log_rank_w", "log_rank_l", "log_rank_diff", "rank_pct_w", "rank_pct_l"]:
            rec[c] = np.nan
    return rec


def _compute_odds_features(odds: dict) -> dict:
    """Compute bookmaker odds feature group from a raw odds dict (b365w, b365l, psw, psl, maxw, maxl)."""
    rec: dict = {}

    b365w, b365l = odds.get("b365w"), odds.get("b365l")
    if b365w and b365l and b365w > 0 and b365l > 0:
        imp_w, imp_l = 1 / b365w, 1 / b365l
        rec["implied_prob_w"] = imp_w
        rec["implied_prob_l"] = imp_l
        rec["overround"]      = imp_w + imp_l - 1
    else:
        rec["implied_prob_w"] = np.nan
        rec["implied_prob_l"] = np.nan
        rec["overround"]      = np.nan

    psw, psl = odds.get("psw"), odds.get("psl")
    if psw and psl and psw > 0 and psl > 0:
        raw_w, raw_l = 1 / psw, 1 / psl
        total  = raw_w + raw_l
        pin_w  = raw_w / total
        rec["pin_prob_w"] = pin_w
        rec["pin_prob_l"] = 1.0 - pin_w
        imp_w = rec.get("implied_prob_w")
        imp_l = rec.get("implied_prob_l")
        if pd.notna(imp_w) and pd.notna(imp_l) and (imp_w + imp_l) > 0:
            b365_norm_w = imp_w / (imp_w + imp_l)
            rec["b365_vs_pin"] = b365_norm_w - pin_w
        else:
            rec["b365_vs_pin"] = np.nan
    else:
        rec["pin_prob_w"]  = np.nan
        rec["pin_prob_l"]  = np.nan
        rec["b365_vs_pin"] = np.nan

    maxw, maxl = odds.get("maxw"), odds.get("maxl")
    if maxw and maxl and maxw > 0 and maxl > 0:
        raw_w, raw_l = 1 / maxw, 1 / maxl
        max_w = raw_w / (raw_w + raw_l)
        rec["max_prob_w"] = max_w
        rec["max_prob_l"] = 1.0 - max_w
    else:
        rec["max_prob_w"] = np.nan
        rec["max_prob_l"] = np.nan

    return rec


def predict(
    model_dir: Path,
    p1: str,
    p2: str,
    surface: str,
    raw_parquet: Path,
    odds: dict | None = None,
    tournament: str | None = None,
    round_label: str | None = None,
) -> dict:
    """Predict win probability for a match.

    Args:
        odds:        Optional dict with keys b365w, b365l, psw, psl, maxw, maxl.
                     When provided the full model (model.lgb) is used; otherwise
                     model_no_odds.lgb is used.
        tournament:  Tournament name — used to set is_grand_slam / is_best_of_5.
        round_label: Round string e.g. "QF", "SF", "F", "R32" — sets round_num.

    Returns:
        Dict with p1_win_prob, p2_win_prob, model_used, features, booster, X, feature_cols.
    """
    from features import (
        build_tracker, get_live_features, GRAND_SLAMS, SURFACE_MAP, ROUND_MAP,
        apply_cold_start_elo, cold_start_form_features, rank_to_elo,
    )

    raw_df = pd.read_parquet(raw_parquet)
    raw_df = raw_df.sort_values("date").reset_index(drop=True)

    print("Building player tracker (scanning historical data)...")
    tracker = build_tracker(raw_df)

    # ── Rank features — prefer live ATP rankings, fall back to historical ──────
    max_rank = float(pd.concat([raw_df["w_rank"], raw_df["l_rank"]]).dropna().max())
    try:
        from atp_scraper import get_live_rank_map
        live_ranks = get_live_rank_map([p1, p2])
        p1_rank = float(live_ranks[p1]) if p1 in live_ranks else _get_last_rank(raw_df, p1)
        p2_rank = float(live_ranks[p2]) if p2 in live_ranks else _get_last_rank(raw_df, p2)
    except Exception:
        p1_rank = _get_last_rank(raw_df, p1)
        p2_rank = _get_last_rank(raw_df, p2)

    # ── Cold-start: fix Elo for players not in historical data ────────────────
    rank_elo_coeffs_path = model_dir / "rank_elo_coeffs.npy"
    rank_elo_poly = None
    if rank_elo_coeffs_path.exists():
        rank_elo_poly = np.poly1d(np.load(str(rank_elo_coeffs_path)))

    known_set = set(raw_df["winner"].tolist() + raw_df["loser"].tolist())
    cold_start_flags = {}
    for player, rank, side in ((p1, p1_rank, "p1"), (p2, p2_rank, "p2")):
        if player not in known_set and rank_elo_poly is not None:
            applied = apply_cold_start_elo(tracker, player, rank, rank_elo_poly)
            cold_start_flags[side] = applied
            if applied:
                est = rank_to_elo(rank, rank_elo_poly)
                print(f"  [Cold-start] {player}: rank {int(rank) if pd.notna(rank) else '?'}"
                      f" → Elo initialised to {est:.0f}")
                # Fetch recent matches from Sofascore to seed form features
                try:
                    from atp_scraper import fetch_sofascore_player_recent
                    recent = fetch_sofascore_player_recent(player, n=15)
                    if recent:
                        print(f"  [Cold-start] {player}: fetched {len(recent)} recent matches from Sofascore")
                    cold_start_flags[f"{side}_recent"] = recent
                except Exception as e:
                    print(f"  [Cold-start] Sofascore fetch failed for {player}: {e}")
                    cold_start_flags[f"{side}_recent"] = []

    feat = get_live_features(tracker, p1, p2, surface)

    # Apply cold-start form features from Sofascore where available
    for side, prefix in (("p1", "w_"), ("p2", "l_")):
        recent = cold_start_flags.get(f"{side}_recent", [])
        if recent:
            form = cold_start_form_features(recent)
            for k, v in form.items():
                feat[f"{prefix}{k}"] = v
            # Recompute diff features that depend on this player
            feat["streak_diff"] = (feat.get("w_current_streak", 0) or 0) - (feat.get("l_current_streak", 0) or 0)
            feat["momentum_diff"] = (
                (feat.get("w_momentum", np.nan) or np.nan) -
                (feat.get("l_momentum", np.nan) or np.nan)
                if pd.notna(feat.get("w_momentum")) and pd.notna(feat.get("l_momentum"))
                else np.nan
            )

    # Tournament context features — set from args so they match training distribution
    feat["surface_code"]  = float(SURFACE_MAP.get(surface, float("nan")))
    is_gs = 1.0 if (tournament and any(gs.lower() in tournament.lower() for gs in GRAND_SLAMS)) else 0.0
    feat["is_grand_slam"] = is_gs
    feat["is_best_of_5"]  = is_gs   # men's Grand Slams are best of 5
    feat["round_num"]     = float(ROUND_MAP.get(str(round_label or "").strip(), float("nan")))

    feat.update(_compute_rank_features(p1_rank, p2_rank, max_rank))

    # Odds features + model selection
    if odds:
        feat.update(_compute_odds_features(odds))
        model_file = model_dir / "model.lgb"
        cols_file  = model_dir / "feature_cols.json"
        model_used = "with_odds"
    else:
        model_file = model_dir / "model_no_odds.lgb"
        cols_file  = model_dir / "feature_cols_no_odds.json"
        model_used = "no_odds"

    booster      = lgb.Booster(model_file=str(model_file))
    feature_cols = json.loads(cols_file.read_text())

    X       = pd.DataFrame([{col: feat.get(col, np.nan) for col in feature_cols}])
    prob_p1 = float(booster.predict(X)[0])

    cs_note = " [cold-start]" if any(cold_start_flags.get(s) for s in ("p1", "p2")) else ""
    print(f"\nPrediction: {p1} vs {p2} on {surface}{cs_note}")
    print(f"  Model:  {model_used}")
    print(f"  {p1} Elo: {feat.get('w_elo', 1500):.1f}  Rank: {p1_rank}")
    print(f"  {p2} Elo: {feat.get('l_elo', 1500):.1f}  Rank: {p2_rank}")
    print(f"  {p1} win probability: {prob_p1:.1%}")
    print(f"  {p2} win probability: {1 - prob_p1:.1%}")

    return {
        "p1": p1,
        "p2": p2,
        "surface": surface,
        "p1_win_prob": round(prob_p1, 4),
        "p2_win_prob": round(1 - prob_p1, 4),
        "model_used": model_used,
        "features": feat,
        "booster": booster,
        "X": X,
        "feature_cols": feature_cols,
    }


def predict_ensemble(
    model_dir: Path,
    p1: str,
    p2: str,
    surface: str,
    raw_parquet: Path,
    odds: dict | None = None,
    weight_odds: float = 0.65,
    tournament: str | None = None,
    round_label: str | None = None,
) -> dict:
    """Blend no-odds and with-odds models into a single probability.

    When odds are available:
        final_prob = weight_odds * prob_with_odds + (1 - weight_odds) * prob_no_odds
        Default weight_odds=0.65 reflects the relative AUC gain (0.781 vs 0.720).

    When odds are unavailable:
        Returns the no-odds model probability unchanged.

    Returns same dict shape as predict(), with additional keys:
        prob_no_odds, prob_with_odds, ensemble_weight_odds.
    """
    kw = {"tournament": tournament, "round_label": round_label}
    result_no = predict(model_dir, p1, p2, surface, raw_parquet, odds=None, **kw)
    prob_no   = result_no["p1_win_prob"]

    if odds:
        result_odds  = predict(model_dir, p1, p2, surface, raw_parquet, odds=odds, **kw)
        prob_with    = result_odds["p1_win_prob"]
        prob_ensemble = weight_odds * prob_with + (1 - weight_odds) * prob_no
    else:
        prob_with     = None
        prob_ensemble = prob_no
        weight_odds   = 0.0

    print(f"\nEnsemble: no-odds={prob_no:.1%}  "
          + (f"with-odds={prob_with:.1%}  " if prob_with else "")
          + f"→ final={prob_ensemble:.1%}")

    return {
        "p1":                   p1,
        "p2":                   p2,
        "surface":              surface,
        "p1_win_prob":          round(prob_ensemble, 4),
        "p2_win_prob":          round(1 - prob_ensemble, 4),
        "prob_no_odds":         round(prob_no, 4),
        "prob_with_odds":       round(prob_with, 4) if prob_with else None,
        "ensemble_weight_odds": weight_odds,
        "model_used":           "ensemble",
    }


def explain_prediction(
    booster: lgb.Booster,
    X: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Return SHAP contributions for a single prediction, sorted by |shap_value|.

    Uses LightGBM's built-in pred_contrib — no external shap library needed.
    The last column returned by pred_contrib is the bias (expected value) and
    is dropped here.
    """
    contribs     = booster.predict(X, pred_contrib=True)
    shap_values  = contribs[0, :-1]       # drop bias term
    feature_vals = X.iloc[0].values

    df = pd.DataFrame({
        "feature":       feature_cols,
        "shap_value":    shap_values,
        "feature_value": feature_vals,
    })
    df["abs_shap"] = df["shap_value"].abs()
    return df.sort_values("abs_shap", ascending=False).drop(columns="abs_shap").reset_index(drop=True)
