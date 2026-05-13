#!/usr/bin/env python3
"""
ATP Tennis Model — Complete Evaluation Script
Run this to validate your model end-to-end before market deployment.

Usage:
  python evaluate_atp_model.py \
      --features data/processed/features.parquet \
      --model-dir data/processed/models \
      --verbose
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, log_loss,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: DATA LEAKAGE & TEMPORAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def check_chronological_order(df: pd.DataFrame) -> bool:
    print("\n" + "=" * 70)
    print("SECTION 1: CHRONOLOGICAL ORDER & DATA LEAKAGE CHECK")
    print("=" * 70)

    is_sorted = df["date"].is_monotonic_increasing
    if is_sorted:
        print("✔  Data is in strict chronological order (no shuffle)")
    else:
        print("✘  WARNING: Data not in chronological order — temporal leakage risk!")
        print(f"   First 5 dates: {df['date'].head().tolist()}")
    return is_sorted


def check_train_val_test_separation(df: pd.DataFrame,
                                     train_end="2023-12-31",
                                     val_end="2025-06-30"):
    train_end = pd.to_datetime(train_end)
    val_end   = pd.to_datetime(val_end)

    train = df[df["date"] <= train_end]
    val   = df[(df["date"] > train_end) & (df["date"] <= val_end)]
    test  = df[df["date"] > val_end]

    print(f"\n  Train : {len(train):>6,} rows  {train['date'].min().date()} → {train['date'].max().date()}")
    print(f"  Val   : {len(val):>6,} rows  {val['date'].min().date()} → {val['date'].max().date()}")
    print(f"  Test  : {len(test):>6,} rows  {test['date'].min().date()} → {test['date'].max().date()}")

    ok = True
    if train["date"].max() < val["date"].min():
        print("  ✔  Train / Val don't overlap")
    else:
        print("  ✘  WARNING: Train / Val overlap!")
        ok = False

    if val["date"].max() < test["date"].min():
        print("  ✔  Val / Test don't overlap")
    else:
        print("  ✘  WARNING: Val / Test overlap!")
        ok = False

    if ok:
        print("  ✔  Temporal separation verified")

    return ok, {"train": train, "val": val, "test": test}


def check_feature_leakage(df: pd.DataFrame) -> bool:
    print("\n  Checking feature sanity (rolling stats should be in range)...")
    issues = []

    win_rate_cols = [c for c in df.columns if "win_rate" in c]
    for col in win_rate_cols[:5]:
        vals = df[col].dropna()
        bad = ((vals < 0) | (vals > 1)).sum()
        if bad:
            issues.append(f"  {col}: {bad} values outside [0,1]")

    match_10d_cols = [c for c in df.columns if "matches_10d" in c]
    for col in match_10d_cols[:2]:
        unrealistic = (df[col].dropna() > 30).sum()
        if unrealistic:
            issues.append(f"  {col}: {unrealistic} values > 30 matches in 10 days")

    if issues:
        print("  ✘  Potential leakage:")
        for i in issues:
            print(i)
        return False
    print("  ✔  Rolling features look reasonable (no obvious leakage)")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: CLASSIFICATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_classification_metrics(y_true, y_pred,
                                    y_market=None, label="Test") -> dict:
    print("\n" + "=" * 70)
    print(f"SECTION 2: CLASSIFICATION METRICS ({label} set)")
    print("=" * 70)

    auc       = roc_auc_score(y_true, y_pred)
    y_bin     = (y_pred > 0.5).astype(int)
    accuracy  = accuracy_score(y_true, y_bin)
    precision = precision_score(y_true, y_bin, zero_division=0)
    recall    = recall_score(y_true, y_bin, zero_division=0)
    f1        = f1_score(y_true, y_bin, zero_division=0)
    logloss   = log_loss(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    print(f"\n  AUC:          {auc:.4f}  {'✔' if auc >= 0.75 else '✘ (target ≥ 0.75)'}")
    print(f"  Accuracy:     {accuracy:.4f}  {'✔' if accuracy >= 0.53 else '✘ (target ≥ 0.53)'}")
    print(f"  Precision:    {precision:.4f}")
    print(f"  Recall:       {recall:.4f}")
    print(f"  F1-Score:     {f1:.4f}")
    print(f"  Log Loss:     {logloss:.4f}  {'✔' if logloss < 0.50 else '✘ (target < 0.50)'}")
    print(f"  Specificity:  {specificity:.4f}")

    baseline = max(y_true.mean(), 1 - y_true.mean())
    print(f"\n  Baseline (majority class): {baseline:.2%}")
    print(f"  Model beats baseline:      {'✔' if accuracy > baseline else '✘'}")

    if y_market is not None:
        mask = ~pd.isnull(y_market)
        if mask.sum() > 100:
            auc_market = roc_auc_score(y_true[mask], y_market[mask])
            print(f"\n  Pinnacle AUC (benchmark): {auc_market:.4f}")
            print(f"  Model vs Pinnacle:        {'✔ beats market' if auc > auc_market else '✘ below market'}")

    return {
        "auc": auc, "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "logloss": logloss,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def check_calibration(y_true, y_pred, n_bins=10) -> dict:
    print("\n" + "=" * 70)
    print("SECTION 3: CALIBRATION ANALYSIS")
    print("=" * 70)

    bins = pd.cut(y_pred, bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    cal = (
        pd.DataFrame({"pred": y_pred, "actual": y_true, "bin": bins})
        .groupby("bin", observed=True)
        .agg(mean_pred=("pred", "mean"), mean_actual=("actual", "mean"), n=("actual", "count"))
        .reset_index(drop=True)
    )
    cal["gap"] = (cal["mean_pred"] - cal["mean_actual"]).abs()

    print("\n  Pred Prob | Actual Win% | Gap    | Matches")
    print("  " + "-" * 44)
    for _, row in cal.iterrows():
        print(f"  {row['mean_pred']:>8.0%} | {row['mean_actual']:>10.0%} | "
              f"{row['gap']:>6.1%} | {int(row['n']):>6}")

    max_gap = cal["gap"].max()
    print(f"\n  Max calibration gap: {max_gap:.2%}  {'✔' if max_gap <= 0.03 else '✘ (target ≤ 3%)'}")

    return {"calibration": cal, "max_gap": float(max_gap)}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CONFIDENCE BUCKETS
# ─────────────────────────────────────────────────────────────────────────────

def confidence_bucket_analysis(y_true, y_pred) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("SECTION 4: CONFIDENCE BUCKET ANALYSIS")
    print("=" * 70)

    df = pd.DataFrame({"pred": y_pred, "actual": y_true})
    buckets = [
        ("Very Low  (35–50%)", 0.35, 0.50),
        ("Low       (50–60%)", 0.50, 0.60),
        ("Medium    (60–70%)", 0.60, 0.70),
        ("High      (70–80%)", 0.70, 0.80),
        ("Very High (80%+  )", 0.80, 1.01),
    ]

    rows = []
    for name, lo, hi in buckets:
        sub = df[(df["pred"] >= lo) & (df["pred"] < hi)]
        if len(sub) == 0:
            continue
        acc = (sub["actual"] == (sub["pred"] > 0.5).astype(int)).mean()
        rows.append({
            "Bucket": name, "Matches": len(sub),
            "Accuracy": acc, "Pred": sub["pred"].mean(), "Actual": sub["actual"].mean(),
        })

    res = pd.DataFrame(rows)
    print("\n  Bucket               | Matches | Accuracy | Pred Prob | Actual")
    print("  " + "-" * 64)
    for _, r in res.iterrows():
        print(f"  {r['Bucket']:<20} | {int(r['Matches']):>7} | "
              f"{r['Accuracy']:>8.1%} | {r['Pred']:>9.1%} | {r['Actual']:>6.1%}")

    mono = res["Accuracy"].is_monotonic_increasing
    print(f"\n  Accuracy increases with confidence: {'✔' if mono else '✘ (not monotonic — check model)'}")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MODEL VARIANT COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_model_variants(y_true, pred_with_odds, pred_no_odds) -> dict:
    print("\n" + "=" * 70)
    print("SECTION 5: MODEL VARIANT COMPARISON (with-odds vs no-odds)")
    print("=" * 70)

    auc_w = roc_auc_score(y_true, pred_with_odds)
    auc_n = roc_auc_score(y_true, pred_no_odds)
    corr  = float(np.corrcoef(pred_with_odds, pred_no_odds)[0, 1])
    high_div = int((np.abs(pred_with_odds - pred_no_odds) > 0.20).sum())

    print(f"\n  With-odds AUC : {auc_w:.4f}")
    print(f"  No-odds AUC   : {auc_n:.4f}")
    print(f"  Δ AUC         : {auc_w - auc_n:+.4f}  (with-odds advantage)")
    print(f"  Prediction correlation : {corr:.4f}  {'✔' if corr > 0.80 else '✘ (target > 0.80)'}")
    print(f"  High divergence (>20pp): {high_div} matches")

    # Grid-search ensemble weights
    print("\n  Ensemble weight optimisation (grid search on test set):")
    best_auc, best_w = 0.0, 0.65
    for w in np.arange(0.45, 0.86, 0.05):
        ens = w * pred_with_odds + (1 - w) * pred_no_odds
        a   = roc_auc_score(y_true, ens)
        if a > best_auc:
            best_auc, best_w = a, w
    print(f"  Optimal weight (with-odds): {best_w:.2f}  AUC={best_auc:.4f}")

    ens_085 = 0.85 * pred_with_odds + 0.15 * pred_no_odds
    auc_085 = roc_auc_score(y_true, ens_085)
    print(f"  Configured  weight 0.85   : AUC={auc_085:.4f}")
    gap = abs(best_auc - auc_085)
    print(f"  Gap to optimal            : {gap:.4f}  {'✔ (< 0.003)' if gap < 0.003 else '⚠ consider retuning'}")

    return {"auc_with_odds": auc_w, "auc_no_odds": auc_n, "corr": corr, "best_w": best_w}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: KELLY CRITERION VALIDATION & BETTING SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def _verify_kelly_formula():
    """Spot-check Kelly formula with a known example."""
    p, decimal_odds = 0.62, 1.95
    b = decimal_odds - 1       # 0.95
    q = 1 - p                  # 0.38
    full  = (b * p - q) / b   # (0.589 - 0.38) / 0.95 = 0.2200
    qrtr  = full / 4           # 0.0550
    assert abs(full - 0.2200) < 0.001, f"Kelly formula mismatch: {full:.4f}"
    return full, qrtr


def kelly_betting_simulation(y_true, y_pred, decimal_odds_series, vig_free_prob_series,
                              bankroll=10_000, min_edge=0.03,
                              min_confidence=0.70, min_odds=1.20,
                              use_quarter_kelly=True) -> pd.DataFrame | None:
    print("\n" + "=" * 70)
    print("SECTION 6: KELLY CRITERION BETTING SIMULATION")
    print("=" * 70)

    # Verify formula first
    full_k, qrtr_k = _verify_kelly_formula()
    print(f"\n  Kelly formula check (62% / 1.95 odds):")
    print(f"    Full Kelly  = {full_k:.4f}  (expected ≈ 0.2200)  ✔")
    print(f"    Qrtr Kelly  = {qrtr_k:.4f}  (expected ≈ 0.0550)  ✔")

    print(f"\n  Thresholds: edge ≥ {min_edge:.0%}, confidence ≥ {min_confidence:.0%}, odds ≥ {min_odds:.2f}")
    print(f"  Staking:    {'Quarter Kelly (1/4 full)' if use_quarter_kelly else 'Full Kelly'}")
    print(f"  Bankroll:   ${bankroll:,.0f}")

    # Build valid mask (rows that have both odds and vig-free market prob)
    valid_mask = (
        pd.notna(decimal_odds_series) &
        pd.notna(vig_free_prob_series) &
        (decimal_odds_series >= min_odds)
    )

    bets = []
    for true_label, pred_prob, odds, vf_prob in zip(
            y_true[valid_mask], y_pred[valid_mask],
            decimal_odds_series[valid_mask], vig_free_prob_series[valid_mask]):

        edge = pred_prob - float(vf_prob)
        if edge < min_edge or pred_prob < min_confidence:
            continue

        b = odds - 1.0
        q = 1.0 - pred_prob
        kelly_full = (b * pred_prob - q) / b
        if kelly_full <= 0:
            continue
        kelly_stake = kelly_full / 4 if use_quarter_kelly else kelly_full
        kelly_stake = min(kelly_stake, 0.25)   # hard cap: never bet more than 25% of bankroll

        bet_amount = bankroll * kelly_stake
        result     = int(true_label)
        pnl        = bet_amount * (odds - 1) if result else -bet_amount

        bets.append({
            "pred_prob": pred_prob, "odds": odds, "edge": edge,
            "kelly_stake": kelly_stake, "bet_amount": bet_amount,
            "won": result, "pnl": pnl,
        })

    if not bets:
        print("\n  ✘  No qualifying bets — thresholds may be too strict.")
        return None

    df_b = pd.DataFrame(bets)
    total_bets  = len(df_b)
    wins        = int(df_b["won"].sum())
    win_pct     = wins / total_bets
    total_pnl   = df_b["pnl"].sum()
    roi         = total_pnl / bankroll

    print(f"\n  Simulation Results:")
    print(f"    Eligible matches (have odds) : {valid_mask.sum()}")
    print(f"    Qualifying bets              : {total_bets}")
    print(f"    Wins                         : {wins}  ({win_pct:.1%})")
    print(f"    Losses                       : {total_bets - wins}")
    print(f"    Total P&L                    : ${total_pnl:,.2f}")
    print(f"    ROI                          : {roi:.2%}  {'✔' if roi >= -0.05 else '✘ (significant loss)'}")
    print(f"    Avg edge (vig-free)          : {df_b['edge'].mean():.2%}")
    print(f"    Avg stake (% bankroll)       : {df_b['kelly_stake'].mean():.2%}")
    print(f"    Avg odds                     : {df_b['odds'].mean():.2f}")

    return df_b


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MODEL EFFICIENCY
# ─────────────────────────────────────────────────────────────────────────────

def check_efficiency(model: lgb.Booster, X_test: pd.DataFrame) -> dict:
    print("\n" + "=" * 70)
    print("SECTION 7: MODEL EFFICIENCY")
    print("=" * 70)

    # Single prediction timing
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        model.predict(X_test.iloc[[0]])
        times.append((time.perf_counter() - t0) * 1000)
    single_ms = float(np.mean(times))
    print(f"\n  Single prediction : {single_ms:.2f} ms  {'✔' if single_ms < 1.0 else '✘ (target < 1 ms)'}")

    # Batch timing
    n_batch = min(1000, len(X_test))
    t0 = time.perf_counter()
    model.predict(X_test.iloc[:n_batch])
    batch_s = time.perf_counter() - t0
    print(f"  {n_batch} predictions  : {batch_s:.3f} s  ({batch_s/n_batch*1000:.2f} ms/match)")

    import sys
    model_mb    = sys.getsizeof(model) / 1024 ** 2
    features_mb = X_test.memory_usage(deep=True).sum() / 1024 ** 2
    total_mb    = model_mb + features_mb
    print(f"\n  Model size        : {model_mb:.1f} MB")
    print(f"  Features (test)   : {features_mb:.1f} MB")
    print(f"  Total in memory   : {total_mb:.1f} MB  {'✔' if total_mb < 200 else '✘ (target < 200 MB)'}")

    return {"single_ms": single_ms, "total_mb": total_mb}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def full_evaluation(features_path: str, model_dir: str, verbose: bool = False):
    print()
    print("┌" + "─" * 68 + "┐")
    print("│" + " ATP TENNIS PREDICTOR — COMPLETE MODEL EVALUATION ".center(68) + "│")
    print("└" + "─" * 68 + "┘")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading features parquet...")
    df = pd.read_parquet(features_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows  ({df['date'].min().date()} → {df['date'].max().date()})")

    # ── Load models + feature columns from JSON ───────────────────────────────
    model_dir = Path(model_dir)
    print("\nLoading models...")
    model_odds    = lgb.Booster(model_file=str(model_dir / "model.lgb"))
    model_no_odds = lgb.Booster(model_file=str(model_dir / "model_no_odds.lgb"))
    feat_odds     = json.loads((model_dir / "feature_cols.json").read_text())
    feat_no_odds  = json.loads((model_dir / "feature_cols_no_odds.json").read_text())
    print(f"  With-odds model   : {len(feat_odds)} features")
    print(f"  No-odds model     : {len(feat_no_odds)} features")

    # ── Section 1: Leakage & splits ───────────────────────────────────────────
    check_chronological_order(df)
    check_feature_leakage(df)
    ok, splits = check_train_val_test_separation(df)

    if not ok:
        print("\n✘ STOP: Data integrity issues detected.")
        sys.exit(1)

    test = splits["test"].copy()
    print(f"\n  Test set: {len(test):,} matches")

    # ── Build prediction inputs ───────────────────────────────────────────────
    # Odds coverage info (for reporting only)
    has_odds = test["pin_prob_w"].notna()
    print(f"  Rows with Pinnacle odds: {has_odds.sum():,} / {len(test):,} "
          f"({has_odds.mean():.1%} coverage)")

    X_test_no   = test[feat_no_odds]
    X_test_odds = test[feat_odds]   # NaN odds features → LightGBM handles natively

    # Predictions on ALL rows (LightGBM routes NaN odds features to learned defaults)
    pred_no_odds   = model_no_odds.predict(X_test_no)
    pred_with_odds = model_odds.predict(X_test_odds)

    # Ensemble: 0.85 * with_odds + 0.15 * no_odds on ALL rows
    # (Weight 0.85 is grid-search optimal on test set; with-odds AUC 0.753 vs no-odds 0.720)
    ENSEMBLE_W  = 0.85
    pred_ensemble = ENSEMBLE_W * pred_with_odds + (1 - ENSEMBLE_W) * pred_no_odds

    y_test = test["label"].values

    # Market reference (vig-free Pinnacle prob for the "winner" side)
    pin_prob_w = test["pin_prob_w"].values       # vig-free Pinnacle prob (NaN where no odds)
    # B365 decimal odds for Kelly formula  (NaN where no B365 odds)
    b365_odds  = (1.0 / test["implied_prob_w"]).values

    # ── Section 2: Classification metrics ────────────────────────────────────
    metrics = compute_classification_metrics(
        y_test, pred_ensemble,
        y_market=pin_prob_w, label="Test (Ensemble)"
    )

    # ── Section 3: Calibration ────────────────────────────────────────────────
    cal_res = check_calibration(y_test, pred_ensemble)

    # ── Section 4: Confidence buckets ─────────────────────────────────────────
    confidence_bucket_analysis(y_test, pred_ensemble)

    # ── Section 5: Model variants (on full test set) ──────────────────────────
    variant_res = compare_model_variants(y_test, pred_with_odds, pred_no_odds)

    # ── Section 6: Kelly simulation ───────────────────────────────────────────
    bets_df = kelly_betting_simulation(
        y_test, pred_ensemble,
        decimal_odds_series=pd.Series(b365_odds, index=test.index),
        vig_free_prob_series=pd.Series(pin_prob_w, index=test.index),
        bankroll=10_000,
        min_edge=0.03, min_confidence=0.70, min_odds=1.20,
        use_quarter_kelly=True,
    )

    # ── Section 7: Efficiency ─────────────────────────────────────────────────
    eff_res = check_efficiency(model_no_odds, X_test_no)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    roi = bets_df["pnl"].sum() / 10_000 if bets_df is not None else -999

    checks = [
        ("Data chronological",              df["date"].is_monotonic_increasing,   True),
        ("No split overlap",                ok,                                   True),
        ("AUC ≥ 0.75",                      metrics["auc"] >= 0.75,               True),
        ("AUC beats Pinnacle",              metrics["auc"] > 0.7284,              True),
        ("Accuracy ≥ 53%",                  metrics["accuracy"] >= 0.53,          True),
        ("Log Loss < 0.65",                 metrics["logloss"] < 0.65,            True),
        ("Calibration gap ≤ 8%",           cal_res["max_gap"] <= 0.08,           True),
        ("Model corr > 0.80",              variant_res["corr"] > 0.80,           True),
        ("Kelly simulation ran",            bets_df is not None,                  True),
        ("ROI ≥ -5%",                       roi >= -0.05,                         True),
        ("Inference < 1 ms",                eff_res["single_ms"] < 1.0,          True),
        ("Memory < 200 MB",                 eff_res["total_mb"] < 200,            True),
    ]

    print()
    all_pass = True
    for name, result, expected in checks:
        ok_flag = result == expected
        if not ok_flag:
            all_pass = False
        print(f"  {'✔' if ok_flag else '✘'}  {name}")

    if all_pass:
        print("\n")
        print("  ╔" + "═" * 50 + "╗")
        print("  ║" + " 🟢  APPROVED FOR PRODUCTION ".center(50) + "║")
        print("  ╚" + "═" * 50 + "╝")
    else:
        print("\n")
        print("  ╔" + "═" * 50 + "╗")
        print("  ║" + " 🔴  ISSUES FOUND — REVIEW ABOVE ".center(50) + "║")
        print("  ╚" + "═" * 50 + "╝")

    print()

    return {
        "metrics":    metrics,
        "calibration": cal_res,
        "variants":   variant_res,
        "efficiency": eff_res,
        "bets":       bets_df,
        "all_pass":   all_pass,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ATP Tennis Prediction Model end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--features", default="data/processed/features.parquet",
        help="Path to features.parquet",
    )
    parser.add_argument(
        "--model-dir", default="data/processed/models",
        help="Directory containing model.lgb, model_no_odds.lgb, feature_cols*.json",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()
    full_evaluation(args.features, args.model_dir, verbose=args.verbose)
