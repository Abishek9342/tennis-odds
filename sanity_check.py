"""End-to-end smoke test.

Run with `python main.py sanity` or directly:
    python sanity_check.py

Verifies:
  1. Required artifacts exist (raw, features, both models, feature col files)
  2. Models load
  3. A single prediction completes without raising
  4. Ensemble weight + calibrator files (if present) are well-formed
  5. No production-tainted-model flag is silently in place

Exits 0 on success, 1 on any failure. Designed to be cheap (~10s) so it can
run before live prediction or as a CI gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "processed"
MODEL_DIR = DATA / "models"

REQUIRED = [
    DATA / "raw.parquet",
    DATA / "features.parquet",
    MODEL_DIR / "model.lgb",
    MODEL_DIR / "model_no_odds.lgb",
    MODEL_DIR / "feature_cols.json",
    MODEL_DIR / "feature_cols_no_odds.json",
]


def _ok(msg):    print(f"  ✓ {msg}")
def _fail(msg):  print(f"  ✗ {msg}")
def _warn(msg):  print(f"  ! {msg}")


def run() -> bool:
    print("Tennis-odds sanity check\n")
    fails: list[str] = []

    print("[1] Artifact presence")
    for p in REQUIRED:
        if p.exists():
            _ok(str(p.relative_to(ROOT)))
        else:
            _fail(f"missing: {p.relative_to(ROOT)}")
            fails.append(str(p))

    if fails:
        print("\nStopping: required artifacts missing.")
        return False

    print("\n[2] Production-leak guard")
    leak_flag = MODEL_DIR / "PRODUCTION_TRAINED_ON_TEST.flag"
    if leak_flag.exists():
        _warn(f"{leak_flag.name} present — this model was trained on the test holdout.")
        _warn("Any evaluation/CLV reporting against it is invalid.")
    else:
        _ok("no production-leak flag")

    print("\n[3] Model load")
    try:
        import lightgbm as lgb
        b_full = lgb.Booster(model_file=str(MODEL_DIR / "model.lgb"))
        b_no   = lgb.Booster(model_file=str(MODEL_DIR / "model_no_odds.lgb"))
        _ok(f"with-odds booster: {b_full.num_trees()} trees")
        _ok(f"no-odds booster:   {b_no.num_trees()} trees")
    except Exception as e:
        _fail(f"load failed: {e}")
        return False

    print("\n[4] Ensemble weight + calibrator")
    ew_path = MODEL_DIR / "ensemble_weight.json"
    if ew_path.exists():
        try:
            w = json.loads(ew_path.read_text()).get("weight_odds")
            _ok(f"ensemble_weight.json: weight_odds={w}")
        except Exception as e:
            _fail(f"ensemble_weight.json invalid: {e}")
            fails.append("ensemble")
    else:
        _warn("no ensemble_weight.json — using default 0.65 (run `python main.py tune-ensemble`)")

    cal_path = MODEL_DIR / "calibrator.json"
    if cal_path.exists():
        try:
            from calibration import load_calibrator
            cal = load_calibrator(cal_path)
            out = cal.transform([0.5])[0]
            _ok(f"calibrator ({cal.method}): 0.50 → {out:.3f}")
        except Exception as e:
            _fail(f"calibrator invalid: {e}")
            fails.append("calibrator")
    else:
        _warn("no calibrator.json — run `python main.py calibrate` to fit one")

    print("\n[5] Single-match smoke prediction")
    try:
        import pandas as pd
        import model
        raw = pd.read_parquet(DATA / "raw.parquet")
        recent = raw.sort_values("date").iloc[-1]
        p1, p2 = recent["winner"], recent["loser"]
        out = model.predict(MODEL_DIR, p1, p2, "Hard", DATA / "raw.parquet")
        prob = out["p1_win_prob"]
        if not (0.0 <= prob <= 1.0):
            _fail(f"prob out of range: {prob}")
            fails.append("predict-range")
        else:
            _ok(f"predict({p1} vs {p2}, Hard) → {prob:.3f}")
    except Exception as e:
        _fail(f"prediction raised: {e}")
        fails.append("predict-call")

    print()
    if fails:
        print(f"FAILED: {len(fails)} issue(s)")
        return False
    print("PASSED")
    return True


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
