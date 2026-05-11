"""Probability calibration layer.

Trains a Platt (sigmoid) or isotonic calibrator on the validation window so
that model output `p` better reflects the true win frequency. The fitted
calibrator is saved as JSON for inference-time use without depending on
sklearn at predict time.

CLI:
    python main.py calibrate --method isotonic
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


VAL_START = pd.Timestamp("2024-01-01")
VAL_END   = pd.Timestamp("2025-06-30")


class _Calibrator:
    """Lightweight wrapper that mimics sklearn's transform() at inference."""

    def __init__(self, method: str, payload: dict):
        self.method = method
        self.payload = payload

    def transform(self, probs):
        probs = np.asarray(probs, dtype=float)
        if self.method == "platt":
            a = self.payload["a"]
            b = self.payload["b"]
            return 1.0 / (1.0 + np.exp(-(a * probs + b)))
        if self.method == "isotonic":
            xs = np.asarray(self.payload["x"], dtype=float)
            ys = np.asarray(self.payload["y"], dtype=float)
            return np.interp(probs, xs, ys, left=ys[0], right=ys[-1])
        raise ValueError(f"Unknown calibrator method: {self.method}")


def load_calibrator(path: Path) -> _Calibrator:
    obj = json.loads(Path(path).read_text())
    return _Calibrator(obj["method"], obj["payload"])


def _val_probs(model_dir: Path, features_parquet: Path):
    df = pd.read_parquet(features_parquet)
    df["date"] = pd.to_datetime(df["date"])
    val = df[(df["date"] >= VAL_START) & (df["date"] <= VAL_END)].copy()
    if val.empty:
        raise RuntimeError("No validation window data — cannot fit calibrator.")

    cols_full = json.loads((model_dir / "feature_cols.json").read_text())
    cols_no   = json.loads((model_dir / "feature_cols_no_odds.json").read_text())
    booster_full = lgb.Booster(model_file=str(model_dir / "model.lgb"))
    booster_no   = lgb.Booster(model_file=str(model_dir / "model_no_odds.lgb"))

    weight_path = model_dir / "ensemble_weight.json"
    w = 0.65
    if weight_path.exists():
        w = float(json.loads(weight_path.read_text()).get("weight_odds", w))

    p_full = booster_full.predict(val[cols_full].astype(float))
    p_no   = booster_no.predict(val[cols_no].astype(float))
    p_ens  = w * p_full + (1.0 - w) * p_no
    return p_ens, val["label"].astype(int).values


def fit_and_save(model_dir: Path, features_parquet: Path, method: str = "isotonic") -> Path:
    """Fit a calibrator on validation predictions, save JSON, print metrics."""
    p, y = _val_probs(model_dir, features_parquet)

    pre_brier = brier_score_loss(y, p)
    pre_ll    = log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))
    print(f"Pre-calibration: Brier={pre_brier:.5f}  LogLoss={pre_ll:.5f}")

    if method == "platt":
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(p.reshape(-1, 1), y)
        a = float(lr.coef_[0][0])
        b = float(lr.intercept_[0])
        payload = {"a": a, "b": b}
        p_cal = 1.0 / (1.0 + np.exp(-(a * p + b)))
    elif method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        # Sample the function on a fine grid for cheap inference-time interp
        xs = np.linspace(0.0, 1.0, 201)
        ys = iso.predict(xs)
        payload = {"x": xs.tolist(), "y": ys.tolist()}
        p_cal = np.interp(p, xs, ys)
    else:
        raise ValueError(f"Unknown method: {method}")

    post_brier = brier_score_loss(y, p_cal)
    post_ll    = log_loss(y, np.clip(p_cal, 1e-6, 1 - 1e-6))
    print(f"Post-calibration ({method}): Brier={post_brier:.5f}  LogLoss={post_ll:.5f}")
    print(f"  Δ Brier:   {post_brier - pre_brier:+.5f}")
    print(f"  Δ LogLoss: {post_ll - pre_ll:+.5f}")

    out_path = model_dir / "calibrator.json"
    out_path.write_text(json.dumps({"method": method, "payload": payload}))
    print(f"Saved → {out_path}")
    return out_path
