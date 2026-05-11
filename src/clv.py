"""Closing Line Value tracking.

CLV is the only model-agnostic proof of edge: bets graded against the closing
Pinnacle price. Positive CLV consistently → real edge; ROI alone is too noisy.

Workflow:
    1. predict_upcoming.py captures the live Pinnacle odds when the bet is recommended
       (these are "open" odds, several hours to days before match).
    2. close_line() — runs near match start (e.g. 15 min before) and re-fetches Pinnacle.
    3. After the match, evaluate_clv() compares model_prob vs vig-free closing_prob.

CLV per bet =
    log(closing_decimal / opening_decimal)         # log-odds change
    OR
    vig_free_closing_p1 - model_p1                  # probability-space gap

A consistently negative gap (model > market at close) means edge; positive means
the bet was on the wrong side of the steam.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path

import pandas as pd

CLV_LOG = Path("logs/clv_log.csv")
_COLS = [
    "date", "p1", "p2", "model_p1", "open_p1_odds", "open_p2_odds",
    "close_p1_odds", "close_p2_odds", "actual_winner",
    "logodds_clv_p1", "probspace_clv_p1",
]


def _vig_free(o_w: float, o_l: float) -> float:
    raw_w, raw_l = 1.0 / o_w, 1.0 / o_l
    return raw_w / (raw_w + raw_l)


def record_open(p1: str, p2: str, model_p1: float, open_p1: float, open_p2: float,
                date: datetime.date | None = None) -> None:
    """Stamp the opening price for a bet so CLV can be computed at close."""
    date = date or datetime.date.today()
    row = {c: None for c in _COLS}
    row.update({
        "date": str(date),
        "p1": p1, "p2": p2,
        "model_p1": round(float(model_p1), 4),
        "open_p1_odds": float(open_p1),
        "open_p2_odds": float(open_p2),
    })
    _append(row)


def record_close(p1: str, p2: str, close_p1: float, close_p2: float,
                 date: datetime.date | None = None) -> None:
    """Update the closing Pinnacle price and compute CLV deltas."""
    df = _read()
    if df.empty:
        print("CLV log empty — call record_open() first.")
        return
    date = str(date or datetime.date.today())
    mask = (df["date"] == date) & (df["p1"] == p1) & (df["p2"] == p2)
    if not mask.any():
        print(f"No open price recorded for {p1} vs {p2} on {date}; skipping close.")
        return

    df.loc[mask, "close_p1_odds"] = float(close_p1)
    df.loc[mask, "close_p2_odds"] = float(close_p2)

    open_p1  = df.loc[mask, "open_p1_odds"].iloc[0]
    open_p2  = df.loc[mask, "open_p2_odds"].iloc[0]
    model_p1 = df.loc[mask, "model_p1"].iloc[0]

    if pd.notna(open_p1) and pd.notna(open_p2):
        df.loc[mask, "logodds_clv_p1"]  = math.log(open_p1) - math.log(close_p1)
        df.loc[mask, "probspace_clv_p1"] = float(model_p1) - _vig_free(close_p1, close_p2)

    df.to_csv(CLV_LOG, index=False)


def summary() -> dict:
    """Print and return aggregate CLV stats over the log."""
    df = _read()
    df = df.dropna(subset=["close_p1_odds"])
    if df.empty:
        print("No closed bets in the CLV log yet.")
        return {}
    avg_log    = float(df["logodds_clv_p1"].mean())
    avg_prob   = float(df["probspace_clv_p1"].mean())
    pct_beat   = float((df["logodds_clv_p1"] > 0).mean())
    print(f"CLV samples       : {len(df)}")
    print(f"Avg log-odds CLV  : {avg_log:+.4f}  (positive = beat the close)")
    print(f"Avg prob-space gap: {avg_prob:+.4f}  (negative = model > market)")
    print(f"% bets beat close : {pct_beat:.1%}")
    return {"n": len(df), "log_clv": avg_log, "prob_gap": avg_prob, "beat_pct": pct_beat}


def _read() -> pd.DataFrame:
    if CLV_LOG.exists():
        return pd.read_csv(CLV_LOG)
    return pd.DataFrame(columns=_COLS)


def _append(row: dict) -> None:
    CLV_LOG.parent.mkdir(parents=True, exist_ok=True)
    df = _read()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(CLV_LOG, index=False)
