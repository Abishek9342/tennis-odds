"""Injury / withdrawal risk signal.

ATP publishes a daily withdrawal/retirement list per tournament. A player who
withdrew from a tournament in the last 14 days or retired mid-match in the last
30 days has materially elevated injury risk — odds typically don't fully price
this until very late in the day.

This module is a stub: the data source is wired but the parser must be
finished against the live HTML structure (which changes occasionally). The
intended consumer is `features.py` which adds two features:
    w_injury_risk_14d, l_injury_risk_14d   # 0.0 to 1.0
    w_retired_30d,     l_retired_30d       # int count

CLI:
    python -m injury_risk --refresh
    python -m injury_risk --player "Sinner J."
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd

CACHE = Path("data/processed/injury_log.parquet")

_SCHEMA = ["player", "event", "tournament", "date", "source"]
# event ∈ {"withdrawal", "retirement", "walkover"}


def load_log() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    return pd.DataFrame(columns=_SCHEMA)


def risk_score(player: str, on_date: datetime.date | None = None) -> dict:
    """Return injury-related features for `player` as of `on_date`.

    Returns:
        dict with keys:
            injury_risk_14d  ∈ [0, 1]   # 1.0 = withdrew in last 14 days
            retired_30d      ∈ int      # number of in-match retirements in last 30d
            walkover_60d     ∈ int      # walkovers given in last 60d
    """
    log = load_log()
    if log.empty:
        return {"injury_risk_14d": 0.0, "retired_30d": 0, "walkover_60d": 0}

    on_date = on_date or datetime.date.today()
    log["date"] = pd.to_datetime(log["date"])
    sub = log[log["player"] == player]

    cutoff_14 = pd.Timestamp(on_date) - pd.Timedelta(days=14)
    cutoff_30 = pd.Timestamp(on_date) - pd.Timedelta(days=30)
    cutoff_60 = pd.Timestamp(on_date) - pd.Timedelta(days=60)

    withdrawals = sub[(sub["event"] == "withdrawal") & (sub["date"] >= cutoff_14)]
    retired     = sub[(sub["event"] == "retirement") & (sub["date"] >= cutoff_30)]
    walkovers   = sub[(sub["event"] == "walkover")   & (sub["date"] >= cutoff_60)]

    return {
        "injury_risk_14d": min(1.0, len(withdrawals) / 1.0),  # cap at 1
        "retired_30d":     int(len(retired)),
        "walkover_60d":    int(len(walkovers)),
    }


def refresh_atp_withdrawals() -> int:
    """Refresh from atptour.com withdrawal pages. STUB — wire to live page.

    The page structure varies per tournament; the implementation should:
      1. Iterate over recent tournaments (last 90 days)
      2. Hit /en/tournaments/{slug}/{year}/draws
      3. Parse the WD / RET annotations in the draw HTML
      4. Append new rows to CACHE

    Returns count of new rows added.
    """
    raise NotImplementedError(
        "Wire this to the atptour.com tournament withdrawal pages.\n"
        "See module docstring for the expected schema."
    )


def append(entries: list[dict]) -> int:
    """Append known events to the cache. Used by tests and any wired scraper."""
    if not entries:
        return 0
    log = load_log()
    fresh = pd.DataFrame(entries)
    out = pd.concat([log, fresh], ignore_index=True).drop_duplicates(
        subset=["player", "event", "tournament", "date"], keep="first"
    )
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    return len(out) - len(log)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--player",  type=str)
    a = ap.parse_args()
    if a.refresh:
        n = refresh_atp_withdrawals()
        print(f"Added {n} rows.")
    elif a.player:
        print(risk_score(a.player))
    else:
        print(load_log())
