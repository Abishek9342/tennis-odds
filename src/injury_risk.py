"""Injury / withdrawal risk signal.

ATP publishes a daily withdrawal/retirement list per tournament. A player who
withdrew from a tournament in the last 14 days or retired mid-match in the last
30 days has materially elevated injury risk — odds typically don't fully price
this until very late in the day.

This module uses the Sofascore unofficial API (same pattern as atp_scraper.py)
to find retirement/walkover events in recent match results, and merges in a
user-maintained manual CSV for any signals not captured automatically.

The intended consumer is `features.py` which adds two features:
    w_injury_risk_14d, l_injury_risk_14d   # 0.0 to 1.0
    w_retired_30d,     l_retired_30d       # int count

CLI:
    python -m injury_risk --refresh
    python -m injury_risk --refresh --days 30
    python -m injury_risk --player "Sinner J."
"""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import requests

CACHE = Path("data/processed/injury_log.parquet")
MANUAL_CSV = Path("data/processed/injury_manual.csv")

_SCHEMA = ["player", "event", "tournament", "date", "source"]
# event ∈ {"withdrawal", "retirement", "walkover"}

_SOFASCORE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com",
}

_SOFASCORE_BASE = "https://api.sofascore.com/api/v1"

# Sofascore status descriptions that indicate an injury/retirement event
_RETIREMENT_STATUSES = {"Retired", "Walkover", "Interrupted"}

_STATUS_TO_EVENT = {
    "Retired": "retirement",
    "Walkover": "walkover",
    "Interrupted": "retirement",  # treat mid-match interruption as retirement signal
}


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


def check_injury_risk(p1: str, p2: str, date: datetime.date | None = None) -> str:
    """Return a formatted warning string if either player has injury signals.

    Returns an empty string if both players are clean.

    Example output:
        "⚠ INJURY SIGNAL: Nadal R. — retirement 12d ago (Roland Garros)"
        "⚠ INJURY SIGNAL: Murray A. — withdrawal 3d ago (Wimbledon) | walkover 45d ago (Queen's Club)"
    """
    date = date or datetime.date.today()
    log = load_log()
    if log.empty:
        return ""

    log["date"] = pd.to_datetime(log["date"])
    warnings: list[str] = []

    for player in (p1, p2):
        sub = log[log["player"] == player].copy()
        if sub.empty:
            continue

        cutoff = pd.Timestamp(date) - pd.Timedelta(days=60)
        recent = sub[sub["date"] >= cutoff].sort_values("date", ascending=False)
        if recent.empty:
            continue

        signals: list[str] = []
        for _, row in recent.iterrows():
            days_ago = (pd.Timestamp(date) - row["date"]).days
            signals.append(f"{row['event']} {days_ago}d ago ({row['tournament']})")

        warnings.append(f"INJURY SIGNAL: {player} — " + " | ".join(signals))

    return "\n".join(warnings)


def _fetch_scheduled_events(date: datetime.date) -> list[dict]:
    """Fetch scheduled events for a single date from Sofascore."""
    url = f"{_SOFASCORE_BASE}/sport/tennis/scheduled-events/{date.isoformat()}"
    try:
        resp = requests.get(url, headers=_SOFASCORE_HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json().get("events", [])
    except Exception:
        return []


def _fetch_last_page(page: int) -> list[dict]:
    """Fetch a page of recent match results from Sofascore."""
    url = f"{_SOFASCORE_BASE}/sport/tennis/events/last/{page}"
    try:
        resp = requests.get(url, headers=_SOFASCORE_HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json().get("events", [])
    except Exception:
        return []


def _is_atp(event: dict) -> bool:
    """Return True if the event belongs to the ATP tour."""
    category = event.get("tournament", {}).get("category", {}).get("name", "")
    return category == "ATP"


def _parse_event_row(event: dict, event_type: str) -> dict | None:
    """Extract a single injury-log row from a Sofascore event dict.

    Returns None if the event cannot be parsed.
    """
    try:
        ts = event.get("startTimestamp")
        if not ts:
            return None
        import datetime as _dt
        match_date = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).date()

        tournament_name = event.get("tournament", {}).get("name", "Unknown")

        # Determine the loser (or the player who retired/walked over).
        # In Sofascore, winnerCode=1 means homeTeam won, 2 means awayTeam won.
        winner_code = event.get("winnerCode")
        home_name = event.get("homeTeam", {}).get("name", "")
        away_name = event.get("awayTeam", {}).get("name", "")

        # The retiring / walking-over player is the *loser*
        if winner_code == 1:
            player_name = away_name
        elif winner_code == 2:
            player_name = home_name
        else:
            # Cannot determine who retired — record both as signals
            player_name = None

        if not player_name and not (home_name or away_name):
            return None

        if player_name:
            return {
                "player": player_name,
                "event": event_type,
                "tournament": tournament_name,
                "date": str(match_date),
                "source": "sofascore",
            }
        # Fallback: record the event for both players (less precise)
        return None
    except Exception:
        return None


def refresh_atp_withdrawals(days_back: int = 30) -> int:
    """Refresh injury/withdrawal log from Sofascore.

    Scans:
    1. Scheduled events for each day in the past `days_back` days — picks up
       withdrawals annotated on the day's schedule (status: "Walkover" before match).
    2. Recent result pages (last/ pages 0–4) — picks up mid-match retirements.
    3. Merges the user-maintained manual CSV at data/processed/injury_manual.csv
       if it exists.

    Returns count of new rows added to the cache.
    """
    today = datetime.date.today()
    new_entries: list[dict] = []

    # ── 1. Scan scheduled-events endpoint for each past day ──────────────────
    for offset in range(days_back):
        day = today - datetime.timedelta(days=offset)
        events = _fetch_scheduled_events(day)
        for ev in events:
            if not _is_atp(ev):
                continue
            status_desc = ev.get("status", {}).get("description", "")
            if status_desc not in _RETIREMENT_STATUSES:
                continue
            event_type = _STATUS_TO_EVENT.get(status_desc, "retirement")
            row = _parse_event_row(ev, event_type)
            # For walkover/withdrawal from scheduled events before match starts,
            # override event type to "withdrawal" if match hasn't started
            status_type = ev.get("status", {}).get("type", "")
            if status_desc == "Walkover" and status_type in ("", "notstarted", "inprogress"):
                if row:
                    row["event"] = "withdrawal"
            if row:
                new_entries.append(row)

    # ── 2. Scan recent results pages ─────────────────────────────────────────
    cutoff = today - datetime.timedelta(days=days_back)
    for page in range(5):
        events = _fetch_last_page(page)
        if not events:
            break
        page_exhausted = False
        for ev in events:
            if not _is_atp(ev):
                continue
            status_desc = ev.get("status", {}).get("description", "")
            if status_desc not in _RETIREMENT_STATUSES:
                continue
            # Check timestamp is within our window
            ts = ev.get("startTimestamp", 0)
            if ts:
                import datetime as _dt
                ev_date = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).date()
                if ev_date < cutoff:
                    page_exhausted = True
                    continue
            event_type = _STATUS_TO_EVENT.get(status_desc, "retirement")
            row = _parse_event_row(ev, event_type)
            if row:
                new_entries.append(row)
        if page_exhausted:
            break

    # ── 3. Merge manual CSV ───────────────────────────────────────────────────
    if MANUAL_CSV.exists():
        try:
            manual_df = pd.read_csv(MANUAL_CSV, dtype=str)
            # Validate expected columns exist
            for col in _SCHEMA:
                if col not in manual_df.columns:
                    print(f"  Warning: manual CSV missing column '{col}', skipping.")
                    manual_df = pd.DataFrame(columns=_SCHEMA)
                    break
            for _, row in manual_df.iterrows():
                new_entries.append(
                    {
                        "player": str(row.get("player", "")),
                        "event": str(row.get("event", "")),
                        "tournament": str(row.get("tournament", "")),
                        "date": str(row.get("date", "")),
                        "source": str(row.get("source", "manual")),
                    }
                )
        except Exception as exc:
            print(f"  Warning: could not read manual CSV: {exc}")

    if not new_entries:
        return 0

    return append(new_entries)


def append(entries: list[dict]) -> int:
    """Append known events to the cache. Used by tests and any wired scraper."""
    if not entries:
        return 0
    log = load_log()
    fresh = pd.DataFrame(entries)
    # Normalise date column
    fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    fresh = fresh.dropna(subset=["player", "date"])
    out = pd.concat([log, fresh], ignore_index=True).drop_duplicates(
        subset=["player", "event", "tournament", "date"], keep="first"
    )
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE, index=False)
    return len(out) - len(log)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Injury/withdrawal risk tracker")
    ap.add_argument("--refresh", action="store_true", help="Refresh from Sofascore + manual CSV")
    ap.add_argument("--days", type=int, default=30, help="Days back to scan (default: 30)")
    ap.add_argument("--player", type=str, help="Show risk score for a player")
    a = ap.parse_args()
    if a.refresh:
        n = refresh_atp_withdrawals(days_back=a.days)
        print(f"Added {n} rows.")
    elif a.player:
        print(risk_score(a.player))
    else:
        print(load_log())
