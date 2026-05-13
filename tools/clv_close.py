"""Record closing Pinnacle odds for open CLV bets, grade results, and print stats.

Typical workflow:
    1. predict_upcoming.py stamps "open" odds when a bet qualifies.
    2. Run this script ~30 min before each match to capture the closing line:
           uv run python clv_close.py
    3. After the match finishes, grade the result:
           uv run python clv_close.py --grade "Sinner" "Alcaraz" "Sinner"

Usage:
    uv run python clv_close.py
    uv run python clv_close.py --date 2026-05-12
    uv run python clv_close.py --manual "Sinner" "Alcaraz" 1.62 2.40
    uv run python clv_close.py --grade "Sinner" "Alcaraz" "Sinner"
    uv run python clv_close.py --summary
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import unicodedata
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import clv  # noqa: E402 — needs sys.path update above

# ── Odds API helpers (duplicated from predict_upcoming.py; see module docstring) ──

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


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def fetch_live_odds(api_key: str) -> dict[tuple[str, str], dict]:
    """Return odds keyed by (home_team_lower, away_team_lower) from The Odds API.

    Values: {psw, psl} (Pinnacle decimal odds, winner/loser from home perspective).
    Mirrors predict_upcoming.py's fetch_live_odds() exactly.
    """
    if not api_key:
        return {}

    base = "https://api.the-odds-api.com/v4"
    all_odds: dict[tuple[str, str], dict] = {}

    try:
        r = requests.get(f"{base}/sports", params={"apiKey": api_key}, timeout=10)
        r.raise_for_status()
        active_keys = [
            s["key"] for s in r.json()
            if s.get("active") and "tennis_atp" in s["key"]
        ]
    except Exception as e:
        print(f"  Warning: could not fetch sport list ({e}).")
        return {}

    print(f"  Active ATP sport keys: {active_keys}")

    for sport_key in active_keys:
        try:
            r = requests.get(
                f"{base}/sports/{sport_key}/odds",
                params={
                    "apiKey":     api_key,
                    "regions":    "eu",
                    "markets":    "h2h",
                    "oddsFormat": "decimal",
                    "bookmakers": "pinnacle,bet365",
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
    """Try to find odds for a player pair using last-name fuzzy matching.

    Mirrors predict_upcoming.py's _match_odds() exactly.
    """
    p1_last = strip_accents(p1_raw.split()[-1]).lower()
    p2_last = strip_accents(p2_raw.split()[-1]).lower()

    for (home, away), odds in odds_map.items():
        home_last = home.split()[-1]
        away_last = away.split()[-1]
        if p1_last in home_last and p2_last in away_last:
            return {
                "psw":   odds.get("psw"),
                "psl":   odds.get("psl"),
                "b365w": odds.get("b365w"),
                "b365l": odds.get("b365l"),
            }
        if p2_last in home_last and p1_last in away_last:
            # Swap so p1 perspective is consistent with how the bet was opened
            return {
                "psw":   odds.get("psl"),
                "psl":   odds.get("psw"),
                "b365w": odds.get("b365l"),
                "b365l": odds.get("b365w"),
            }
    return None


# ── CLV grading helper ────────────────────────────────────────────────────────

def _grade_bet(p1: str, p2: str, winner: str, date: str) -> None:
    """Update actual_winner in the CLV log and mark won (1) or lost (0)."""
    log_path = clv.CLV_LOG
    if not log_path.exists():
        print("CLV log not found — nothing to grade.")
        return

    df = pd.read_csv(log_path)
    mask = (df["date"].astype(str) == date) & (df["p1"] == p1) & (df["p2"] == p2)

    if not mask.any():
        print(f"No CLV entry found for {p1} vs {p2} on {date}.")
        return

    winner_norm = winner.strip().lower()
    p1_norm     = p1.strip().lower()
    p2_norm     = p2.strip().lower()

    if winner_norm in p1_norm or p1_norm in winner_norm:
        actual = p1
        won    = 1
    elif winner_norm in p2_norm or p2_norm in winner_norm:
        actual = p2
        won    = 0
    else:
        # Winner string didn't match either player — store as-is, no won flag
        actual = winner
        won    = None
        print(f"  Warning: winner '{winner}' didn't clearly match '{p1}' or '{p2}'. "
              f"Storing raw name; won/lost not set.")

    df.loc[mask, "actual_winner"] = actual
    if won is not None:
        # We treat p1 as the side we bet on (consistent with record_open convention)
        # "won" = 1 means the p1 side won
        if "won" not in df.columns:
            df["won"] = None
        df.loc[mask, "won"] = won

    df.to_csv(log_path, index=False)
    result_str = "WON" if won == 1 else ("LOST" if won == 0 else "recorded")
    print(f"Graded {p1} vs {p2} on {date}: winner={actual} — {result_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record closing Pinnacle odds for open CLV bets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date to close bets for (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--manual", nargs=4, metavar=("P1", "P2", "CLOSE_P1_ODDS", "CLOSE_P2_ODDS"),
        help="Manually supply closing odds for a single match.",
    )
    parser.add_argument(
        "--grade", nargs=3, metavar=("P1", "P2", "WINNER"),
        help="Grade a result: supply both player names and the winner's name.",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print aggregate CLV stats from the log and exit.",
    )
    args = parser.parse_args()

    target_date = datetime.date.fromisoformat(args.date) if args.date else datetime.date.today()
    date_str    = str(target_date)

    # ── Summary-only mode ─────────────────────────────────────────────────────
    if args.summary:
        clv.summary()
        return

    # ── Grade mode ────────────────────────────────────────────────────────────
    if args.grade:
        p1, p2, winner = args.grade
        _grade_bet(p1, p2, winner, date_str)
        clv.summary()
        return

    # ── Read open (unclosed) bets for the target date ─────────────────────────
    log_path = clv.CLV_LOG
    if not log_path.exists():
        print("CLV log not found — no open bets to close.")
        return

    df = pd.read_csv(log_path)
    open_mask = (
        (df["date"].astype(str) == date_str)
        & df["open_p1_odds"].notna()
        & (df["close_p1_odds"].isna() | (df["close_p1_odds"].astype(str).str.strip() == ""))
    )
    open_bets = df[open_mask].copy()

    if open_bets.empty:
        print(f"No open (unclosed) bets found for {date_str}.")
        clv.summary()
        return

    print(f"Found {len(open_bets)} open bet(s) to close for {date_str}.")
    for _, row in open_bets.iterrows():
        print(f"  - {row['p1']} vs {row['p2']}  "
              f"(open: {row['open_p1_odds']:.2f} / {row['open_p2_odds']:.2f})")

    # ── Manual override ───────────────────────────────────────────────────────
    if args.manual:
        p1_arg, p2_arg, c1_str, c2_str = args.manual
        try:
            close_p1 = float(c1_str)
            close_p2 = float(c2_str)
        except ValueError:
            print(f"Error: closing odds must be numbers, got '{c1_str}' and '{c2_str}'.")
            sys.exit(1)

        print(f"\nManual close: {p1_arg} vs {p2_arg} — {close_p1:.2f} / {close_p2:.2f}")
        clv.record_close(p1_arg, p2_arg, close_p1, close_p2, date=target_date)
        print(f"  Recorded close for {p1_arg} vs {p2_arg}.")
        print()
        clv.summary()
        return

    # ── Fetch live odds and match to open bets ────────────────────────────────
    api_key = os.getenv("ODDS_API_KEY", "")
    if not api_key:
        print("ODDS_API_KEY not set. Use --manual to enter closing odds by hand.")
        sys.exit(1)

    print("\nFetching live Pinnacle odds from The Odds API...")
    odds_map = fetch_live_odds(api_key)

    if not odds_map:
        print("No odds returned. The matches may already be live/finished, "
              "or an API error occurred.")
        sys.exit(1)

    matched   = 0
    unmatched = []

    for _, row in open_bets.iterrows():
        p1   = str(row["p1"])
        p2   = str(row["p2"])
        found = _match_odds(odds_map, p1, p2)

        if found and found.get("psw") and found.get("psl"):
            close_p1 = float(found["psw"])
            close_p2 = float(found["psl"])
            clv.record_close(p1, p2, close_p1, close_p2, date=target_date)
            print(f"  Matched & closed: {p1} vs {p2} — "
                  f"{close_p1:.2f} / {close_p2:.2f}")
            matched += 1
        else:
            unmatched.append((p1, p2))

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nClosed {matched}/{len(open_bets)} bet(s).")

    if unmatched:
        print(f"\nUnmatched bets ({len(unmatched)}) — odds not found in API response:")
        for p1, p2 in unmatched:
            print(f"  - {p1} vs {p2}")
        print("\nTip: use --manual to enter closing odds by hand for unmatched bets:")
        for p1, p2 in unmatched:
            print(f'  uv run python clv_close.py --manual "{p1}" "{p2}" <close_p1> <close_p2>'
                  f' --date {date_str}')

    print()
    clv.summary()


if __name__ == "__main__":
    main()
