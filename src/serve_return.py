"""Serve / return statistics module.

Tennis-data.co.uk does NOT publish serve/return splits; the strongest free
public source is the Jeff Sackmann tennis_atp GitHub repo (Match Charting
Project) and per-player aggregates from atptour.com `/stats` pages or
Sofascore player-stats endpoints.

This module exposes a small surface area so feature engineering can pull
serve/return numbers without each downstream caller learning the source.

Adding ~10 features here typically buys 3-5 AUC points on tennis models:
    first_serve_pct, ace_rate, double_fault_rate, first_serve_won_pct,
    second_serve_won_pct, break_points_saved_pct, return_points_won_pct,
    break_points_converted_pct, service_games_won_pct, return_games_won_pct

Schema (per row in the cache):
    player_name, surface, season, matches,
    + each metric above as a float

CLI to refresh the cache (manual, since rate-limited):
    python -m serve_return --refresh
    python -m serve_return --refresh --years 2022 2023 2024
    python -m serve_return --show "Sinner J."
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path("data/processed/serve_return_stats.parquet")

_REQUIRED_METRICS = [
    "first_serve_pct", "ace_rate", "double_fault_rate",
    "first_serve_won_pct", "second_serve_won_pct",
    "break_points_saved_pct", "return_points_won_pct",
    "break_points_converted_pct",
    "service_games_won_pct", "return_games_won_pct",
]

_DEFAULT_YEARS = list(range(2020, 2026))

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


# ---------------------------------------------------------------------------
# Public read API — unchanged so downstream callers are not broken
# ---------------------------------------------------------------------------

def load_cache() -> pd.DataFrame:
    """Return the on-disk cache; empty DataFrame if not yet built."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    return pd.DataFrame(columns=["player_name", "surface", "season", "matches"] + _REQUIRED_METRICS)


def get_player_stats(player: str, surface: str | None = None,
                     season: int | None = None) -> dict:
    """Return latest serve/return stats for `player`, optionally filtered by
    surface / season. Returns NaN dict if no record exists yet.

    The diff features expected by the model are:
        first_serve_won_diff = p1.first_serve_won_pct - p2.first_serve_won_pct
        return_points_diff   = p1.return_points_won_pct - p2.return_points_won_pct
        break_saved_diff     = p1.break_points_saved_pct - p2.break_points_saved_pct
    """
    df = load_cache()
    if df.empty:
        return {m: float("nan") for m in _REQUIRED_METRICS}

    sub = df[df["player_name"] == player]
    if surface:
        sub = sub[sub["surface"] == surface]
    if season:
        sub = sub[sub["season"] == season]
    if sub.empty:
        return {m: float("nan") for m in _REQUIRED_METRICS}

    sub = sub.sort_values(["season", "matches"], ascending=[False, False])
    return {m: float(sub.iloc[0][m]) for m in _REQUIRED_METRICS if m in sub.columns}


def build_diff_features(p1: str, p2: str, surface: str | None = None) -> dict:
    """Return the diff features needed for inference (p1 - p2)."""
    s1 = get_player_stats(p1, surface)
    s2 = get_player_stats(p2, surface)
    out = {}
    for m in _REQUIRED_METRICS:
        v1, v2 = s1.get(m, float("nan")), s2.get(m, float("nan"))
        out[f"sr_{m}_diff"] = (v1 - v2) if (pd.notna(v1) and pd.notna(v2)) else float("nan")
    return out


# ---------------------------------------------------------------------------
# Safe division helper
# ---------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> float:
    """Return num/den, or NaN if denominator is zero or either is NaN."""
    if pd.isna(num) or pd.isna(den) or den == 0:
        return float("nan")
    return num / den


# ---------------------------------------------------------------------------
# Sackmann CSV aggregation
# ---------------------------------------------------------------------------

def refresh_from_sackmann(data_dir: str = "data/raw/sackmann",
                           years: list[int] | None = None) -> None:
    """Pull serve/return aggregates from Jeff Sackmann's tennis_atp CSV files.

    Repo: https://github.com/JeffSackmann/tennis_atp
    Files: atp_matches_YYYY.csv

    Parameters
    ----------
    data_dir:
        Directory containing atp_matches_YYYY.csv files.
        Default: data/raw/sackmann
    years:
        List of seasons to process. Default: 2020–2025.
    """
    if years is None:
        years = _DEFAULT_YEARS

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Directory not found: {data_path}")
        print("Clone https://github.com/JeffSackmann/tennis_atp into that path first.")
        return

    # Accumulator: {(player, surface, season): {col: running_sum, ...}}
    # We track raw sums and compute ratios at the end to avoid averaging-of-averages.
    AccKey = tuple  # (player_name, surface, season)

    # Winner-side serve columns
    W_SERVE_COLS = [
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon",
        "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    ]
    # Loser-side serve columns
    L_SERVE_COLS = [
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon",
        "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
    ]
    METADATA_COLS = ["winner_name", "loser_name", "surface", "tourney_date", "best_of"]

    # Keys in our accumulator dict (normalised names, no w_/l_ prefix)
    ACC_COLS = ["ace", "df", "svpt", "first_in", "first_won",
                "second_won", "sv_gms", "bp_saved", "bp_faced",
                "matches",
                # opponent stats for return metrics
                "opp_svpt", "opp_first_in", "opp_first_won",
                "opp_second_won", "opp_bp_saved", "opp_bp_faced", "opp_sv_gms"]

    sums: dict[AccKey, dict[str, float]] = {}

    def _get(key: AccKey) -> dict[str, float]:
        if key not in sums:
            sums[key] = {c: 0.0 for c in ACC_COLS}
        return sums[key]

    def _add_player_stats(player: str, surface: str, season: int,
                          serve_row: dict, opp_serve_row: dict) -> None:
        """Add one player's match serve stats into the accumulator."""
        key = (player, surface, season)
        acc = _get(key)
        acc["matches"] += 1

        for src_col, dst_col in [
            ("ace", "ace"), ("df", "df"), ("svpt", "svpt"),
            ("first_in", "first_in"), ("first_won", "first_won"),
            ("second_won", "second_won"), ("sv_gms", "sv_gms"),
            ("bp_saved", "bp_saved"), ("bp_faced", "bp_faced"),
        ]:
            v = serve_row.get(src_col, 0)
            if pd.notna(v):
                acc[src_col] += float(v)

        for src_col, dst_col in [
            ("svpt", "opp_svpt"), ("first_in", "opp_first_in"),
            ("first_won", "opp_first_won"), ("second_won", "opp_second_won"),
            ("bp_saved", "opp_bp_saved"), ("bp_faced", "opp_bp_faced"),
            ("sv_gms", "opp_sv_gms"),
        ]:
            v = opp_serve_row.get(src_col, 0)
            if pd.notna(v):
                acc[dst_col] += float(v)

    all_frames: list[pd.DataFrame] = []

    for year in years:
        path = data_path / f"atp_matches_{year}.csv"
        if not path.exists():
            print(f"  Skipping {year}: file not found ({path})")
            continue

        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            print(f"  Skipping {year}: could not read CSV — {exc}")
            continue

        # Normalise surface column (Sofascore/ATP use capitalised names)
        if "surface" in df.columns:
            df["surface"] = df["surface"].str.strip().str.title().fillna("Unknown")

        # Extract season from tourney_date (YYYYMMDD integer or string)
        if "tourney_date" in df.columns:
            df["season"] = pd.to_datetime(
                df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
            ).dt.year.fillna(year).astype(int)
        else:
            df["season"] = year

        # Coerce serve stat columns to numeric, ignore missing ones silently
        for col in W_SERVE_COLS + L_SERVE_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        n_matches = 0
        for _, row in df.iterrows():
            surface = str(row.get("surface", "Unknown"))
            season = int(row.get("season", year))
            winner = str(row.get("winner_name", "")).strip()
            loser = str(row.get("loser_name", "")).strip()
            if not winner or not loser:
                continue

            # Build per-player serve dicts (strip w_/l_ prefix)
            def _extract(prefix: str) -> dict[str, float]:
                mapping = {
                    "ace": f"{prefix}ace", "df": f"{prefix}df",
                    "svpt": f"{prefix}svpt", "first_in": f"{prefix}1stIn",
                    "first_won": f"{prefix}1stWon", "second_won": f"{prefix}2ndWon",
                    "sv_gms": f"{prefix}SvGms",
                    "bp_saved": f"{prefix}bpSaved", "bp_faced": f"{prefix}bpFaced",
                }
                return {k: row.get(v, 0) for k, v in mapping.items()}

            w_stats = _extract("w_")
            l_stats = _extract("l_")

            # Process winner (their serve stats vs loser's serve as opp)
            _add_player_stats(winner, surface, season, w_stats, l_stats)
            # Process loser (their serve stats vs winner's serve as opp)
            _add_player_stats(loser, surface, season, l_stats, w_stats)
            n_matches += 1

        n_players = len({k[0] for k in sums})
        print(f"Processed {year}: {n_matches} matches, {n_players} players")

    if not sums:
        print("No data processed — nothing to write.")
        return

    # ── Convert accumulated sums → per-metric ratios ─────────────────────────
    rows: list[dict] = []
    for (player, surface, season), acc in sums.items():
        svpt = acc["svpt"]
        first_in = acc["first_in"]
        second_svpt = svpt - first_in  # 2nd serve points played

        opp_svpt = acc["opp_svpt"]
        opp_first_in = acc["opp_first_in"]

        # Opponent's total service points won (first + second)
        opp_pts_won = acc["opp_first_won"] + acc["opp_second_won"]
        # Return points won = 1 - opp_pts_won / opp_svpt
        opp_pts_won_pct = _safe_div(opp_pts_won, opp_svpt)
        return_pts_won = (1 - opp_pts_won_pct) if pd.notna(opp_pts_won_pct) else float("nan")

        # Opponent's bp data for break_points_converted
        opp_bp_faced = acc["opp_bp_faced"]
        opp_bp_saved = acc["opp_bp_saved"]
        bp_converted = opp_bp_faced - opp_bp_saved
        bp_converted_pct = _safe_div(bp_converted, opp_bp_faced)

        # service_games_won_pct — Sackmann only records SvGms for the server,
        # not which ones were won. We approximate using hold rate:
        # P(hold) ≈ P(no break) which we don't have directly.
        # Best approximation: 1 - (opp_bp_converted / opp_sv_gms)
        # But opp_sv_gms is the opponent's service games, not the player's return games.
        # Use: player sv_gms = total serve games; break rate ≈ bp_faced but not won / sv_gms
        bp_faced = acc["bp_faced"]
        bp_saved = acc["bp_saved"]
        bp_against = bp_faced - bp_saved
        sv_gms = acc["sv_gms"]
        svc_games_won_pct = _safe_div(sv_gms - bp_against, sv_gms)
        # Cap at [0, 1] — rounding can push slightly outside
        if pd.notna(svc_games_won_pct):
            svc_games_won_pct = max(0.0, min(1.0, svc_games_won_pct))

        # Return games won pct — approximate as opp_sv_gms with opp breaks converted
        opp_sv_gms = acc["opp_sv_gms"]
        return_games_won_pct = _safe_div(
            opp_bp_faced - opp_bp_saved,  # breaks converted by player
            opp_sv_gms
        )
        if pd.notna(return_games_won_pct):
            return_games_won_pct = max(0.0, min(1.0, return_games_won_pct))

        rows.append({
            "player_name": player,
            "surface": surface,
            "season": season,
            "matches": int(acc["matches"]),
            "first_serve_pct": _safe_div(first_in, svpt),
            "ace_rate": _safe_div(acc["ace"], svpt),
            "double_fault_rate": _safe_div(acc["df"], svpt),
            "first_serve_won_pct": _safe_div(acc["first_won"], first_in),
            "second_serve_won_pct": _safe_div(acc["second_won"], second_svpt),
            "break_points_saved_pct": _safe_div(bp_saved, bp_faced),
            "return_points_won_pct": return_pts_won,
            "break_points_converted_pct": bp_converted_pct,
            "service_games_won_pct": svc_games_won_pct,
            "return_games_won_pct": return_games_won_pct,
        })

    result_df = pd.DataFrame(rows)

    # Merge with existing cache (new data wins for same player+surface+season)
    existing = load_cache()
    if not existing.empty:
        merge_key = ["player_name", "surface", "season"]
        existing_trimmed = existing[~existing.set_index(merge_key).index.isin(
            result_df.set_index(merge_key).index
        )]
        result_df = pd.concat([existing_trimmed, result_df], ignore_index=True)

    result_df = result_df.sort_values(["player_name", "season", "surface"]).reset_index(drop=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(CACHE, index=False)
    print(f"\nSaved {len(result_df):,} rows ({result_df['player_name'].nunique()} players) → {CACHE}")


# ---------------------------------------------------------------------------
# Sofascore player stats (lighter alternative / fallback)
# ---------------------------------------------------------------------------

def refresh_from_sofascore(player: str, surface: str | None = None) -> dict | None:
    """Fetch serve/return stats for a single player from Sofascore.

    Requires knowing the Sofascore player ID. This function resolves it via
    the search endpoint, then fetches per-season statistics.

    Returns a dict of metrics if successful, None on any failure.
    This is optional/supplemental — it does not write to the cache directly.
    Call `append_stats()` to persist the result.

    Parameters
    ----------
    player:
        Player name (any format — will be searched on Sofascore).
    surface:
        If given, filter to that surface (Sofascore does not split by surface
        in its season stats, so this is informational only).
    """
    # ── Step 1: resolve player ID via search ─────────────────────────────────
    try:
        resp = requests.get(
            f"{_SOFASCORE_BASE}/search/all",
            headers=_SOFASCORE_HEADERS,
            params={"q": player},
            timeout=15,
        )
        if resp.status_code != 200:
            return None
        hits = resp.json().get("players", [])
    except Exception:
        return None

    player_id = None
    for hit in hits:
        sport = hit.get("sport", {}).get("name", "").lower()
        if "tennis" in sport:
            player_id = hit.get("id")
            break
    if player_id is None and hits:
        player_id = hits[0].get("id")
    if not player_id:
        print(f"  Sofascore: could not find player ID for '{player}'")
        return None

    # ── Step 2: fetch current-season statistics ───────────────────────────────
    import datetime as _dt
    current_year = _dt.date.today().year
    stats_data = None
    for year in (current_year, current_year - 1):
        try:
            url = f"{_SOFASCORE_BASE}/player/{player_id}/statistics/season/{year}"
            resp = requests.get(url, headers=_SOFASCORE_HEADERS, timeout=15)
            if resp.status_code == 200:
                stats_data = resp.json()
                break
        except Exception:
            continue

    if not stats_data:
        print(f"  Sofascore: no season statistics for '{player}' (id={player_id})")
        return None

    # ── Step 3: parse — Sofascore statistics structure varies; do best-effort ─
    stats = stats_data.get("statistics", {})
    if not stats:
        stats = stats_data  # sometimes the dict IS the stats

    def _pct(val: object) -> float:
        """Convert a percentage value (0–100 or 0–1) to 0–1 float."""
        try:
            v = float(val)  # type: ignore[arg-type]
            return v / 100 if v > 1.5 else v
        except (TypeError, ValueError):
            return float("nan")

    result = {
        "player_name": player,
        "surface": surface or "All",
        "season": current_year,
        "matches": int(stats.get("matchesPlayed", stats.get("matches", 0)) or 0),
        "first_serve_pct": _pct(stats.get("firstServePercentage", float("nan"))),
        "ace_rate": _pct(stats.get("acePercentage", stats.get("aces", float("nan")))),
        "double_fault_rate": _pct(stats.get("doubleFaultPercentage", float("nan"))),
        "first_serve_won_pct": _pct(stats.get("firstServePointsWonPercentage", float("nan"))),
        "second_serve_won_pct": _pct(stats.get("secondServePointsWonPercentage", float("nan"))),
        "break_points_saved_pct": _pct(stats.get("breakPointsSavedPercentage", float("nan"))),
        "return_points_won_pct": _pct(stats.get("returnPointsWonPercentage", float("nan"))),
        "break_points_converted_pct": _pct(stats.get("breakPointsConvertedPercentage", float("nan"))),
        "service_games_won_pct": _pct(stats.get("serviceGamesWonPercentage", float("nan"))),
        "return_games_won_pct": _pct(stats.get("returnGamesWonPercentage", float("nan"))),
    }
    return result


# ---------------------------------------------------------------------------
# Refresh — placeholder; wire to your source of choice (Sackmann CSV / Sofascore)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Serve/return stats cache builder")
    ap.add_argument("--refresh", action="store_true",
                    help="Rebuild cache from Sackmann CSVs")
    ap.add_argument("--years", type=int, nargs="+", default=None,
                    help="Years to process (default: 2020–2025)")
    ap.add_argument("--data-dir", type=str, default="data/raw/sackmann",
                    help="Directory containing atp_matches_YYYY.csv files")
    ap.add_argument("--show", type=str, metavar="PLAYER",
                    help="Print stats for a player name")
    a = ap.parse_args()

    if a.refresh:
        refresh_from_sackmann(data_dir=a.data_dir, years=a.years)
    elif a.show:
        stats = get_player_stats(a.show)
        all_nan = all(pd.isna(v) for v in stats.values())
        if all_nan:
            print(f"No stats found for '{a.show}' in cache.")
        else:
            print(f"\nStats for: {a.show}")
            print("-" * 40)
            for metric, value in stats.items():
                if pd.notna(value):
                    print(f"  {metric:<35} {value:.4f}")
                else:
                    print(f"  {metric:<35} N/A")
    else:
        df = load_cache()
        print(f"Cache rows: {len(df)}")
        if not df.empty:
            print(df.head())
