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
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE = Path("data/processed/serve_return_stats.parquet")

_REQUIRED_METRICS = [
    "first_serve_pct", "ace_rate", "double_fault_rate",
    "first_serve_won_pct", "second_serve_won_pct",
    "break_points_saved_pct", "return_points_won_pct",
    "break_points_converted_pct",
    "service_games_won_pct", "return_games_won_pct",
]


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
# Refresh — placeholder; wire to your source of choice (Sackmann CSV / Sofascore)
# ---------------------------------------------------------------------------

def refresh_from_sackmann(years: list[int]) -> None:
    """Pull serve/return aggregates from Jeff Sackmann's tennis_atp repo.

    Repo: https://github.com/JeffSackmann/tennis_atp
    Files: atp_matches_YYYY.csv contain per-match w_ace, w_df, w_svpt, w_1stIn,
           w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced (and l_ equivalents).
    Aggregating these by player + surface + season gives all _REQUIRED_METRICS.

    This is wired as a stub; to enable:
        1. Clone tennis_atp into data/raw/sackmann/
        2. Implement the aggregation below
        3. Run `python -m serve_return --refresh`
    """
    raise NotImplementedError(
        "Sackmann refresh is wired as a stub. See module docstring for the\n"
        "expected schema and the columns to aggregate from atp_matches_YYYY.csv."
    )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    a = ap.parse_args()
    if a.refresh:
        refresh_from_sackmann(a.years)
    else:
        df = load_cache()
        print(f"Cache rows: {len(df)}")
        if not df.empty:
            print(df.head())
