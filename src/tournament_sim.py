"""Monte Carlo tournament simulator.

Given a single-elimination bracket of N=2^k players and a function `prob(p1, p2)`
that returns P(p1 beats p2), runs S simulations and returns each player's
probability of winning the tournament, reaching the final, semis, etc.

This is the standalone-differentiator capability: futures markets price a
tournament-winner outright (e.g. "Sinner to win Roland Garros @ 3.50"). A
sharp simulator converts your per-match model into a futures probability,
revealing where the future market disagrees with you. Most retail bettors
have no way to do this.

Example:
    from tournament_sim import simulate
    bracket = ["Sinner J.", "Alcaraz C.", "Djokovic N.", "Zverev A.", ...]   # power of 2
    res = simulate(bracket, prob_fn=lambda a, b: model_prob(a, b), n_sims=20000)
    print(res.sort_values("p_win", ascending=False).head(10))
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
import pandas as pd


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def simulate(
    bracket: list[str],
    prob_fn: Callable[[str, str], float],
    n_sims: int = 10_000,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate `n_sims` runs of a single-elimination bracket.

    Args:
        bracket: list of players in bracket order. Adjacent pairs play in R1.
                 Length must be a power of 2.
        prob_fn: prob_fn(p1, p2) → P(p1 wins). Should be deterministic and
                 cheap to call (cached if backed by an ML model).
        n_sims:  number of Monte Carlo runs.
        seed:    optional RNG seed for reproducible output.

    Returns:
        DataFrame indexed by player with columns:
            p_r2, p_qf, p_sf, p_f, p_win  (probability of reaching each stage)
    """
    n = len(bracket)
    if not _is_power_of_two(n):
        raise ValueError(f"Bracket size must be a power of 2, got {n}")

    rng = random.Random(seed)
    rounds = int(np.log2(n))
    # stage_counts[player][round_index] = times this player reached that round
    reached = {p: np.zeros(rounds + 1, dtype=int) for p in bracket}
    won = {p: 0 for p in bracket}

    # Cache pairwise probabilities — typical bracket has at most n*(n-1)/2 pairs
    pair_cache: dict[tuple[str, str], float] = {}
    def _prob(a: str, b: str) -> float:
        if (a, b) in pair_cache:
            return pair_cache[(a, b)]
        p = float(prob_fn(a, b))
        pair_cache[(a, b)] = p
        pair_cache[(b, a)] = 1.0 - p
        return p

    for _ in range(n_sims):
        alive = list(bracket)
        # Stage 0: everyone "reached" R1
        for p in alive:
            reached[p][0] += 1

        round_idx = 0
        while len(alive) > 1:
            round_idx += 1
            next_alive = []
            for i in range(0, len(alive), 2):
                a, b = alive[i], alive[i + 1]
                p_a_wins = _prob(a, b)
                winner = a if rng.random() < p_a_wins else b
                next_alive.append(winner)
                reached[winner][round_idx] += 1
            alive = next_alive
        won[alive[0]] += 1

    # Build result frame. reached has rounds+1 entries: index i means
    # "still alive at the start of round i+1". p_win is tracked separately.
    label_map_by_rounds = {
        6: ["r1", "r2", "r3", "r4", "qf", "sf", "f"],     # 64-draw
        5: ["r1", "r2", "r3", "qf", "sf", "f"],            # 32-draw
        4: ["r1", "r2", "qf", "sf", "f"],                  # 16-draw
        3: ["r1", "qf", "sf", "f"],                        # 8-draw
        2: ["sf", "f"],                                    # 4-draw
        1: ["f"],                                          # 2-draw
    }
    label_map = label_map_by_rounds.get(rounds, [f"r{i+1}" for i in range(rounds)])
    # Always emits rounds entries; champion is the separate p_win column.

    rows = []
    for p in bracket:
        row = {"player": p}
        # reached has rounds entries that correspond 1:1 with label_map
        for stage_i, name in enumerate(label_map):
            row[f"p_{name}"] = reached[p][stage_i] / n_sims
        row["p_win"] = won[p] / n_sims
        rows.append(row)

    out = pd.DataFrame(rows).set_index("player")
    return out.sort_values("p_win", ascending=False)


def futures_edge(sim_result: pd.DataFrame, market_odds: dict[str, float]) -> pd.DataFrame:
    """Compare simulated win probabilities against a futures market.

    Args:
        sim_result:  output of `simulate()`.
        market_odds: dict of {player: decimal_odds} for the outright winner market.

    Returns:
        DataFrame with model_p_win, implied_p_win, edge, kelly_full (capped at 0).
    """
    rows = []
    for player, odds in market_odds.items():
        if player not in sim_result.index:
            continue
        mp = float(sim_result.loc[player, "p_win"])
        ip = 1.0 / odds
        edge = mp - ip
        b = odds - 1.0
        q = 1.0 - mp
        kelly = max((b * mp - q) / b, 0.0) if b > 0 else 0.0
        rows.append({
            "player": player,
            "model_p_win": round(mp, 4),
            "implied_p_win": round(ip, 4),
            "decimal_odds": odds,
            "edge": round(edge, 4),
            "kelly_full": round(kelly, 4),
        })
    return pd.DataFrame(rows).sort_values("edge", ascending=False).reset_index(drop=True)
