"""Stage 2: Extensive feature engineering from raw ATP match data.

All features are computed with strict temporal ordering — no future data leaks into
any match's feature vector.

Single-pass architecture: one chronological loop builds all player-level features
(Elo, streaks, form trend, surface affinity, set dominance, H2H extended) to avoid
redundant iterations over the dataset.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

GRAND_SLAMS = {
    "Australian Open", "Roland Garros", "Wimbledon", "US Open",
    # alternate spellings in the dataset
    "Australian open", "Us Open", "U.s. Open",
    # tennis-data.co.uk uses "French Open" not "Roland Garros"
    "French Open",
}

ROUND_MAP = {
    "R128": 1, "R64": 2, "R32": 3, "R16": 4,
    "QF": 5, "SF": 6, "F": 7, "RR": 3,  # Round Robin ~ R32
}

SURFACE_MAP = {"Hard": 0, "Clay": 1, "Grass": 2, "Carpet": 3}


# ---------------------------------------------------------------------------
# Extended Player Tracker — single instance used for ALL player-level features
# ---------------------------------------------------------------------------

class _PlayerTracker:
    """
    Maintains per-player chronological history for computing all player-level
    features in one pass.

    History record per match:
        (date, surface, won, sets_won, sets_lost, opp_elo, best_of)
    H2H record per pair:
        (date, winner, surface)
    """

    def __init__(self, initial_elo: float = 1500.0, k: int = 32):
        self._initial_elo = initial_elo
        self._k = k
        self._global_elo: dict[str, float] = defaultdict(lambda: initial_elo)
        self._surface_elo: dict[tuple, float] = defaultdict(lambda: initial_elo)
        # history[player] = list of (date, surface, won, sets_won, sets_lost, opp_elo, best_of)
        self._history: dict[str, list] = defaultdict(list)
        # h2h[(p1,p2)] canonical sorted = list of (date, winner, surface)
        self._h2h: dict[tuple, list] = defaultdict(list)

    # ------------------------------------------------------------------
    # Elo queries
    # ------------------------------------------------------------------

    def get_elo(self, player: str) -> float:
        return self._global_elo[player]

    def get_surface_elo(self, player: str, surface: str) -> float:
        return self._surface_elo[(player, surface)]

    # ------------------------------------------------------------------
    # Player feature queries (call BEFORE update for any given match)
    # ------------------------------------------------------------------

    def query_player(self, player: str, date: pd.Timestamp, surface: str,
                     windows: list[int] = [10, 30, 90]) -> dict:
        """Return all player-level features for one player, strictly before `date`."""
        history = self._history[player]
        rec: dict[str, float] = {}

        # --- Days since last match ---
        if history:
            rec["days_since_last"] = float((date - history[-1][0]).days)
        else:
            rec["days_since_last"] = np.nan

        # --- Last match result ---
        if history:
            rec["last_match_won"] = float(history[-1][2])
        else:
            rec["last_match_won"] = np.nan

        # --- Streak (signed: + = winning, - = losing) ---
        streak = 0
        if history:
            outcome = history[-1][2]
            for entry in reversed(history):
                if entry[2] == outcome:
                    streak += (1 if outcome else -1)
                else:
                    break
        rec["current_streak"] = float(streak)

        # --- Last 5 wins ---
        last5 = history[-5:]
        rec["streak_5"] = float(sum(e[2] for e in last5)) if last5 else np.nan

        # --- Rolling window stats ---
        win_rate_by_window: dict[int, float] = {}
        win_rate_surf_by_window: dict[int, float] = {}

        for n in windows:
            cutoff = date - pd.Timedelta(days=n)
            recent = [e for e in history if e[0] >= cutoff]
            total = len(recent)
            wins = sum(e[2] for e in recent)
            surf_rec = [e for e in recent if e[1] == surface]
            surf_total = len(surf_rec)
            surf_wins = sum(e[2] for e in surf_rec)

            wr = wins / total if total > 0 else np.nan
            wrs = surf_wins / surf_total if surf_total > 0 else np.nan

            rec[f"win_rate_{n}d"] = wr
            rec[f"win_rate_surf_{n}d"] = wrs
            rec[f"matches_{n}d"] = float(total)
            rec[f"matches_surf_{n}d"] = float(surf_total)

            win_rate_by_window[n] = wr
            win_rate_surf_by_window[n] = wrs

        # --- Momentum: win_rate_10d / win_rate_90d ---
        # Prior = 1.0 (neutral momentum) when either window is empty. Using NaN
        # collides with LightGBM's default-NaN-direction tree branches and
        # causes train/inference drift since rookies fall into a different bin.
        wr10 = win_rate_by_window.get(10, np.nan)
        wr90 = win_rate_by_window.get(90, np.nan)
        rec["momentum"] = wr10 / wr90 if (pd.notna(wr10) and pd.notna(wr90) and wr90 > 0) else 1.0

        # --- Win rate trend: 30d − 90d --- (prior 0.0 = no trend)
        wr30 = win_rate_by_window.get(30, np.nan)
        rec["win_rate_trend"] = (wr30 - wr90) if (pd.notna(wr30) and pd.notna(wr90)) else 0.0

        # --- Consistency: std dev of recent binary results --- (prior 0.5 = max binary stddev)
        cutoff30 = date - pd.Timedelta(days=30)
        recent30 = [e for e in history if e[0] >= cutoff30]
        if len(recent30) >= 3:
            results = [float(e[2]) for e in recent30]
            rec["consistency_30d"] = float(np.std(results))
        else:
            rec["consistency_30d"] = 0.5

        # --- Surface affinity: surf win rate − overall win rate ---
        wrs30 = win_rate_surf_by_window.get(30, np.nan)
        wrs90 = win_rate_surf_by_window.get(90, np.nan)
        rec["surface_affinity_30d"] = (wrs30 - wr30) if (pd.notna(wrs30) and pd.notna(wr30)) else np.nan
        rec["surface_affinity_90d"] = (wrs90 - wr90) if (pd.notna(wrs90) and pd.notna(wr90)) else np.nan

        # --- Strength of schedule: avg opponent Elo ---
        # opp_elo is index 5 in each record
        cutoff30 = date - pd.Timedelta(days=30)
        cutoff90 = date - pd.Timedelta(days=90)
        opp_elos_30 = [e[5] for e in history if e[0] >= cutoff30]
        opp_elos_90 = [e[5] for e in history if e[0] >= cutoff90]
        rec["avg_opp_elo_30d"] = float(np.mean(opp_elos_30)) if opp_elos_30 else np.nan
        rec["avg_opp_elo_90d"] = float(np.mean(opp_elos_90)) if opp_elos_90 else np.nan

        # --- Set dominance ---
        # sets_won=index3, sets_lost=index4, best_of=index6
        for n_days, tag in [(30, "30d"), (90, "90d")]:
            cutoff = date - pd.Timedelta(days=n_days)
            recent_n = [e for e in history if e[0] >= cutoff and e[3] is not None and e[4] is not None]
            if recent_n:
                sw = np.mean([e[3] for e in recent_n])
                sl = np.mean([e[4] for e in recent_n])
                total_sets = sw + sl
                rec[f"avg_sets_won_{tag}"] = float(sw)
                rec[f"avg_sets_lost_{tag}"] = float(sl)
                rec[f"set_ratio_{tag}"] = float(sw / total_sets) if total_sets > 0 else np.nan
                # Straight sets: won and sets_lost == 0 (opponent won no sets)
                wins_n = [e for e in recent_n if e[2]]
                if wins_n:
                    straight = sum(1 for e in wins_n if e[4] == 0)
                    rec[f"straight_sets_rate_{tag}"] = float(straight / len(wins_n))
                else:
                    rec[f"straight_sets_rate_{tag}"] = np.nan
            else:
                rec[f"avg_sets_won_{tag}"] = np.nan
                rec[f"avg_sets_lost_{tag}"] = np.nan
                rec[f"set_ratio_{tag}"] = np.nan
                rec[f"straight_sets_rate_{tag}"] = np.nan

        return rec

    def query_h2h(self, p1: str, p2: str, date: pd.Timestamp, surface: str) -> dict:
        """Return H2H features strictly before `date`."""
        key = tuple(sorted([p1, p2]))
        history = self._h2h[key]
        past = [e for e in history if e[0] < date]

        n = len(past)
        if n == 0:
            return {
                "h2h_win_rate": 0.5,
                "h2h_surface": 0.5,
                "h2h_recent_2y": 0.5,
                "h2h_n_matches": 0.0,
            }

        p1_wins = sum(1 for e in past if e[1] == p1)
        h2h_rate = p1_wins / n

        surf_past = [e for e in past if e[2] == surface]
        if surf_past:
            p1_surf = sum(1 for e in surf_past if e[1] == p1)
            h2h_surf = p1_surf / len(surf_past)
        else:
            h2h_surf = 0.5

        cutoff_2y = date - pd.Timedelta(days=730)
        recent_2y = [e for e in past if e[0] >= cutoff_2y]
        if recent_2y:
            p1_recent = sum(1 for e in recent_2y if e[1] == p1)
            h2h_recent = p1_recent / len(recent_2y)
        else:
            h2h_recent = h2h_rate  # fall back to overall if no recent

        return {
            "h2h_win_rate": h2h_rate,
            "h2h_surface": h2h_surf,
            "h2h_recent_2y": h2h_recent,
            "h2h_n_matches": float(n),
        }

    # ------------------------------------------------------------------
    # Update (call AFTER querying for current match)
    # ------------------------------------------------------------------

    def update(self, winner: str, loser: str, date: pd.Timestamp, surface: str,
               w_sets, l_sets, best_of):
        w_elo_pre = self._global_elo[winner]
        l_elo_pre = self._global_elo[loser]
        w_elo_surf_pre = self._surface_elo[(winner, surface)]
        l_elo_surf_pre = self._surface_elo[(loser, surface)]

        # Update global Elo
        exp_w = 1 / (1 + 10 ** ((l_elo_pre - w_elo_pre) / 400))
        self._global_elo[winner] = w_elo_pre + self._k * (1 - exp_w)
        self._global_elo[loser]  = l_elo_pre + self._k * (0 - (1 - exp_w))

        # Update surface Elo
        exp_w_s = 1 / (1 + 10 ** ((l_elo_surf_pre - w_elo_surf_pre) / 400))
        self._surface_elo[(winner, surface)] = w_elo_surf_pre + self._k * (1 - exp_w_s)
        self._surface_elo[(loser,  surface)] = l_elo_surf_pre + self._k * (0 - (1 - exp_w_s))

        # Safe set values
        ws = float(w_sets) if pd.notna(w_sets) else None
        ls = float(l_sets) if pd.notna(l_sets) else None
        bo = float(best_of) if pd.notna(best_of) else None

        self._history[winner].append((date, surface, True,  ws, ls, l_elo_pre, bo))
        self._history[loser].append( (date, surface, False, ls, ws, w_elo_pre, bo))

        key = tuple(sorted([winner, loser]))
        self._h2h[key].append((date, winner, surface))


# ---------------------------------------------------------------------------
# Single-pass player feature builder
# ---------------------------------------------------------------------------

def _build_player_features(df: pd.DataFrame,
                            windows: list[int] = [10, 30, 90]) -> pd.DataFrame:
    """
    One chronological loop: builds Elo, streaks, form trend, surface affinity,
    set dominance, and extended H2H features for every match.
    """
    tracker = _PlayerTracker()
    rows = []

    for _, row in df.iterrows():
        w     = row["winner"]
        l     = row["loser"]
        date  = row["date"]
        surf  = str(row.get("surface") or "Hard")
        ws    = row.get("w_sets")
        ls    = row.get("l_sets")
        bo    = row.get("best_of")

        # --- Pre-match Elo ---
        w_elo      = tracker.get_elo(w)
        l_elo      = tracker.get_elo(l)
        w_elo_surf = tracker.get_surface_elo(w, surf)
        l_elo_surf = tracker.get_surface_elo(l, surf)

        # --- All player-level features ---
        wf = tracker.query_player(w, date, surf, windows)
        lf = tracker.query_player(l, date, surf, windows)
        h2h = tracker.query_h2h(w, l, date, surf)

        rec: dict = {
            # Elo
            "w_elo":           w_elo,
            "l_elo":           l_elo,
            "elo_diff":        w_elo - l_elo,
            "w_elo_surf":      w_elo_surf,
            "l_elo_surf":      l_elo_surf,
            "elo_surf_diff":   w_elo_surf - l_elo_surf,
        }

        # Prefix all winner/loser player features
        for key, val in wf.items():
            rec[f"w_{key}"] = val
        for key, val in lf.items():
            rec[f"l_{key}"] = val

        # Diff features (winner minus loser, _diff suffix for auto-negation)
        rec["streak_diff"]              = wf["current_streak"]     - lf["current_streak"]
        rec["momentum_diff"]            = _safe_diff(wf.get("momentum"),          lf.get("momentum"))
        rec["consistency_diff"]         = _safe_diff(wf.get("consistency_30d"),   lf.get("consistency_30d"))
        rec["win_rate_trend_diff"]      = _safe_diff(wf.get("win_rate_trend"),    lf.get("win_rate_trend"))
        rec["sos_30d_diff"]             = _safe_diff(wf.get("avg_opp_elo_30d"),   lf.get("avg_opp_elo_30d"))
        rec["sos_90d_diff"]             = _safe_diff(wf.get("avg_opp_elo_90d"),   lf.get("avg_opp_elo_90d"))
        rec["surface_affinity_30d_diff"]= _safe_diff(wf.get("surface_affinity_30d"), lf.get("surface_affinity_30d"))
        rec["surface_affinity_90d_diff"]= _safe_diff(wf.get("surface_affinity_90d"), lf.get("surface_affinity_90d"))
        rec["surface_exp_diff"]         = _safe_diff(wf.get("matches_surf_90d"),  lf.get("matches_surf_90d"))
        rec["set_ratio_90d_diff"]       = _safe_diff(wf.get("set_ratio_90d"),     lf.get("set_ratio_90d"))
        rec["set_ratio_30d_diff"]       = _safe_diff(wf.get("set_ratio_30d"),     lf.get("set_ratio_30d"))
        rec["dominance_diff"]           = _safe_diff(wf.get("avg_sets_won_90d"),  lf.get("avg_sets_won_90d"))

        # H2H (h2h_win_rate is for winner = "w" perspective)
        rec["h2h_win_rate_w"]  = h2h["h2h_win_rate"]
        rec["h2h_surface_w"]   = h2h["h2h_surface"]
        rec["h2h_recent_2y_w"] = h2h["h2h_recent_2y"]
        rec["h2h_n_matches"]   = h2h["h2h_n_matches"]

        rows.append(rec)

        # Update tracker AFTER recording features
        tracker.update(w, l, date, surf, ws, ls, bo)

    return pd.DataFrame(rows, index=df.index)


def _safe_diff(a, b):
    if a is None or b is None:
        return np.nan
    if pd.isna(a) or pd.isna(b):
        return np.nan
    return float(a) - float(b)


# ---------------------------------------------------------------------------
# Bookmaker odds features (vectorised)
# ---------------------------------------------------------------------------

def build_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        rec: dict[str, float] = {}

        b365w, b365l = row.get("b365w"), row.get("b365l")
        if pd.notna(b365w) and pd.notna(b365l) and b365w > 0 and b365l > 0:
            imp_w, imp_l = 1 / b365w, 1 / b365l
            rec["implied_prob_w"] = imp_w
            rec["implied_prob_l"] = imp_l
            rec["overround"]      = imp_w + imp_l - 1
        else:
            rec["implied_prob_w"] = np.nan
            rec["implied_prob_l"] = np.nan
            rec["overround"]      = np.nan

        psw, psl = row.get("psw"), row.get("psl")
        if pd.notna(psw) and pd.notna(psl) and psw > 0 and psl > 0:
            raw_w, raw_l = 1 / psw, 1 / psl
            total = raw_w + raw_l
            pin_w = raw_w / total
            rec["pin_prob_w"] = pin_w
            rec["pin_prob_l"] = 1.0 - pin_w
            if pd.notna(rec.get("implied_prob_w")) and rec["overround"] > 0:
                b365_norm_w = rec["implied_prob_w"] / (rec["implied_prob_w"] + rec["implied_prob_l"])
                rec["b365_vs_pin"] = b365_norm_w - pin_w
            else:
                rec["b365_vs_pin"] = np.nan
        else:
            rec["pin_prob_w"] = np.nan
            rec["pin_prob_l"] = np.nan
            rec["b365_vs_pin"] = np.nan

        maxw, maxl = row.get("maxw"), row.get("maxl")
        if pd.notna(maxw) and pd.notna(maxl) and maxw > 0 and maxl > 0:
            raw_w, raw_l = 1 / maxw, 1 / maxl
            max_w = raw_w / (raw_w + raw_l)
            rec["max_prob_w"] = max_w
            rec["max_prob_l"] = 1.0 - max_w
        else:
            rec["max_prob_w"] = np.nan
            rec["max_prob_l"] = np.nan

        rows.append(rec)
    return pd.DataFrame(rows, index=df.index)


# ---------------------------------------------------------------------------
# Rank features (vectorised)
# ---------------------------------------------------------------------------

def build_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    # Use an expanding max so that rank percentiles at any row only reflect
    # the highest rank number seen UP TO that point in time (no future leakage).
    row_max = df[["w_rank", "l_rank"]].max(axis=1)
    expanding_max = row_max.expanding().max()

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        wr, lr = row.get("w_rank"), row.get("l_rank")
        max_rank = expanding_max.iloc[i]
        rec: dict[str, float] = {
            "w_rank": float(wr) if pd.notna(wr) else np.nan,
            "l_rank": float(lr) if pd.notna(lr) else np.nan,
        }
        if pd.notna(wr) and pd.notna(lr) and wr > 0 and lr > 0:
            rec["rank_diff"]       = float(lr - wr)
            rec["log_rank_w"]      = float(np.log1p(wr))
            rec["log_rank_l"]      = float(np.log1p(lr))
            rec["log_rank_diff"]   = float(np.log1p(lr) - np.log1p(wr))
            rec["rank_pct_w"]      = float(wr / max_rank) if pd.notna(max_rank) and max_rank > 0 else np.nan
            rec["rank_pct_l"]      = float(lr / max_rank) if pd.notna(max_rank) and max_rank > 0 else np.nan
        else:
            for c in ["rank_diff", "log_rank_w", "log_rank_l", "log_rank_diff",
                      "rank_pct_w", "rank_pct_l"]:
                rec[c] = np.nan
        rows.append(rec)
    return pd.DataFrame(rows, index=df.index)


# ---------------------------------------------------------------------------
# Tournament context features (fully vectorised)
# ---------------------------------------------------------------------------

def build_tournament_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    out["is_grand_slam"] = df["tournament"].apply(
        lambda t: 1.0 if any(gs.lower() in str(t).lower() for gs in GRAND_SLAMS) else 0.0
    )
    out["is_best_of_5"] = df["best_of"].apply(
        lambda b: 1.0 if pd.notna(b) and float(b) == 5 else 0.0
    )
    out["round_num"] = df["round"].apply(
        lambda r: float(ROUND_MAP.get(str(r).strip(), np.nan))
    )
    out["surface_code"] = df["surface"].apply(
        lambda s: float(SURFACE_MAP.get(str(s).strip(), np.nan))
    )
    return out


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def build_features(raw_parquet: Path, out: Path,
                   windows: list[int] = [10, 30, 90]) -> pd.DataFrame:
    """Build the full feature matrix from raw match data and save to Parquet."""
    df = pd.read_parquet(raw_parquet)
    df = df.sort_values("date").reset_index(drop=True)

    print("Building player features (single pass: Elo + streaks + form + sets + H2H)...")
    player_feats = _build_player_features(df, windows)

    print("Building odds features...")
    odds = build_odds_features(df)

    print("Building rank features...")
    ranks = build_rank_features(df)

    print("Building tournament context features...")
    context = build_tournament_context_features(df)

    # Combine
    base = df[["date", "winner", "loser", "surface", "round", "best_of", "tournament"]].copy()
    combined = pd.concat([base, player_feats, odds, ranks, context], axis=1)

    # Fill H2H NaN with 0.5 uniformly (prevents NaN-pattern leakage)
    for h2h_col in ["h2h_win_rate_w", "h2h_surface_w", "h2h_recent_2y_w"]:
        if h2h_col in combined.columns:
            combined[h2h_col] = combined[h2h_col].fillna(0.5)

    # -----------------------------------------------------------------------
    # One row per match: deterministic hash selects p1/p2 assignment
    # -----------------------------------------------------------------------
    diff_cols = [c for c in combined.columns if c.endswith("_diff")]
    w_cols = [c for c in combined.columns
              if c.startswith("w_") and not c.startswith("w_elo")]
    l_cols = [c for c in combined.columns
              if c.startswith("l_") and not c.startswith("l_elo")]

    # Probability pairs to swap for label=0 rows
    prob_w_cols = [c for c in combined.columns if c.endswith("_prob_w")]

    # H2H cols that need flipping (directional, winner-perspective)
    h2h_flip_cols = ["h2h_win_rate_w", "h2h_surface_w", "h2h_recent_2y_w"]

    def _flip_row(src: pd.Series) -> pd.Series:
        """Return a copy of src with all directional features flipped to loser-as-p1."""
        row = src.copy()

        # Negate all *_diff columns
        for col in diff_cols:
            if col in row.index and pd.notna(row[col]):
                row[col] = -row[col]

        # Swap w_* / l_* player pairs (excluding w_elo / l_elo — handled below)
        for wc, lc in zip(w_cols, l_cols):
            if wc in row.index and lc in row.index:
                row[wc], row[lc] = src[lc], src[wc]

        # Swap Elo absolute values
        for elo_col in ["w_elo", "l_elo", "w_elo_surf", "l_elo_surf"]:
            if elo_col in row.index:
                opp = elo_col.replace("w_", "l_") if elo_col.startswith("w_") else elo_col.replace("l_", "w_")
                if opp in src.index:
                    row[elo_col] = src[opp]

        # Swap probability _w / _l pairs
        for wc in prob_w_cols:
            lc = wc[:-2] + "_l"
            if wc in row.index and lc in row.index:
                row[wc], row[lc] = src[lc], src[wc]

        # implied_prob_w / implied_prob_l
        if "implied_prob_w" in row.index and "implied_prob_l" in row.index:
            row["implied_prob_w"], row["implied_prob_l"] = src["implied_prob_l"], src["implied_prob_w"]

        # Negate b365_vs_pin
        if "b365_vs_pin" in row.index and pd.notna(row["b365_vs_pin"]):
            row["b365_vs_pin"] = -src["b365_vs_pin"]

        # Flip H2H directional columns. h2h_*_w is winner-perspective. For
        # label=0 rows p1=loser, so 1.0 - val converts to p1-perspective. At
        # inference get_live_features() queries h2h directly from p1's
        # perspective, so the training and inference distributions match.
        for col in h2h_flip_cols:
            if col in row.index:
                val = src[col]
                row[col] = 1.0 - (val if pd.notna(val) else 0.5)

        # log_rank_w / log_rank_l and rank_pct_w / rank_pct_l — swap
        for wc, lc in [("log_rank_w", "log_rank_l"), ("rank_pct_w", "rank_pct_l")]:
            if wc in row.index and lc in row.index:
                row[wc], row[lc] = src[lc], src[wc]

        return row

    rows_out = []
    for _, src in combined.iterrows():
        key = hash(f"{src['winner']}{src['loser']}{src['date']}")
        if key % 2 == 0:
            row = src.copy()
            row["label"] = 1
            row["p1"] = src["winner"]
            row["p2"] = src["loser"]
        else:
            row = _flip_row(src)
            row["label"] = 0
            row["p1"] = src["loser"]
            row["p2"] = src["winner"]
        rows_out.append(row)

    result = pd.DataFrame(rows_out).reset_index(drop=True)
    result = result.sort_values("date").reset_index(drop=True)

    label_bal = result["label"].mean()
    print(f"Label balance: {label_bal:.2f} (target ~0.50)")

    # Drop rows with >50% NaN across feature columns
    meta = {"date", "winner", "loser", "p1", "p2", "surface", "round",
            "best_of", "tournament", "label", "year"}
    feat_cols = [c for c in result.columns if c not in meta]
    nan_frac = result[feat_cols].isna().mean(axis=1)
    before = len(result)
    result = result[nan_frac <= 0.5].reset_index(drop=True)
    print(f"Dropped {before - len(result):,} rows with >50% NaN features")

    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out, index=False)
    print(f"Saved {len(result):,} rows × {len(result.columns)} cols → {out}")
    return result


# ---------------------------------------------------------------------------
# Live prediction helpers (used by model.py and app.py)
# ---------------------------------------------------------------------------

def build_tracker(raw_df: pd.DataFrame) -> _PlayerTracker:
    """Build and return a fully-populated _PlayerTracker over all of raw_df.

    Call this once and cache the result; then use get_live_features() to query
    features for any player pair without re-scanning the dataset.
    """
    tracker = _PlayerTracker()
    for _, row in raw_df.sort_values("date").iterrows():
        w    = row["winner"]
        l    = row["loser"]
        date = row["date"]
        surf = str(row.get("surface") or "Hard")
        ws   = row.get("w_sets")
        ls   = row.get("l_sets")
        bo   = row.get("best_of")
        tracker.update(w, l, date, surf, ws, ls, bo)
    return tracker


def get_live_features(
    tracker: _PlayerTracker,
    p1: str,
    p2: str,
    surface: str,
    date: pd.Timestamp | None = None,
) -> dict:
    """Return a feature dict for a hypothetical p1-vs-p2 match on surface.

    Features are in p1's perspective (w_* = p1, l_* = p2), matching the
    training data convention.  Call AFTER all historical matches have been
    fed into `tracker` via build_tracker().
    """
    if date is None:
        date = pd.Timestamp.now().normalize()

    p1_elo      = tracker.get_elo(p1)
    p2_elo      = tracker.get_elo(p2)
    p1_elo_surf = tracker.get_surface_elo(p1, surface)
    p2_elo_surf = tracker.get_surface_elo(p2, surface)

    p1f = tracker.query_player(p1, date, surface)
    p2f = tracker.query_player(p2, date, surface)
    h2h = tracker.query_h2h(p1, p2, date, surface)

    rec: dict = {
        "w_elo":         p1_elo,
        "l_elo":         p2_elo,
        "elo_diff":      p1_elo - p2_elo,
        "w_elo_surf":    p1_elo_surf,
        "l_elo_surf":    p2_elo_surf,
        "elo_surf_diff": p1_elo_surf - p2_elo_surf,
    }

    for key, val in p1f.items():
        rec[f"w_{key}"] = val
    for key, val in p2f.items():
        rec[f"l_{key}"] = val

    rec["streak_diff"]               = p1f["current_streak"] - p2f["current_streak"]
    rec["momentum_diff"]             = _safe_diff(p1f.get("momentum"),             p2f.get("momentum"))
    rec["consistency_diff"]          = _safe_diff(p1f.get("consistency_30d"),      p2f.get("consistency_30d"))
    rec["win_rate_trend_diff"]       = _safe_diff(p1f.get("win_rate_trend"),       p2f.get("win_rate_trend"))
    rec["sos_30d_diff"]              = _safe_diff(p1f.get("avg_opp_elo_30d"),      p2f.get("avg_opp_elo_30d"))
    rec["sos_90d_diff"]              = _safe_diff(p1f.get("avg_opp_elo_90d"),      p2f.get("avg_opp_elo_90d"))
    rec["surface_affinity_30d_diff"] = _safe_diff(p1f.get("surface_affinity_30d"), p2f.get("surface_affinity_30d"))
    rec["surface_affinity_90d_diff"] = _safe_diff(p1f.get("surface_affinity_90d"), p2f.get("surface_affinity_90d"))
    rec["surface_exp_diff"]          = _safe_diff(p1f.get("matches_surf_90d"),     p2f.get("matches_surf_90d"))
    rec["set_ratio_90d_diff"]        = _safe_diff(p1f.get("set_ratio_90d"),        p2f.get("set_ratio_90d"))
    rec["set_ratio_30d_diff"]        = _safe_diff(p1f.get("set_ratio_30d"),        p2f.get("set_ratio_30d"))
    rec["dominance_diff"]            = _safe_diff(p1f.get("avg_sets_won_90d"),     p2f.get("avg_sets_won_90d"))

    # H2H from p1's perspective (matches training convention of h2h_*_w)
    rec["h2h_win_rate_w"]  = h2h["h2h_win_rate"]
    rec["h2h_surface_w"]   = h2h["h2h_surface"]
    rec["h2h_recent_2y_w"] = h2h["h2h_recent_2y"]
    rec["h2h_n_matches"]   = h2h["h2h_n_matches"]

    return rec


def known_players(raw_df: pd.DataFrame) -> list[str]:
    """Return a sorted list of all player names that appear in raw_df."""
    winners = set(raw_df["winner"].dropna().unique())
    losers  = set(raw_df["loser"].dropna().unique())
    return sorted(winners | losers)


# ---------------------------------------------------------------------------
# Cold-start: rank → Elo calibration for players not in historical data
# ---------------------------------------------------------------------------

def build_rank_elo_table(feat_df: pd.DataFrame) -> "np.poly1d":
    """
    Fit Elo ≈ f(log(rank)) from historical data.

    Uses both winner and loser (rank, Elo) pairs to build a smooth polynomial
    that maps any ATP rank to an expected Elo.  Saved alongside model files so
    predict() can initialise unknown players without recomputing.

    Returns a numpy poly1d (degree-1 fit in log-rank space).
    """
    w_pairs = feat_df[["w_rank", "w_elo"]].rename(columns={"w_rank": "rank", "w_elo": "elo"})
    l_pairs = feat_df[["l_rank", "l_elo"]].rename(columns={"l_rank": "rank", "l_elo": "elo"})
    both = pd.concat([w_pairs, l_pairs]).dropna()
    both = both[(both["rank"] > 0) & (both["rank"] <= 2000)].copy()

    # Bin by rank and take median per bin to reduce noise from outliers
    both["log_rank"] = np.log1p(both["rank"])
    both["rank_bin"] = pd.cut(both["rank"], bins=100)
    summary = (
        both.groupby("rank_bin", observed=True)
        .agg(mid_log=("log_rank", "median"), med_elo=("elo", "median"))
        .dropna()
    )

    coeffs = np.polyfit(summary["mid_log"], summary["med_elo"], 1)
    return np.poly1d(coeffs)


def rank_to_elo(rank: float, poly: "np.poly1d") -> float:
    """
    Estimate Elo from ATP rank using the calibrated polynomial.

    Clamps to [1100, 2300] to avoid absurd extrapolations.
    """
    if rank is None or np.isnan(rank) or rank <= 0:
        return 1500.0
    return float(np.clip(poly(np.log1p(rank)), 1100.0, 2300.0))


def apply_cold_start_elo(
    tracker: "_PlayerTracker",
    player: str,
    rank: float,
    poly: "np.poly1d",
) -> bool:
    """
    If *player* has no match history (truly cold-start), override their Elo
    with a rank-calibrated estimate instead of the default 1500.

    Also sets surface-specific Elo to the same estimate (no surface data yet).

    Returns True if cold-start was applied.
    """
    if not pd.notna(rank) or rank <= 0:
        return False
    if len(tracker._history[player]) > 0:
        return False  # player is known — do not override

    est = rank_to_elo(rank, poly)
    tracker._global_elo[player] = est
    for surf in ("Hard", "Clay", "Grass", "Carpet"):
        tracker._surface_elo[(player, surf)] = est
    return True


def cold_start_form_features(recent_matches: list[dict]) -> dict:
    """
    Compute form features for a cold-start player from their Sofascore recent
    matches, without touching the main tracker.

    Each entry in *recent_matches* should be a dict with keys:
        date (pd.Timestamp), surface (str), won (bool),
        sets_won (int|None), sets_lost (int|None)

    Returns a flat dict of player-level features (same keys as query_player()).
    Empty dict if *recent_matches* is empty.
    """
    if not recent_matches:
        return {}

    matches = sorted(recent_matches, key=lambda m: m["date"])
    today = pd.Timestamp.now().normalize()

    # ── Days since last match ──────────────────────────────────────────────────
    rec: dict = {}
    last = matches[-1]
    rec["days_since_last"] = float((today - last["date"]).days)
    rec["last_match_won"]  = float(last["won"])

    # ── Current streak ─────────────────────────────────────────────────────────
    last_outcome = matches[-1]["won"]
    streak = 0
    for m in reversed(matches):
        if m["won"] == last_outcome:
            streak += 1 if last_outcome else -1
        else:
            break
    rec["current_streak"] = float(streak)

    # ── Rolling window win rates ───────────────────────────────────────────────
    def _window(n_days: int, surface: str | None = None):
        cutoff = today - pd.Timedelta(days=n_days)
        sub = [m for m in matches if m["date"] >= cutoff]
        if surface:
            sub = [m for m in sub if m.get("surface") == surface]
        if not sub:
            return np.nan, 0
        return sum(m["won"] for m in sub) / len(sub), len(sub)

    for nd in (10, 30, 90):
        wr, cnt = _window(nd)
        rec[f"win_rate_{nd}d"]  = wr
        rec[f"matches_{nd}d"]   = float(cnt)

    # Surface-specific — we don't know the current surface context here,
    # so we leave those as NaN (predict() will pass the correct surface)
    for nd in (10, 30, 90):
        rec[f"win_rate_surf_{nd}d"] = np.nan
        rec[f"matches_surf_{nd}d"]  = np.nan

    # ── Momentum ──────────────────────────────────────────────────────────────
    wr10 = rec.get("win_rate_10d", np.nan)
    wr90 = rec.get("win_rate_90d", np.nan)
    rec["momentum"] = wr10 / wr90 if (pd.notna(wr10) and pd.notna(wr90) and wr90 > 0) else np.nan

    # ── Win rate trend ────────────────────────────────────────────────────────
    wr30 = rec.get("win_rate_30d", np.nan)
    rec["win_rate_trend"] = (wr30 - wr90) if (pd.notna(wr30) and pd.notna(wr90)) else np.nan

    # ── Consistency ───────────────────────────────────────────────────────────
    cutoff30 = today - pd.Timedelta(days=30)
    recent30 = [m for m in matches if m["date"] >= cutoff30]
    if len(recent30) >= 3:
        rec["consistency_30d"] = float(np.std([float(m["won"]) for m in recent30]))
    else:
        rec["consistency_30d"] = np.nan

    # ── Set dominance ─────────────────────────────────────────────────────────
    for n_days, tag in ((30, "30d"), (90, "90d")):
        cutoff = today - pd.Timedelta(days=n_days)
        with_sets = [m for m in matches if m["date"] >= cutoff
                     and m.get("sets_won") is not None and m.get("sets_lost") is not None]
        if with_sets:
            sw = np.mean([m["sets_won"] for m in with_sets])
            sl = np.mean([m["sets_lost"] for m in with_sets])
            rec[f"avg_sets_won_{tag}"]  = float(sw)
            rec[f"avg_sets_lost_{tag}"] = float(sl)
            rec[f"set_ratio_{tag}"]     = float(sw / (sw + sl)) if (sw + sl) > 0 else np.nan
            wins_n = [m for m in with_sets if m["won"]]
            if wins_n:
                rec[f"straight_sets_rate_{tag}"] = float(
                    sum(1 for m in wins_n if m.get("sets_lost", 1) == 0) / len(wins_n)
                )
            else:
                rec[f"straight_sets_rate_{tag}"] = np.nan
        else:
            for k in (f"avg_sets_won_{tag}", f"avg_sets_lost_{tag}",
                      f"set_ratio_{tag}", f"straight_sets_rate_{tag}"):
                rec[k] = np.nan

    # Remaining features that require full tracker context → NaN
    for k in ("surface_affinity_30d", "surface_affinity_90d",
              "avg_opp_elo_30d", "avg_opp_elo_90d", "streak_5"):
        rec[k] = np.nan

    return rec
