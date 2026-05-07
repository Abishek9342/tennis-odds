"""Auto-verify predictions against actual ATP match results.

Fetches recent completed ATP scores, matches them against predictions_log.csv,
fills in the 'Actual Winner' column, and prints a running accuracy summary.

Usage:
    uv run python check_results.py
    uv run python check_results.py --log predictions_log.csv
"""

import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _last_name(full: str) -> str:
    """Extract normalised last name from any name format."""
    parts = strip_accents(str(full).strip()).split()
    return parts[-1].lower() if parts else ""


def match_result_to_prediction(result_p1: str, result_p2: str,
                                pred_p1: str, pred_p2: str) -> str | None:
    """
    Try to match a result row (full ATP names) to a prediction row (internal names).
    Returns the internal name of the winner if matched, else None.
    """
    r1_last = _last_name(result_p1)
    r2_last = _last_name(result_p2)
    p1_last = _last_name(pred_p1)
    p2_last = _last_name(pred_p2)

    if r1_last in p1_last or p1_last in r1_last:
        if r2_last in p2_last or p2_last in r2_last:
            return pred_p1   # result_p1 won (listed first = winner on ATP site)
    if r1_last in p2_last or p2_last in r1_last:
        if r2_last in p1_last or p1_last in r2_last:
            return pred_p2   # result_p1 = our p2
    return None


def main():
    parser = argparse.ArgumentParser(description="Auto-fill match results into predictions log")
    parser.add_argument("--log", default="predictions_log.csv", help="Path to predictions CSV")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"No predictions log found at {log_path}. Run predict_upcoming.py first.")
        return

    log_df = pd.read_csv(log_path)
    if "Actual Winner" not in log_df.columns:
        log_df["Actual Winner"] = ""

    # Only process rows where Actual Winner is still blank
    pending_mask = log_df["Actual Winner"].isna() | (log_df["Actual Winner"].astype(str).str.strip() == "")
    pending = log_df[pending_mask].copy()

    if pending.empty:
        print("All predictions already have results filled in.")
    else:
        print(f"{len(pending)} predictions awaiting results. Fetching ATP scores...")

        from atp_scraper import fetch_atp_scores
        try:
            scores_df = fetch_atp_scores()
            print(f"  Fetched {len(scores_df)} recent results.\n")
        except Exception as e:
            print(f"  Could not fetch ATP scores: {e}")
            print("  Fill in 'Actual Winner' manually in the log file.")
            scores_df = pd.DataFrame()

        if not scores_df.empty and "player1" in scores_df.columns:
            filled = 0
            for idx, pred_row in pending.iterrows():
                p1 = str(pred_row.get("Player 1", ""))
                p2 = str(pred_row.get("Player 2", ""))

                for _, score_row in scores_df.iterrows():
                    winner = match_result_to_prediction(
                        str(score_row.get("player1", "")),
                        str(score_row.get("player2", "")),
                        p1, p2,
                    )
                    if winner:
                        log_df.at[idx, "Actual Winner"] = winner
                        filled += 1
                        break

            print(f"Auto-filled {filled} / {len(pending)} results.")
        else:
            print("  No score data available for auto-fill. Fill manually.")

    # ── Accuracy summary ──────────────────────────────────────────────────────
    done = log_df[
        log_df["Actual Winner"].notna() &
        (log_df["Actual Winner"].astype(str).str.strip() != "")
    ].copy()

    if done.empty:
        print("\nNo completed predictions to score yet.")
    else:
        # Determine what the model predicted as favourite
        if "Favourite" in done.columns:
            done["predicted_winner"] = done["Favourite"].astype(str).str.extract(r"^(.+?)\s*\(")[0]
        elif "P1 Win % (no odds)" in done.columns:
            done["predicted_winner"] = done.apply(
                lambda r: r["Player 1"] if float(str(r.get("P1 Win % (no odds)", "0%")).rstrip("%") or 0) > 50 else r["Player 2"],
                axis=1,
            )
        else:
            done["predicted_winner"] = done["Player 1"]

        done["correct"] = (
            done["predicted_winner"].apply(_last_name) ==
            done["Actual Winner"].apply(_last_name)
        )

        total    = len(done)
        correct  = done["correct"].sum()
        accuracy = correct / total

        print("\n" + "=" * 50)
        print("RUNNING PREDICTION ACCURACY")
        print("=" * 50)
        print(f"  Total predictions scored : {total}")
        print(f"  Correct                  : {correct}")
        print(f"  Accuracy                 : {accuracy:.1%}")

        # Breakdown by confidence bucket
        if "P1 Win % (no odds)" in done.columns:
            def conf_bucket(row):
                fav_pct_str = str(row.get("Favourite", ""))
                try:
                    pct = float(fav_pct_str.split("(")[1].rstrip("%)")) / 100
                except Exception:
                    return "unknown"
                if pct >= 0.80: return "80%+"
                if pct >= 0.70: return "70-80%"
                if pct >= 0.60: return "60-70%"
                return "50-60%"

            done["conf_bucket"] = done.apply(conf_bucket, axis=1)
            grp = done.groupby("conf_bucket").agg(
                predictions=("correct", "count"),
                correct=("correct", "sum"),
            ).assign(accuracy=lambda d: (d["correct"] / d["predictions"]).map("{:.1%}".format))
            print("\n  By confidence bucket:")
            print(grp[["predictions", "correct", "accuracy"]].to_string())

        # Per-match table
        print("\n  Per-match results:")
        show_cols = ["Date", "Player 1", "Player 2", "Surface",
                     "Favourite", "Actual Winner", "correct"]
        show_cols = [c for c in show_cols if c in done.columns]
        print(done[show_cols].to_string(index=False))

    # Save updated log
    log_df.to_csv(log_path, index=False)
    print(f"\nLog updated → {log_path}")


if __name__ == "__main__":
    main()
