"""CLI entry point for the ATP tennis odds pipeline.

Usage:
    python main.py scrape    [--years 2020 2021] [--force]
    python main.py features  [--raw data/processed/raw.parquet]
    python main.py train     [--features data/processed/features.parquet] [--splits 5]
    python main.py evaluate  [--test-years 2023 2024]
    python main.py predict   --p1 "Sinner J." --p2 "Zverev A." --surface Hard
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

DATA_DIR = Path("data")
RAW_PARQUET = DATA_DIR / "processed" / "raw.parquet"
FEATURES_PARQUET = DATA_DIR / "processed" / "features.parquet"
MODEL_DIR = DATA_DIR / "processed" / "models"


def cmd_scrape(args):
    import scraper
    paths = scraper.download_all(DATA_DIR, force=args.force)
    if args.years:
        paths = [p for p in paths if int(p.stem.split("_")[0]) in args.years]
    scraper.load_and_normalise(paths, out=RAW_PARQUET)


def cmd_features(args):
    import features
    features.build_features(raw_parquet=Path(args.raw), out=FEATURES_PARQUET)


def cmd_train(args):
    import model
    # holdout_years=[] means production mode (include test period in training)
    holdout_years = [] if args.production else None
    model.train(
        features_parquet=Path(args.features),
        out_dir=MODEL_DIR,
        n_splits=args.splits,
        holdout_years=holdout_years,
    )


def cmd_evaluate(args):
    import model
    model.evaluate(
        model_dir=MODEL_DIR,
        features_parquet=FEATURES_PARQUET,
    )


def cmd_predict(args):
    import model
    model.predict(
        model_dir=MODEL_DIR,
        p1=args.p1,
        p2=args.p2,
        surface=args.surface,
        raw_parquet=RAW_PARQUET,
        tournament=args.tournament,
        round_label=args.round,
    )


def cmd_ensemble(args):
    import model
    odds = None
    if args.b365w and args.b365l:
        odds = {"b365w": args.b365w, "b365l": args.b365l}
    if args.psw and args.psl:
        odds = odds or {}
        odds.update({"psw": args.psw, "psl": args.psl})
    model.predict_ensemble(
        model_dir=MODEL_DIR,
        p1=args.p1,
        p2=args.p2,
        surface=args.surface,
        raw_parquet=RAW_PARQUET,
        odds=odds,
        tournament=args.tournament,
        round_label=args.round,
    )


def main():
    parser = argparse.ArgumentParser(description="ATP Tennis Odds Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    # scrape
    p_scrape = sub.add_parser("scrape", help="Download raw ATP data files")
    p_scrape.add_argument("--years", type=int, nargs="+", help="Filter to specific years")
    p_scrape.add_argument("--force", action="store_true", help="Re-download existing files")

    # features
    p_feat = sub.add_parser("features", help="Build feature matrix from raw data")
    p_feat.add_argument("--raw", default=str(RAW_PARQUET), help="Path to raw.parquet")

    # train
    p_train = sub.add_parser(
        "train",
        help="Train LightGBM models (with-odds + no-odds). "
             "Core: 2000–2023-12-31, Val: 2024-01-01–2025-06-30, "
             "Test holdout: 2025-07-01–2026-04-30. "
             "Pass --production to include test period in final model.",
    )
    p_train.add_argument("--features", default=str(FEATURES_PARQUET))
    p_train.add_argument("--splits", type=int, default=5, help="CV folds")
    p_train.add_argument(
        "--production", action="store_true", default=False,
        help="Train on ALL data including test period (production deployment only).",
    )

    # evaluate
    p_eval = sub.add_parser(
        "evaluate",
        help="Evaluate model on holdout test period (2025-07-01 to 2026-04-30).",
    )

    # predict
    p_pred = sub.add_parser("predict", help="Predict a single match")
    p_pred.add_argument("--p1", required=True)
    p_pred.add_argument("--p2", required=True)
    p_pred.add_argument("--surface", required=True, choices=["Hard", "Clay", "Grass", "Carpet"])
    p_pred.add_argument("--tournament", default=None, help="Tournament name (used for Grand Slam detection)")
    p_pred.add_argument("--round",      default=None, help="Round e.g. QF, SF, F, R32")

    # ensemble
    p_ens = sub.add_parser("ensemble", help="Blended prediction (no-odds + with-odds)")
    p_ens.add_argument("--p1", required=True)
    p_ens.add_argument("--p2", required=True)
    p_ens.add_argument("--surface", required=True, choices=["Hard", "Clay", "Grass", "Carpet"])
    p_ens.add_argument("--tournament", default=None, help="Tournament name")
    p_ens.add_argument("--round",      default=None, help="Round e.g. QF, SF, F, R32")
    p_ens.add_argument("--b365w", type=float, default=None, help="Bet365 P1 odds")
    p_ens.add_argument("--b365l", type=float, default=None, help="Bet365 P2 odds")
    p_ens.add_argument("--psw",   type=float, default=None, help="Pinnacle P1 odds")
    p_ens.add_argument("--psl",   type=float, default=None, help="Pinnacle P2 odds")

    args = parser.parse_args()
    {
        "scrape":   cmd_scrape,
        "features": cmd_features,
        "train":    cmd_train,
        "evaluate": cmd_evaluate,
        "predict":  cmd_predict,
        "ensemble": cmd_ensemble,
    }[args.command](args)


if __name__ == "__main__":
    main()
