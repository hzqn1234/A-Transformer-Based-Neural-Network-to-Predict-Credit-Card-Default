from __future__ import annotations

import argparse

from credit_default_prediction.training.amex_lightgbm import train_series_lgbm_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the AMEX row-level LightGBM used for target-history features.")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=4500)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=24)
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--max-train-customers", type=int, default=None)
    parser.add_argument("--max-test-customers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_series_lgbm_cv(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        folds=args.folds,
        seed=args.seed,
        rounds=args.rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        num_threads=args.num_threads,
        nrows=args.nrows,
        max_train_customers=args.max_train_customers,
        max_test_customers=args.max_test_customers,
    )
    print(summary)


if __name__ == "__main__":
    main()
