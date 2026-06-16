from __future__ import annotations

import argparse

from credit_default_prediction.training.amex_lightgbm import train_lgbm_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AMEX LightGBM model.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-set", choices=["manual", "series_oof"], required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=4500)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--num-threads", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_lgbm_cv(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        feature_set=args.feature_set,
        folds=args.folds,
        seed=args.seed,
        rounds=args.rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        num_threads=args.num_threads,
    )
    print(summary)


if __name__ == "__main__":
    main()
