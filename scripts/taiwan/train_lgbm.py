from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from credit_default_prediction.data.sequence import ID_COLUMN, TARGET_COLUMN, read_table
from credit_default_prediction.training.taiwan_lightgbm import train_taiwan_lightgbm_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Taiwan LightGBM baseline.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=1500)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    return parser.parse_args()


def load_taiwan_tabular_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train_features = read_table(data_dir / "train.feather")
    test_features = read_table(data_dir / "test.feather")
    labels = read_table(data_dir / "train_labels.csv")[[ID_COLUMN, TARGET_COLUMN]]
    train = train_features.merge(labels, on=ID_COLUMN, how="inner")
    return train, test_features


def main() -> None:
    args = parse_args()
    train, test = load_taiwan_tabular_data(args.data_dir)
    summary = train_taiwan_lightgbm_cv(
        train_df=train,
        test_df=test,
        output_dir=args.output_dir,
        metric="auc",
        folds=args.folds,
        seed=args.seed,
        rounds=args.rounds,
        early_stopping_rounds=args.early_stopping_rounds,
        boosting="dart",
        max_depth=-1,
        min_data_in_leaf=256,
        max_bin=63,
        min_data_in_bin=256,
        tree_learner="serial",
        boost_from_average="false",
        lambda_l1=0.1,
        lambda_l2=30.0,
        num_threads=24,
        verbosity=1,
    )
    print(summary)


if __name__ == "__main__":
    main()
