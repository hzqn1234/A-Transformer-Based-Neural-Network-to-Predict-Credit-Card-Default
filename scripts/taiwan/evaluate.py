from __future__ import annotations

import argparse

import pandas as pd

from credit_default_prediction.metrics import binary_auc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Taiwan predictions with AUC.")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.labels)
    predictions = pd.read_csv(args.predictions)
    merged = labels.merge(predictions, on="customer_ID", how="inner")
    score = binary_auc(merged["target"], merged["prediction"])
    print({"auc": score, "rows": len(merged)})


if __name__ == "__main__":
    main()
