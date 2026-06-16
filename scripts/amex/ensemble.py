from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average AMEX prediction files with fixed weights.")
    parser.add_argument("--prediction", action="append", required=True, help="CSV with customer_ID,prediction.")
    parser.add_argument("--weight", action="append", type=float, default=None)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.prediction
    weights = args.weight or [1.0 / len(paths)] * len(paths)
    if len(weights) != len(paths):
        raise ValueError("Number of --weight values must match number of --prediction files.")
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    frames = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path)[["customer_ID", "prediction"]]
        frames.append(df.rename(columns={"prediction": f"prediction_{i}"}))
    merged = reduce(lambda left, right: left.merge(right, on="customer_ID"), frames)
    merged["prediction"] = 0.0
    for i, weight in enumerate(weights):
        merged["prediction"] += weight * merged[f"prediction_{i}"]
    out = merged[["customer_ID", "prediction"]]
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    print(out.head())


if __name__ == "__main__":
    main()
