from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average two Taiwan prediction files.")
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--first-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p0 = pd.read_csv(args.first)
    p1 = pd.read_csv(args.second)
    merged = p0.merge(p1, on="customer_ID", suffixes=("_first", "_second"))
    w = args.first_weight
    merged["prediction"] = w * merged["prediction_first"] + (1.0 - w) * merged["prediction_second"]
    out = merged[["customer_ID", "prediction"]]
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    print(out.head())


if __name__ == "__main__":
    main()
