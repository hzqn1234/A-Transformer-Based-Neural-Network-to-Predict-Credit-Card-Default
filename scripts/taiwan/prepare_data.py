from __future__ import annotations

import argparse

from credit_default_prediction.data.taiwan import prepare_taiwan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Taiwan credit card default dataset.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-csv", default=None, help="Optional local CSV with X1-X23 and target.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_taiwan(
        output_dir=args.output_dir,
        seed=args.seed,
        source_csv=args.source_csv,
        train_ratio=args.train_ratio,
    )
    print(metadata)


if __name__ == "__main__":
    main()
