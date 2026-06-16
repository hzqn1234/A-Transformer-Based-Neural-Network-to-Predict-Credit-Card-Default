from __future__ import annotations

import argparse

from credit_default_prediction.data.amex import prepare_amex_sequences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AMEX sequence data for the paper models.")
    parser.add_argument(
        "--raw-dir",
        required=True,
        help=(
            "Directory with AMEX train_data.csv/test_data.csv. If train.feather/test.feather "
            "are present and --nrows is not set, those denoised intermediates are used."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory for processed feather/csv files.")
    parser.add_argument("--nrows", type=int, default=None, help="Optional row cap for quick checks.")
    parser.add_argument("--max-train-customers", type=int, default=None)
    parser.add_argument("--max-test-customers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = prepare_amex_sequences(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        nrows=args.nrows,
        max_train_customers=args.max_train_customers,
        max_test_customers=args.max_test_customers,
    )
    print(metadata)


if __name__ == "__main__":
    main()
