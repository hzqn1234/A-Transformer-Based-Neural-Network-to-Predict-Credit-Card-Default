from __future__ import annotations

import argparse

from credit_default_prediction.data.amex_features import (
    build_model_feature_tables,
    prepare_manual_feature_blocks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AMEX manual and series-OOF feature tables.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manual = subparsers.add_parser("manual-blocks", help="Build manual aggregate feature blocks.")
    manual.add_argument("--raw-dir", required=True)
    manual.add_argument("--output-dir", required=True)
    manual.add_argument("--nrows", type=int, default=None)
    manual.add_argument("--max-customers", type=int, default=None)
    manual.add_argument("--chunk-customers", type=int, default=None)

    model = subparsers.add_parser("model-tables", help="Combine manual blocks with row-level LGBM OOF features.")
    model.add_argument("--feature-dir", required=True)
    model.add_argument("--output-dir", required=True)
    model.add_argument("--train-labels-file", required=True)
    model.add_argument("--row-oof-file", required=True)
    model.add_argument("--row-submission-file", required=True)
    model.add_argument("--sequence-length", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "manual-blocks":
        summary = prepare_manual_feature_blocks(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            nrows=args.nrows,
            max_customers=args.max_customers,
            chunk_customers=args.chunk_customers,
        )
    elif args.command == "model-tables":
        summary = build_model_feature_tables(
            feature_dir=args.feature_dir,
            output_dir=args.output_dir,
            train_labels_file=args.train_labels_file,
            row_oof_file=args.row_oof_file,
            row_submission_file=args.row_submission_file,
            sequence_length=args.sequence_length,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(summary)


if __name__ == "__main__":
    main()
