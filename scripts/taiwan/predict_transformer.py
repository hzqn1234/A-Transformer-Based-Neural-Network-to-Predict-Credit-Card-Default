from __future__ import annotations

import argparse

from credit_default_prediction.training.taiwan_transformer import predict_taiwan_transformer_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Taiwan test probabilities from trained folds.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-customers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = predict_taiwan_transformer_cv(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_file=args.output_file,
        split="test",
        batch_size=args.batch_size,
        device=args.device,
        num_workers=args.num_workers,
        max_customers=args.max_customers,
    )
    print(out.head())


if __name__ == "__main__":
    main()
