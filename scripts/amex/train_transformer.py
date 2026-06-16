from __future__ import annotations

import argparse

from credit_default_prediction.training.amex_transformer import (
    predict_transformer_cv,
    train_transformer_cv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or predict an AMEX Transformer/GRU model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train AMEX Transformer/GRU folds.")
    train.add_argument("--data-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--use-features", action="store_true")
    train.add_argument("--no-feature-complement", action="store_true")
    train.add_argument("--fixed-sequence-length", type=int, default=13)
    train.add_argument("--max-train-customers", type=int, default=None)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--transformer-layers", type=int, default=3)
    train.add_argument("--attention-heads", type=int, default=32)
    train.add_argument("--feedforward-dim", type=int, default=256)
    train.add_argument("--transformer-dropout", type=float, default=0.05)
    train.add_argument("--feature-dropout", type=float, default=0.01)
    train.add_argument("--output-dropout", type=float, default=0.1)
    train.add_argument("--feature-hidden-layers", type=int, default=3)
    train.add_argument("--positional-encoding", default="sinusoidal", choices=["none", "sinusoidal"])

    predict = subparsers.add_parser("predict", help="Predict with trained AMEX Transformer/GRU folds.")
    predict.add_argument("--data-dir", required=True)
    predict.add_argument("--checkpoint-dir", required=True)
    predict.add_argument("--output-file", required=True)
    predict.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    predict.add_argument("--batch-size", type=int, default=512)
    predict.add_argument("--num-workers", type=int, default=0)
    predict.add_argument("--max-customers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        model_kwargs = {
            "hidden_dim": args.hidden_dim,
            "transformer_layers": args.transformer_layers,
            "attention_heads": args.attention_heads,
            "feedforward_dim": args.feedforward_dim,
            "transformer_dropout": args.transformer_dropout,
            "feature_dropout": args.feature_dropout,
            "output_dropout": args.output_dropout,
            "feature_hidden_layers": args.feature_hidden_layers,
            "positional_encoding": args.positional_encoding,
        }
        summary = train_transformer_cv(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            use_features=args.use_features,
            model_kwargs=model_kwargs,
            folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=args.device,
            num_workers=args.num_workers,
            feature_complement=not args.no_feature_complement,
            fixed_sequence_length=args.fixed_sequence_length,
            max_train_customers=args.max_train_customers,
        )
        print(summary)
    elif args.command == "predict":
        out = predict_transformer_cv(
            data_dir=args.data_dir,
            checkpoint_dir=args.checkpoint_dir,
            output_file=args.output_file,
            batch_size=args.batch_size,
            device=args.device,
            num_workers=args.num_workers,
            max_customers=args.max_customers,
        )
        print(out.head())
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
