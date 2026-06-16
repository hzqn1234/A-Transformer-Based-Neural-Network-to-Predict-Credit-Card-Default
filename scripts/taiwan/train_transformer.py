from __future__ import annotations

import argparse

from credit_default_prediction.training.taiwan_transformer import train_taiwan_transformer_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Taiwan transformer model.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.0009)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-features", action="store_true")
    parser.add_argument("--max-train-customers", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--feedforward-dim", type=int, default=4)
    parser.add_argument("--transformer-dropout", type=float, default=0.005)
    parser.add_argument("--feature-dropout", type=float, default=0.005)
    parser.add_argument("--output-dropout", type=float, default=0.025)
    parser.add_argument("--feature-hidden-layers", type=int, default=2)
    parser.add_argument("--positional-encoding", default="none", choices=["none", "sinusoidal"])
    parser.add_argument("--use-padding-mask", action="store_true")
    parser.add_argument("--gru-pooling", default="hidden", choices=["hidden", "last_output"])
    parser.add_argument("--feature-complement", action="store_true")
    parser.add_argument("--fixed-sequence-length", type=int, default=None)
    parser.add_argument("--reload-best-for-oof", action="store_true")
    parser.add_argument("--optimizer-weight-decay", type=float, default=0.0)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        "use_padding_mask": args.use_padding_mask,
        "gru_pooling": args.gru_pooling,
    }
    summary = train_taiwan_transformer_cv(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_name="taiwan",
        metric="auc",
        use_features=args.use_features,
        model_kwargs=model_kwargs,
        folds=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
        scheduler_name="cosine",
        max_train_customers=args.max_train_customers,
        feature_complement=args.feature_complement,
        fixed_sequence_length=args.fixed_sequence_length,
        reload_best_for_oof=args.reload_best_for_oof,
        optimizer_weight_decay=args.optimizer_weight_decay,
        clip_grad_norm=args.clip_grad_norm,
    )
    print(summary)


if __name__ == "__main__":
    main()
