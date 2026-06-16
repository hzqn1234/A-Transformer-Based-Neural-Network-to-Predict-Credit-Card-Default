from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from credit_default_prediction.data.sequence import (
    SequenceDataset,
    collate_sequences,
    load_sequence_arrays,
)
from credit_default_prediction.metrics import score_binary_predictions
from credit_default_prediction.models import AmexTransformer
from credit_default_prediction.training.torch_utils import (
    predict_loader,
    resolve_device,
    set_seed,
    train_one_fold,
)


def train_transformer_cv(
    data_dir: str | Path,
    output_dir: str | Path,
    use_features: bool,
    model_kwargs: dict[str, Any],
    folds: int = 5,
    epochs: int = 12,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    seed: int = 42,
    device: str = "cuda",
    num_workers: int = 0,
    scheduler_name: str = "paper_step",
    feature_complement: bool = True,
    fixed_sequence_length: int = 13,
    max_train_customers: int | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = resolve_device(device)

    arrays = load_sequence_arrays(
        data_dir,
        split="train",
        use_features=use_features,
        feature_complement=feature_complement,
        max_customers=max_train_customers,
    )
    if arrays.labels is None:
        raise ValueError("Training labels are required.")

    labels = arrays.labels.astype(np.int64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(labels), dtype=np.float32)
    fold_scores: list[float] = []
    resolved_model_kwargs = {
        **model_kwargs,
        "series_dim": len(arrays.series_columns),
        "feature_dim": len(arrays.feature_columns) if use_features else 0,
    }
    collate_fn = partial(collate_sequences, fixed_length=fixed_sequence_length)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        train_ds = SequenceDataset(arrays, train_idx)
        valid_ds = SequenceDataset(arrays, valid_idx)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(train_ds) > batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        model = AmexTransformer(**resolved_model_kwargs).to(device_obj)
        _, score, best_state = train_one_fold(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device_obj,
            epochs=epochs,
            learning_rate=learning_rate,
            metric="amex",
            scheduler_name=scheduler_name,
            weight_decay=0.0,
            clip_grad_norm=0.0,
        )
        model.load_state_dict(best_state)
        _, valid_preds = predict_loader(model, valid_loader, device_obj)
        score = score_binary_predictions(labels[valid_idx], valid_preds, metric="amex")
        oof[valid_idx] = valid_preds
        fold_scores.append(score)
        torch.save(
            {
                "model_type": "amex_transformer",
                "model_state": model.state_dict(),
                "model_kwargs": resolved_model_kwargs,
                "dataset_name": "amex",
                "metric": "amex",
                "series_columns": arrays.series_columns,
                "feature_columns": arrays.feature_columns,
                "use_features": use_features,
                "feature_complement": feature_complement,
                "fixed_sequence_length": fixed_sequence_length,
            },
            output_dir / f"fold_{fold}.pt",
        )

    overall = score_binary_predictions(arrays.labels, oof, metric="amex")
    pd.DataFrame({"customer_ID": arrays.ids, "target": arrays.labels, "prediction": oof}).to_csv(
        output_dir / "oof.csv", index=False
    )
    summary = {
        "dataset": "amex",
        "metric": "amex",
        "fold_scores": fold_scores,
        "mean_fold_score": float(np.mean(fold_scores)),
        "global_oof_score": float(overall),
        "training": {
            "folds": folds,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "scheduler": scheduler_name,
            "feature_complement": feature_complement,
            "fixed_sequence_length": fixed_sequence_length,
        },
        "model_kwargs": resolved_model_kwargs,
    }
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def predict_transformer_cv(
    data_dir: str | Path,
    checkpoint_dir: str | Path,
    output_file: str | Path,
    split: str = "test",
    batch_size: int = 512,
    device: str = "cuda",
    num_workers: int = 0,
    max_customers: int | None = None,
) -> pd.DataFrame:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("fold_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No fold checkpoints found in {checkpoint_dir}")

    first = torch.load(checkpoints[0], map_location="cpu")
    arrays = load_sequence_arrays(
        data_dir,
        split=split,
        use_features=bool(first["use_features"]),
        feature_complement=bool(first.get("feature_complement", True)),
        max_customers=max_customers,
    )
    dataset = SequenceDataset(arrays)
    collate_fn = partial(collate_sequences, fixed_length=first.get("fixed_sequence_length", 13))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    device_obj = resolve_device(device)
    all_preds = []
    ids = None
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location=device_obj)
        model = AmexTransformer(**payload["model_kwargs"]).to(device_obj)
        model.load_state_dict(payload["model_state"])
        ids, preds = predict_loader(model, loader, device_obj)
        all_preds.append(preds)

    prediction = np.mean(np.vstack(all_preds), axis=0)
    out = pd.DataFrame({"customer_ID": ids, "prediction": prediction})
    sample_submission = Path(data_dir) / "sample_submission.csv"
    if split == "test" and sample_submission.exists():
        sample = pd.read_csv(sample_submission)[["customer_ID"]]
        out = sample.merge(out, on="customer_ID", how="left")
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_file, index=False)
    return out
