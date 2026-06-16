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
from credit_default_prediction.models import TaiwanTransformer
from credit_default_prediction.training.torch_utils import (
    predict_loader,
    resolve_device,
    set_seed,
    train_one_fold,
)


def train_taiwan_transformer_cv(
    data_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    metric: str,
    use_features: bool,
    model_kwargs: dict[str, Any],
    folds: int = 5,
    epochs: int = 12,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    seed: int = 42,
    device: str = "cuda",
    num_workers: int = 0,
    scheduler_name: str = "none",
    include_year_month: bool = False,
    max_train_customers: int | None = None,
    feature_complement: bool = False,
    fixed_sequence_length: int | None = None,
    reload_best_for_oof: bool = False,
    optimizer_weight_decay: float = 0.0,
    clip_grad_norm: float | None = 1.0,
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
        include_year_month=include_year_month,
        max_customers=max_train_customers,
    )
    if arrays.labels is None:
        raise ValueError("Training labels are required.")

    labels = arrays.labels.astype(np.int64)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(labels), dtype=np.float32)
    fold_scores = []

    resolved_model_kwargs = {
        **model_kwargs,
        "series_dim": len(arrays.series_columns),
        "feature_dim": len(arrays.feature_columns) if use_features else 0,
    }
    collate_fn = partial(collate_sequences, fixed_length=fixed_sequence_length)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        train_ds = SequenceDataset(arrays, train_idx)
        valid_ds = SequenceDataset(arrays, valid_idx)
        drop_last_train = len(train_ds) > batch_size
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last_train,
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
        model = TaiwanTransformer(**resolved_model_kwargs).to(device_obj)
        valid_preds, score, best_state = train_one_fold(
            model,
            train_loader,
            valid_loader,
            device_obj,
            epochs=epochs,
            learning_rate=learning_rate,
            metric=metric,
            scheduler_name=scheduler_name,
            weight_decay=optimizer_weight_decay,
            clip_grad_norm=clip_grad_norm,
        )
        model_to_save = model
        if reload_best_for_oof:
            model_to_save = TaiwanTransformer(**resolved_model_kwargs).to(device_obj)
            model_to_save.load_state_dict(best_state)
            _, valid_preds = predict_loader(model_to_save, valid_loader, device_obj)
            score = score_binary_predictions(labels[valid_idx], valid_preds, metric=metric)
        oof[valid_idx] = valid_preds
        fold_scores.append(score)
        torch.save(
            {
                "model_state": model_to_save.state_dict(),
                "model_kwargs": resolved_model_kwargs,
                "dataset_name": dataset_name,
                "metric": metric,
                "series_columns": arrays.series_columns,
                "feature_columns": arrays.feature_columns,
                "use_features": use_features,
                "feature_complement": feature_complement,
                "fixed_sequence_length": fixed_sequence_length,
            },
            output_dir / f"fold_{fold}.pt",
        )

    overall = score_binary_predictions(arrays.labels, oof, metric=metric)
    pd.DataFrame({"customer_ID": arrays.ids, "target": arrays.labels, "prediction": oof}).to_csv(
        output_dir / "oof.csv", index=False
    )
    summary = {
        "dataset": dataset_name,
        "metric": metric,
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
            "reload_best_for_oof": reload_best_for_oof,
            "optimizer_weight_decay": optimizer_weight_decay,
            "clip_grad_norm": clip_grad_norm,
        },
        "model_kwargs": resolved_model_kwargs,
    }
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def predict_taiwan_transformer_cv(
    data_dir: str | Path,
    checkpoint_dir: str | Path,
    output_file: str | Path,
    split: str = "test",
    batch_size: int = 512,
    device: str = "cuda",
    num_workers: int = 0,
    include_year_month: bool = False,
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
        feature_complement=bool(first.get("feature_complement", False)),
        include_year_month=include_year_month,
        max_customers=max_customers,
    )
    dataset = SequenceDataset(arrays)
    collate_fn = partial(collate_sequences, fixed_length=first.get("fixed_sequence_length"))
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
    for path in checkpoints:
        payload = torch.load(path, map_location=device_obj)
        model = TaiwanTransformer(**payload["model_kwargs"]).to(device_obj)
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
