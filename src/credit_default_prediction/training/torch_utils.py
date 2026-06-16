from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from credit_default_prediction.metrics import score_binary_predictions


def set_seed(seed: int) -> None:
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def resolve_device(device: str) -> torch.device:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested with --device cuda, but no CUDA device is available.")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device}")


def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list, np.ndarray]:
    model.eval()
    ids: list = []
    preds = []
    with torch.no_grad():
        for batch in loader:
            ids.extend(batch["customer_id"])
            batch = move_batch(batch, device)
            pred = model(batch).detach().cpu().numpy().reshape(-1)
            preds.append(pred)
    return ids, np.concatenate(preds) if preds else np.asarray([], dtype=np.float32)


def train_one_fold(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    metric: str,
    scheduler_name: str = "none",
    weight_decay: float = 0.0,
    clip_grad_norm: float | None = 1.0,
) -> tuple[np.ndarray, float, dict[str, torch.Tensor]]:
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    elif scheduler_name == "paper_step":
        scheduler = None
    elif scheduler_name == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    best_score = -np.inf
    best_state = None
    best_valid_preds = None
    valid_targets = np.concatenate([batch["target"].numpy().reshape(-1) for batch in valid_loader])

    for epoch in range(epochs):
        if scheduler_name == "paper_step":
            lr_scale = 0.01 if epoch > 8 else 0.1 if epoch > 4 else 1.0
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate * lr_scale
        model.train()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch)
            loss = criterion(pred, batch["target"])
            loss.backward()
            if clip_grad_norm is not None and clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        _, valid_preds = predict_loader(model, valid_loader, device)
        score = score_binary_predictions(valid_targets, valid_preds, metric=metric)
        if score > best_score:
            best_score = score
            best_valid_preds = valid_preds.copy()
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None or best_valid_preds is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    model.load_state_dict(best_state)
    return best_valid_preds, float(best_score), best_state
