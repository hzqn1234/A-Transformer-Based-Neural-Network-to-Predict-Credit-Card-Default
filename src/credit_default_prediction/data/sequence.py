from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ID_COLUMN = "customer_ID"
TARGET_COLUMN = "target"


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".feather":
        return pd.read_feather(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".feather":
        df.reset_index(drop=True).to_feather(path)
    elif path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def sequence_index(series_df: pd.DataFrame, id_col: str = ID_COLUMN) -> pd.DataFrame:
    indexed = series_df[[id_col]].copy()
    indexed["_row_index"] = np.arange(len(indexed), dtype=np.int64)
    out = indexed.groupby(id_col, sort=False)["_row_index"].agg(["min", "max"]).reset_index(drop=True)
    out["feature_idx"] = np.arange(len(out), dtype=np.int64)
    return out


def numeric_feature_columns(
    df: pd.DataFrame,
    id_col: str = ID_COLUMN,
    drop_columns: Iterable[str] = (),
) -> list[str]:
    dropped = {id_col, *drop_columns}
    cols = []
    for col in df.columns:
        if col in dropped:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


@dataclass
class SequenceArrays:
    ids: np.ndarray
    series: np.ndarray
    bounds: np.ndarray
    features: np.ndarray | None
    labels: np.ndarray | None
    series_columns: list[str]
    feature_columns: list[str]


def load_sequence_arrays(
    data_dir: str | Path,
    split: str,
    use_features: bool = False,
    feature_complement: bool = False,
    include_year_month: bool = False,
    max_customers: int | None = None,
) -> SequenceArrays:
    data_dir = Path(data_dir)
    series_df = read_table(data_dir / f"df_nn_series_{split}.feather")
    idx_df = read_table(data_dir / f"df_nn_series_idx_{split}.feather")
    label_path = data_dir / f"{split}_labels.csv"
    labels_df = read_table(label_path) if label_path.exists() else None

    drop_cols = [] if include_year_month else ["year_month"]
    series_columns = numeric_feature_columns(series_df, drop_columns=drop_cols)
    series_values = series_df[series_columns].to_numpy(dtype=np.float32, copy=True)

    if max_customers is not None:
        idx_df = idx_df.iloc[:max_customers].reset_index(drop=True)

    bounds = idx_df[["min", "max"]].to_numpy(dtype=np.int64, copy=True)
    ids = []
    for start, _ in bounds:
        ids.append(series_df.iloc[int(start)][ID_COLUMN])
    ids = np.asarray(ids)

    features = None
    feature_columns: list[str] = []
    if use_features:
        feature_file = data_dir / f"df_nn_feature_{split}.feather"
        if feature_file.exists():
            feature_df = read_table(feature_file)
            if max_customers is not None:
                feature_df = feature_df.iloc[:max_customers].reset_index(drop=True)
            feature_columns = numeric_feature_columns(feature_df)
            features = feature_df[feature_columns].to_numpy(dtype=np.float32, copy=True)
            if feature_complement:
                complement = features.copy()
                nonzero = complement != 0
                complement[nonzero] = 1.0 - complement[nonzero] + 0.001
                features = np.concatenate([features, complement], axis=1)
                feature_columns = feature_columns + [f"{col}_complement" for col in feature_columns]

    labels = None
    if labels_df is not None:
        if max_customers is not None:
            labels_df = labels_df.iloc[:max_customers].reset_index(drop=True)
        if len(labels_df) == len(bounds):
            labels = labels_df[TARGET_COLUMN].to_numpy(dtype=np.float32, copy=True)
        else:
            label_map = labels_df.set_index(ID_COLUMN)[TARGET_COLUMN]
            labels = label_map.reindex(ids).to_numpy(dtype=np.float32)

    return SequenceArrays(
        ids=ids,
        series=series_values,
        bounds=bounds,
        features=features,
        labels=labels,
        series_columns=series_columns,
        feature_columns=feature_columns,
    )


class SequenceDataset(Dataset):
    def __init__(self, arrays: SequenceArrays, sample_indices: np.ndarray | None = None):
        self.arrays = arrays
        if sample_indices is None:
            sample_indices = np.arange(len(arrays.bounds), dtype=np.int64)
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, item: int) -> dict:
        sample_idx = int(self.sample_indices[item])
        start, end = self.arrays.bounds[sample_idx]
        series = self.arrays.series[int(start) : int(end) + 1]
        out = {
            "series": torch.from_numpy(series),
            "customer_id": self.arrays.ids[sample_idx],
        }
        if self.arrays.features is not None:
            out["features"] = torch.from_numpy(self.arrays.features[sample_idx])
        if self.arrays.labels is not None:
            out["target"] = torch.tensor(self.arrays.labels[sample_idx], dtype=torch.float32)
        return out


def collate_sequences(batch: list[dict], fixed_length: int | None = None) -> dict:
    lengths = torch.tensor([item["series"].shape[0] for item in batch], dtype=torch.long)
    feature_dim = batch[0]["series"].shape[1]
    max_len = int(lengths.max().item())
    if fixed_length is not None:
        if fixed_length < max_len:
            raise ValueError(f"fixed_length={fixed_length} is shorter than a sequence of length {max_len}.")
        max_len = fixed_length
    series = torch.zeros(len(batch), max_len, feature_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, item in enumerate(batch):
        length = item["series"].shape[0]
        series[i, :length] = item["series"]
        mask[i, :length] = True

    out = {
        "series": series,
        "mask": mask,
        "lengths": lengths,
        "customer_id": [item["customer_id"] for item in batch],
    }
    if "features" in batch[0]:
        out["features"] = torch.stack([item["features"] for item in batch])
    if "target" in batch[0]:
        out["target"] = torch.stack([item["target"] for item in batch]).view(-1, 1)
    return out
