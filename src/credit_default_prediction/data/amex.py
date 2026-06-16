from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from credit_default_prediction.data.sequence import sequence_index, write_table


ID_COLUMN = "customer_ID"
DATE_COLUMN = "S_2"
TARGET_COLUMN = "target"

AMEX_CATEGORICAL_COLUMNS = [
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
]

D63_MAP = {"CR": 0, "XZ": 1, "XM": 2, "CO": 3, "CL": 4, "XL": 5}
D64_MAP = {"O": 0, "-1": 1, "R": 2, "U": 3}


def load_raw_amex(raw_dir: str | Path, nrows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    raw_dir = Path(raw_dir)
    feather_train = raw_dir / "train.feather"
    feather_test = raw_dir / "test.feather"
    if nrows is None and feather_train.exists() and feather_test.exists():
        train = pd.read_feather(feather_train)
        test = pd.read_feather(feather_test)
        return train, test, "denoised_feather"

    train = pd.read_csv(raw_dir / "train_data.csv", nrows=nrows)
    test = pd.read_csv(raw_dir / "test_data.csv", nrows=nrows)
    return train, test, "csv"


def load_encoded_amex(raw_dir: str | Path, nrows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    raw_dir = Path(raw_dir)
    encoded_path = raw_dir / "nn_series.feather"
    count_path = raw_dir / "train_df_count_df.feather"
    if nrows is not None or not encoded_path.exists() or not count_path.exists():
        return None

    full = pd.read_feather(encoded_path)
    train_count = int(pd.read_feather(count_path).iloc[0, 0])
    train_series = full.iloc[:train_count].reset_index(drop=True)
    test_series = full.iloc[train_count:].reset_index(drop=True)
    for frame in (train_series, test_series):
        if DATE_COLUMN in frame:
            frame.drop(columns=[DATE_COLUMN], inplace=True)
    return train_series, test_series


def denoise_amex_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "D_63" in out:
        out["D_63"] = out["D_63"].map(D63_MAP).astype("float32")
    if "D_64" in out:
        out["D_64"] = out["D_64"].map(D64_MAP).fillna(-1).astype("float32")

    skip = {ID_COLUMN, DATE_COLUMN, "D_63", "D_64"}
    for col in out.columns:
        if col not in skip and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = np.floor(out[col].astype("float32") * 100.0)
    return out


def one_hot_categories(df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    present = [col for col in categorical_columns if col in df.columns]
    if not present:
        return df.copy()
    prefixes = {col: f"oneHot_{col}" for col in present}
    return pd.get_dummies(df, columns=present, prefix=prefixes)


def scale_and_fill(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if DATE_COLUMN in out:
        out = out.drop(columns=[DATE_COLUMN])
    for col in out.columns:
        if col == ID_COLUMN:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype("float32") / 100.0
            out[col] = out[col].fillna(0.0)
        else:
            out[col] = out[col].fillna("")
    return out


def limit_customers(df: pd.DataFrame, max_customers: int | None) -> pd.DataFrame:
    if max_customers is None:
        return df
    keep = df[ID_COLUMN].drop_duplicates().iloc[:max_customers]
    return df[df[ID_COLUMN].isin(keep)].reset_index(drop=True)


def prepare_amex_sequences(
    raw_dir: str | Path,
    output_dir: str | Path,
    nrows: int | None = None,
    max_train_customers: int | None = None,
    max_test_customers: int | None = None,
) -> dict:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encoded = load_encoded_amex(raw_dir, nrows=nrows)
    if encoded is not None:
        train_series, test_series = encoded
        raw_source = "encoded_feather"
        train_series = limit_customers(train_series, max_train_customers)
        test_series = limit_customers(test_series, max_test_customers)
    else:
        train, test, raw_source = load_raw_amex(raw_dir, nrows=nrows)
        if raw_source == "csv":
            train = train.sort_values([ID_COLUMN, DATE_COLUMN])
            test = test.sort_values([ID_COLUMN, DATE_COLUMN])
        train = limit_customers(train, max_train_customers)
        test = limit_customers(test, max_test_customers)

        train_count = len(train)
        if raw_source == "denoised_feather":
            full = pd.concat([train, test], axis=0, ignore_index=True)
        else:
            full = pd.concat([denoise_amex_frame(train), denoise_amex_frame(test)], axis=0, ignore_index=True)
        full = one_hot_categories(full, AMEX_CATEGORICAL_COLUMNS)
        full = scale_and_fill(full)

        train_series = full.iloc[:train_count].reset_index(drop=True)
        test_series = full.iloc[train_count:].reset_index(drop=True)
    train_idx = sequence_index(train_series)
    test_idx = sequence_index(test_series)

    write_table(train_series, output_dir / "df_nn_series_train.feather")
    write_table(test_series, output_dir / "df_nn_series_test.feather")
    write_table(train_idx, output_dir / "df_nn_series_idx_train.feather")
    write_table(test_idx, output_dir / "df_nn_series_idx_test.feather")

    labels = pd.read_csv(raw_dir / "train_labels.csv")
    if max_train_customers is not None:
        labels = labels[labels[ID_COLUMN].isin(train_series[ID_COLUMN].drop_duplicates())]
    labels.to_csv(output_dir / "train_labels.csv", index=False)

    sample_path = raw_dir / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        if max_test_customers is not None:
            sample = sample[sample[ID_COLUMN].isin(test_series[ID_COLUMN].drop_duplicates())]
        sample.to_csv(output_dir / "sample_submission.csv", index=False)

    metadata = {
        "train_rows": int(len(train_series)),
        "test_rows": int(len(test_series)),
        "train_customers": int(len(train_idx)),
        "test_customers": int(len(test_idx)),
        "raw_source": raw_source,
        "series_columns": [c for c in train_series.columns if c != ID_COLUMN],
        "categorical_columns": AMEX_CATEGORICAL_COLUMNS,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
