from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from credit_default_prediction.data.amex import AMEX_CATEGORICAL_COLUMNS, denoise_amex_frame, load_raw_amex
from credit_default_prediction.data.sequence import ID_COLUMN, TARGET_COLUMN, read_table, write_table


DATE_COLUMN = "S_2"
BlockKind = Literal["categorical", "numeric", "difference"]
RankMode = Literal["none", "customer", "year_month"]


@dataclass(frozen=True)
class FeatureBlockSpec:
    name: str
    kind: BlockKind
    recent_periods: int | None = None
    rank_mode: RankMode = "none"


MANUAL_FEATURE_SPECS = [
    FeatureBlockSpec("cat", "categorical"),
    FeatureBlockSpec("num", "numeric"),
    FeatureBlockSpec("diff", "difference"),
    FeatureBlockSpec("rank_num", "numeric", rank_mode="customer"),
    FeatureBlockSpec("last3_cat", "categorical", recent_periods=3),
    FeatureBlockSpec("last3_num", "numeric", recent_periods=3),
    FeatureBlockSpec("last3_diff", "difference", recent_periods=3),
    FeatureBlockSpec("last6_num", "numeric", recent_periods=6),
    FeatureBlockSpec("ym_rank_num", "numeric", rank_mode="year_month"),
]
MANUAL_FEATURE_BLOCKS = [spec.name for spec in MANUAL_FEATURE_SPECS]


def one_hot_columns(df: pd.DataFrame, columns: list[str], drop: bool = False) -> pd.DataFrame:
    out = df.copy()
    present = [col for col in columns if col in out.columns]
    for col in present:
        dummies = pd.get_dummies(pd.Series(out[col]), prefix=f"oneHot_{col}")
        out = pd.concat([out, dummies], axis=1)
    if drop and present:
        out = out.drop(columns=present)
    return out


def _base_numeric_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in {ID_COLUMN, DATE_COLUMN, *AMEX_CATEGORICAL_COLUMNS}]


def _fill_sequence_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [col for col in out.columns if col.startswith(("S_", "P_")) and col != DATE_COLUMN]:
        out[col] = out[col].fillna(0)
    return out


def _select_recent_periods(df: pd.DataFrame, periods: int | None) -> pd.DataFrame:
    if periods is None:
        return df
    ranked = df.copy()
    ranked["_recent_rank"] = ranked.groupby(ID_COLUMN, sort=False)[DATE_COLUMN].rank(ascending=False)
    return ranked[ranked["_recent_rank"] <= periods].drop(columns=["_recent_rank"]).reset_index(drop=True)


def _rank_numeric_frame(
    df: pd.DataFrame,
    numeric_columns: list[str],
    rank_mode: RankMode,
) -> tuple[pd.DataFrame, list[str]]:
    if rank_mode == "none":
        return df, numeric_columns
    if rank_mode == "customer":
        ranked = df.groupby(ID_COLUMN, sort=False)[numeric_columns].rank(pct=True).add_prefix("rank_")
        ranked.insert(0, ID_COLUMN, df[ID_COLUMN].to_numpy())
        return ranked, [f"rank_{col}" for col in numeric_columns]
    if rank_mode == "year_month":
        with_month = df.copy()
        with_month["_year_month"] = with_month[DATE_COLUMN].astype(str).str[:7]
        ranked = with_month.groupby("_year_month", sort=False)[numeric_columns].rank(pct=True).add_prefix("ym_rank_")
        ranked.insert(0, ID_COLUMN, with_month[ID_COLUMN].to_numpy())
        return ranked, [f"ym_rank_{col}" for col in numeric_columns]
    raise ValueError(f"Unsupported rank mode: {rank_mode}")


def _frame_for_block(full: pd.DataFrame, spec: FeatureBlockSpec) -> tuple[pd.DataFrame, list[str]]:
    frame = _fill_sequence_missing_values(full)
    frame = _select_recent_periods(frame, spec.recent_periods)
    numeric_columns = _base_numeric_columns(frame)
    frame, numeric_columns = _rank_numeric_frame(frame, numeric_columns, spec.rank_mode)
    return frame, numeric_columns


def _customer_chunks(df: pd.DataFrame, chunk_customers: int | None) -> list[pd.DataFrame]:
    if chunk_customers is None:
        return [df]
    ids = df[ID_COLUMN].drop_duplicates().to_numpy()
    chunks = []
    for start in range(0, len(ids), chunk_customers):
        keep = set(ids[start : start + chunk_customers])
        chunks.append(df[df[ID_COLUMN].isin(keep)])
    return chunks


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = ["_".join(col) if isinstance(col, tuple) else str(col) for col in out.columns]
    return out


def _categorical_aggregates(df: pd.DataFrame, categorical_columns: list[str], lastk: int | None) -> pd.DataFrame:
    one_hot_features = [col for col in df.columns if "oneHot" in col]
    one_hot_aggs = ["mean", "std", "sum", "last"] if lastk is None else ["mean", "std", "sum"]
    cat_aggs = ["last", "nunique"] if lastk is None else ["nunique"]
    frames = []
    if one_hot_features:
        frames.append(_flatten_columns(df.groupby(ID_COLUMN, sort=False)[one_hot_features].agg(one_hot_aggs)))
    present_cat = [col for col in categorical_columns if col in df.columns]
    if present_cat:
        frames.append(_flatten_columns(df.groupby(ID_COLUMN, sort=False)[present_cat].agg(cat_aggs)))
    frames.append(_flatten_columns(df.groupby(ID_COLUMN, sort=False)[[DATE_COLUMN]].agg(["count"])))
    return pd.concat(frames, axis=1).reset_index()


def _numeric_aggregates(df: pd.DataFrame, numeric_columns: list[str], lastk: int | None) -> pd.DataFrame:
    if numeric_columns and numeric_columns[0].startswith("rank_"):
        aggs = ["last"]
    else:
        aggs = ["mean", "std", "min", "max", "sum", "last"] if lastk is None else [
            "mean",
            "std",
            "min",
            "max",
            "sum",
        ]
    out = _flatten_columns(df.groupby(ID_COLUMN, sort=False)[numeric_columns].agg(aggs)).reset_index()
    if numeric_columns and not numeric_columns[0].startswith("rank_"):
        for col in out.columns:
            if col != ID_COLUMN:
                out[col] = out[col] // 0.01
    return out


def _diff_aggregates(df: pd.DataFrame, numeric_columns: list[str], lastk: int | None) -> pd.DataFrame:
    diff_columns = [f"diff_{col}" for col in numeric_columns]
    out = df.groupby(ID_COLUMN, sort=False)[numeric_columns].diff().add_prefix("diff_")
    out.insert(0, ID_COLUMN, df[ID_COLUMN].to_numpy())
    aggs = ["mean", "std", "min", "max", "sum", "last"] if lastk is None else [
        "mean",
        "std",
        "min",
        "max",
        "sum",
    ]
    out = _flatten_columns(out.groupby(ID_COLUMN, sort=False)[diff_columns].agg(aggs)).reset_index()
    for col in out.columns:
        if col != ID_COLUMN:
            out[col] = out[col] // 0.01
    return out


def _build_feature_block(
    full: pd.DataFrame,
    spec: FeatureBlockSpec,
    chunk_customers: int | None,
) -> pd.DataFrame:
    frame, numeric_columns = _frame_for_block(full, spec)
    chunks = _customer_chunks(frame, chunk_customers)
    if spec.kind == "categorical":
        encoded_chunks = [one_hot_columns(chunk, AMEX_CATEGORICAL_COLUMNS, drop=False) for chunk in chunks]
        parts = [_categorical_aggregates(chunk, AMEX_CATEGORICAL_COLUMNS, spec.recent_periods) for chunk in encoded_chunks]
    elif spec.kind == "numeric":
        parts = [_numeric_aggregates(chunk, numeric_columns, spec.recent_periods) for chunk in chunks]
    elif spec.kind == "difference":
        parts = [_diff_aggregates(chunk, numeric_columns, spec.recent_periods) for chunk in chunks]
    else:
        raise ValueError(f"Unsupported feature block kind: {spec.kind}")
    return pd.concat(parts, axis=0).reset_index(drop=True)


def _load_denoised_train_test(raw_dir: str | Path, nrows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test, source = load_raw_amex(raw_dir, nrows=nrows)
    if source == "csv":
        train = denoise_amex_frame(train)
        test = denoise_amex_frame(test)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def prepare_manual_feature_blocks(
    raw_dir: str | Path,
    output_dir: str | Path,
    nrows: int | None = None,
    max_customers: int | None = None,
    chunk_customers: int | None = None,
) -> dict:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, test = _load_denoised_train_test(raw_dir, nrows=nrows)
    if max_customers is not None:
        train_ids = train[ID_COLUMN].drop_duplicates().iloc[:max_customers]
        test_ids = test[ID_COLUMN].drop_duplicates().iloc[:max_customers]
        train = train[train[ID_COLUMN].isin(train_ids)]
        test = test[test[ID_COLUMN].isin(test_ids)]
    full = pd.concat([train, test], axis=0, ignore_index=True)

    created: list[str] = []
    for spec in MANUAL_FEATURE_SPECS:
        feature_block = _build_feature_block(full, spec, chunk_customers)
        file_name = f"{spec.name}_feature.feather"
        write_table(feature_block, output_dir / file_name)
        created.append(file_name)

    metadata = {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "created": created,
    }
    (output_dir / "manual_feature_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def pad_customer_targets(values: pd.Series, length: int = 13) -> list[float]:
    out = np.full(length, np.nan, dtype=np.float32)
    arr = values.to_numpy(dtype=np.float32)[-length:]
    out[-len(arr) :] = arr
    return out.tolist()


def _prediction_column(df: pd.DataFrame) -> str:
    if "prediction" in df.columns:
        return "prediction"
    if TARGET_COLUMN in df.columns:
        return TARGET_COLUMN
    raise ValueError("Prediction file must contain either 'prediction' or 'target'.")


def build_target_history_features(
    row_oof_file: str | Path,
    row_submission_file: str | Path,
    sequence_length: int = 13,
) -> pd.DataFrame:
    oof = read_table(row_oof_file)
    sub = read_table(row_submission_file)
    oof_col = _prediction_column(oof)
    sub_col = _prediction_column(sub)
    columns = [f"target{i}" for i in range(1, sequence_length + 1)]

    def history_frame(df: pd.DataFrame, prediction_col: str) -> pd.DataFrame:
        targets = df.groupby(ID_COLUMN, sort=False)[prediction_col].agg(
            lambda values: pad_customer_targets(values, sequence_length)
        )
        frame = pd.DataFrame(data=targets.tolist(), columns=columns)
        frame.insert(0, ID_COLUMN, targets.index.to_numpy())
        return frame

    return pd.concat(
        [history_frame(oof, oof_col), history_frame(sub, sub_col)],
        axis=0,
        ignore_index=True,
    )


def _load_feature_block(feature_dir: Path, name: str, is_first: bool) -> pd.DataFrame:
    block = read_table(feature_dir / f"{name}_feature.feather")
    if not is_first:
        block = block.drop(columns=[ID_COLUMN])
    if name.startswith("last"):
        prefix = "_".join(name.split("_")[:-1]) + "_"
        block = block.add_prefix(prefix)
        if is_first:
            block = block.rename(columns={f"{prefix}{ID_COLUMN}": ID_COLUMN})
    return block


def greedy_find_bin(
    distinct_values: np.ndarray,
    counts: np.ndarray,
    max_bin: int,
    total_count: int,
    min_data_in_bin: int = 3,
) -> list[float]:
    value_count = len(distinct_values)
    if value_count == 0:
        return [float("Inf")]
    if value_count <= max_bin:
        bounds = []
        bin_size = 0
        for left, right, count in zip(distinct_values[:-1], distinct_values[1:], counts[:-1]):
            bin_size += int(count)
            if bin_size >= min_data_in_bin:
                bounds.append(float((left + right) / 2.0))
                bin_size = 0
        bounds.append(float("Inf"))
        return bounds

    bin_budget = max_bin
    if min_data_in_bin > 0:
        bin_budget = min(max_bin, max(total_count // min_data_in_bin, 1))

    target_size = total_count / bin_budget
    large_values = counts >= target_size
    remaining_bins = bin_budget - int(large_values.sum())
    remaining_count = total_count - int(counts[large_values].sum())
    target_size = remaining_count / max(remaining_bins, 1)

    right_edges: list[float] = []
    next_left_edges: list[float] = []
    accumulated = 0
    for idx, count in enumerate(counts[:-1]):
        if not large_values[idx]:
            remaining_count -= int(count)
        accumulated += int(count)
        next_is_large = bool(large_values[idx + 1])
        closes_bin = (
            bool(large_values[idx])
            or accumulated >= target_size
            or (next_is_large and accumulated >= max(1.0, target_size * 0.5))
        )
        if closes_bin:
            right_edges.append(float(distinct_values[idx]))
            next_left_edges.append(float(distinct_values[idx + 1]))
            if len(right_edges) >= bin_budget - 1:
                break
            accumulated = 0
            if not large_values[idx]:
                remaining_bins -= 1
                target_size = remaining_count / max(remaining_bins, 1)

    bounds = [
        float((right_edges[i] + next_left_edges[i]) / 2.0)
        for i in range(max(len(right_edges) - 1, 0))
    ]
    bounds.append(float("Inf"))
    return bounds


def _binned_feature_block(df: pd.DataFrame, max_bin: int = 255) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == ID_COLUMN:
            continue
        values = pd.to_numeric(out[col], errors="coerce").astype("float64")
        missing = values.isna().to_numpy()
        counts = values.value_counts(dropna=True).sort_index()
        if counts.empty:
            out[col] = 0.0
            continue
        bins = greedy_find_bin(counts.index.to_numpy(), counts.to_numpy(), max_bin, int(counts.sum()))
        binned = np.digitize(values.to_numpy(dtype=np.float64), [-np.inf, *bins]).astype(np.float32)
        if missing.any():
            binned[missing] = 0.0
        binned[binned == len(bins) + 1] = 0.0
        max_value = binned.max()
        out[col] = binned / max_value if max_value > 0 else binned
    return out


def build_model_feature_tables(
    feature_dir: str | Path,
    output_dir: str | Path,
    train_labels_file: str | Path,
    row_oof_file: str | Path,
    row_submission_file: str | Path,
    sequence_length: int = 13,
) -> dict:
    feature_dir = Path(feature_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_features = build_target_history_features(
        row_oof_file=row_oof_file,
        row_submission_file=row_submission_file,
        sequence_length=sequence_length,
    )

    blocks = [
        _load_feature_block(feature_dir, name, is_first=i == 0)
        for i, name in enumerate(MANUAL_FEATURE_BLOCKS)
    ]
    all_features = pd.concat(blocks, axis=1)
    all_features = all_features.merge(target_features, on=ID_COLUMN, how="left", sort=False, validate="one_to_one")
    write_table(all_features, output_dir / "all_feature.feather")
    write_table(all_features.head(1), output_dir / "all_feature_sample.feather")

    nn_blocks = [
        _binned_feature_block(_load_feature_block(feature_dir, name, is_first=i == 0))
        for i, name in enumerate(MANUAL_FEATURE_BLOCKS)
    ]
    target_features_nn = target_features.copy()
    target_columns = [col for col in target_features_nn.columns if col != ID_COLUMN]
    target_features_nn[target_columns] = target_features_nn[target_columns].fillna(0.0)
    nn_features = pd.concat(nn_blocks, axis=1)
    nn_features = nn_features.merge(
        target_features_nn, on=ID_COLUMN, how="left", sort=False, validate="one_to_one"
    )
    write_table(nn_features, output_dir / "nn_all_feature.feather")

    labels = read_table(train_labels_file)
    train_count = len(labels)
    write_table(nn_features.iloc[:train_count].reset_index(drop=True), output_dir / "df_nn_feature_train.feather")
    write_table(nn_features.iloc[train_count:].reset_index(drop=True), output_dir / "df_nn_feature_test.feather")

    metadata = {
        "feature_dir": str(feature_dir),
        "output_dir": str(output_dir),
        "rows": int(len(all_features)),
        "columns": int(len(all_features.columns)),
        "train_customers": int(train_count),
        "test_customers": int(len(all_features) - train_count),
    }
    (output_dir / "model_feature_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata
