from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from credit_default_prediction.data.sequence import ID_COLUMN, TARGET_COLUMN, sequence_index, write_table


TAIWAN_FEATURE_COLUMNS = [f"X{i}" for i in range(1, 24)]
TAIWAN_SERIES_MONTHS = [
    ("X11", "X17", "X23", 4),
    ("X10", "X16", "X22", 5),
    ("X9", "X15", "X21", 6),
    ("X8", "X14", "X20", 7),
    ("X7", "X13", "X19", 8),
    ("X6", "X12", "X18", 9),
]


def fetch_taiwan_from_uci() -> tuple[pd.DataFrame, pd.Series]:
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=350)
    features = dataset.data.features.copy()
    target = dataset.data.targets.iloc[:, 0].copy()
    return features, target


def read_taiwan_csv(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    target_candidates = ["Y", "target", "default payment next month", "default.payment.next.month"]
    target_col = next((col for col in target_candidates if col in df.columns), None)
    if target_col is None:
        raise ValueError("Could not find Taiwan target column in CSV.")
    feature_cols = [col for col in TAIWAN_FEATURE_COLUMNS if col in df.columns]
    if len(feature_cols) != 23:
        raise ValueError("Expected columns X1 through X23 in Taiwan CSV.")
    return df[feature_cols].copy(), df[target_col].copy()


def add_customer_id(features: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = features.reset_index(drop=True).copy()
    x.insert(0, ID_COLUMN, np.arange(len(x), dtype=np.int64))
    y = pd.DataFrame({ID_COLUMN: x[ID_COLUMN].values, TARGET_COLUMN: target.reset_index(drop=True).values})
    return x, y


def split_ids(labels: pd.DataFrame, seed: int, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    train_ids = []
    test_ids = []
    for cid in labels[ID_COLUMN].values:
        if rng.random_sample() < train_ratio:
            train_ids.append(cid)
        else:
            test_ids.append(cid)
    return (
        pd.DataFrame({ID_COLUMN: train_ids}),
        pd.DataFrame({ID_COLUMN: test_ids}),
    )


def build_taiwan_series(features: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for repay_col, bill_col, paid_col, month in TAIWAN_SERIES_MONTHS:
        frames.append(
            features[[ID_COLUMN, repay_col, bill_col, paid_col]]
            .rename(columns={repay_col: "repay_status", bill_col: "bill", paid_col: "paid"})
            .assign(S2=month)
        )
    return pd.concat(frames, axis=0).sort_values([ID_COLUMN, "S2"]).reset_index(drop=True)


def greedy_bin_edges(values: np.ndarray, max_bin: int = 31, min_data_in_bin: int = 3) -> list[float]:
    value_counts = pd.Series(values).value_counts().sort_index()
    distinct_values = value_counts.index.to_numpy()
    counts = value_counts.to_numpy()
    num_distinct = len(distinct_values)
    total_count = int(counts.sum())
    if num_distinct <= 1:
        return [float("inf")]

    bin_upper_bound: list[float] = []
    if num_distinct <= max_bin:
        current_count = 0
        for i in range(num_distinct - 1):
            current_count += counts[i]
            if current_count >= min_data_in_bin:
                bin_upper_bound.append(float((distinct_values[i] + distinct_values[i + 1]) / 2.0))
                current_count = 0
        bin_upper_bound.append(float("inf"))
        return bin_upper_bound

    if min_data_in_bin > 0:
        max_bin = min(max_bin, total_count // min_data_in_bin)
        max_bin = max(max_bin, 1)

    mean_bin_size = total_count / max_bin
    rest_bin_count = max_bin
    rest_sample_count = total_count
    is_big_count_value = [False] * num_distinct
    for i, count in enumerate(counts):
        if count >= mean_bin_size:
            is_big_count_value[i] = True
            rest_bin_count -= 1
            rest_sample_count -= count

    if rest_bin_count > 0:
        mean_bin_size = rest_sample_count / rest_bin_count
    upper_bounds = [float("inf")] * max_bin
    lower_bounds = [float("inf")] * max_bin
    bin_count = 0
    lower_bounds[bin_count] = distinct_values[0]
    current_count = 0

    for i in range(num_distinct - 1):
        if not is_big_count_value[i]:
            rest_sample_count -= counts[i]
        current_count += counts[i]

        next_is_big = is_big_count_value[i + 1]
        enough_for_half_bin = current_count >= max(1.0, mean_bin_size * 0.5)
        if (
            is_big_count_value[i]
            or current_count >= mean_bin_size
            or (next_is_big and enough_for_half_bin)
        ):
            upper_bounds[bin_count] = distinct_values[i]
            bin_count += 1
            lower_bounds[bin_count] = distinct_values[i + 1]
            if bin_count >= max_bin - 1:
                break
            current_count = 0
            if not is_big_count_value[i]:
                rest_bin_count -= 1
                if rest_bin_count > 0:
                    mean_bin_size = rest_sample_count / rest_bin_count

    for i in range(bin_count - 1):
        bin_upper_bound.append(float((upper_bounds[i] + lower_bounds[i + 1]) / 2.0))
    bin_upper_bound.append(float("inf"))
    return bin_upper_bound


def bin_feature_frame(df: pd.DataFrame, max_bin: int = 31) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == ID_COLUMN:
            continue
        edges = greedy_bin_edges(out[col].to_numpy(), max_bin=max_bin)
        out[col] = np.digitize(out[col], [-np.inf] + edges)
        out.loc[out[col] == len(edges) + 1, col] = 0
        max_value = out[col].max()
        out[col] = out[col] / max_value if max_value else out[col]
    return out


def prepare_taiwan(
    output_dir: str | Path,
    seed: int = 0,
    source_csv: str | Path | None = None,
    train_ratio: float = 0.7,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if source_csv is None:
        features, target = fetch_taiwan_from_uci()
    else:
        features, target = read_taiwan_csv(source_csv)
    all_features, labels = add_customer_id(features, target)
    train_ids, test_ids = split_ids(labels, seed=seed, train_ratio=train_ratio)

    train_labels = labels.merge(train_ids, on=ID_COLUMN)
    test_labels = labels.merge(test_ids, on=ID_COLUMN)
    train_features = all_features.merge(train_ids, on=ID_COLUMN)
    test_features = all_features.merge(test_ids, on=ID_COLUMN)

    series = build_taiwan_series(all_features)
    train_series = series.merge(train_ids, on=ID_COLUMN)
    test_series = series.merge(test_ids, on=ID_COLUMN)
    train_idx = sequence_index(train_series)
    test_idx = sequence_index(test_series)

    all_feature_branch = bin_feature_frame(all_features)
    train_feature_branch = bin_feature_frame(train_features)
    test_feature_branch = bin_feature_frame(test_features)

    all_features.to_csv(output_dir / "all_feature.csv", index=False)
    labels.to_csv(output_dir / "label.csv", index=False)
    train_labels.to_csv(output_dir / "train_labels.csv", index=False)
    test_labels.to_csv(output_dir / "test_labels.csv", index=False)
    write_table(train_features, output_dir / "train.feather")
    write_table(test_features, output_dir / "test.feather")
    write_table(train_series, output_dir / "df_nn_series_train.feather")
    write_table(test_series, output_dir / "df_nn_series_test.feather")
    write_table(train_idx, output_dir / "df_nn_series_idx_train.feather")
    write_table(test_idx, output_dir / "df_nn_series_idx_test.feather")
    write_table(all_feature_branch, output_dir / "nn_all_feature.feather")
    write_table(train_feature_branch, output_dir / "df_nn_feature_train.feather")
    write_table(test_feature_branch, output_dir / "df_nn_feature_test.feather")

    metadata = {
        "seed": seed,
        "train_ratio": train_ratio,
        "rows": int(len(labels)),
        "train_customers": int(len(train_labels)),
        "test_customers": int(len(test_labels)),
        "default_count": int(labels[TARGET_COLUMN].sum()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
