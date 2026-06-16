from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from credit_default_prediction.data.amex import (
    DATE_COLUMN,
    denoise_amex_frame,
    limit_customers,
    load_raw_amex,
)
from credit_default_prediction.data.sequence import ID_COLUMN, TARGET_COLUMN, read_table
from credit_default_prediction.metrics import score_binary_predictions


TabularFeatureSet = Literal["manual", "series_oof"]


def lgbm_params(seed: int, num_threads: int | None = 24) -> dict:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting": "dart",
        "max_depth": -1,
        "num_leaves": 64,
        "learning_rate": 0.035,
        "bagging_freq": 5,
        "bagging_fraction": 0.75,
        "feature_fraction": 0.05,
        "min_data_in_leaf": 256,
        "max_bin": 63,
        "min_data_in_bin": 256,
        "tree_learner": "serial",
        "boost_from_average": "false",
        "lambda_l1": 0.1,
        "lambda_l2": 30.0,
        "verbosity": 1,
        "seed": seed,
    }
    if num_threads is not None:
        params["num_threads"] = num_threads
    return params


def series_lgb_params(seed: int, num_threads: int | None = 24) -> dict:
    params = lgbm_params(seed=seed, num_threads=num_threads)
    params.update({"bagging_fraction": 0.7, "feature_fraction": 0.7})
    return params


def _best_iteration(model: lgb.Booster) -> int | None:
    best_iteration = getattr(model, "best_iteration", None)
    return int(best_iteration) if best_iteration else None


def _attach_missing_ids(features: pd.DataFrame, labels: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    if ID_COLUMN in features.columns:
        return features
    sample_path = data_dir / "sample_submission.csv"
    if not sample_path.exists():
        raise ValueError(
            f"{data_dir / 'all_feature.feather'} does not contain {ID_COLUMN}, "
            f"and {sample_path} is unavailable for test-row alignment."
        )
    sample = read_table(sample_path)[[ID_COLUMN]]
    ids = pd.concat([labels[[ID_COLUMN]], sample], axis=0, ignore_index=True)
    if len(ids) != len(features):
        raise ValueError("Cannot attach customer_ID: label/sample rows do not match all_feature rows.")
    out = features.copy()
    out.insert(0, ID_COLUMN, ids[ID_COLUMN].to_numpy())
    return out


def _coerce_lgbm_feature_dtypes(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        train_ok = pd.api.types.is_numeric_dtype(train[col]) or pd.api.types.is_bool_dtype(train[col])
        test_ok = pd.api.types.is_numeric_dtype(test[col]) or pd.api.types.is_bool_dtype(test[col])
        if train_ok and test_ok:
            continue
        train[col] = pd.to_numeric(train[col], errors="coerce").astype("float32")
        test[col] = pd.to_numeric(test[col], errors="coerce").astype("float32")


def load_lgbm_data(
    data_dir: str | Path,
    feature_set: TabularFeatureSet,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    data_dir = Path(data_dir)
    features = read_table(data_dir / "all_feature.feather")
    labels = read_table(data_dir / "train_labels.csv")[[ID_COLUMN, TARGET_COLUMN]]
    features = _attach_missing_ids(features, labels, data_dir)

    train_count = len(labels)
    train = features.iloc[:train_count].reset_index(drop=True)
    test = features.iloc[train_count:].reset_index(drop=True)
    train = train.drop(columns=[TARGET_COLUMN], errors="ignore").merge(
        labels, on=ID_COLUMN, how="left", validate="one_to_one"
    )
    excluded = {ID_COLUMN, TARGET_COLUMN, DATE_COLUMN}
    selected = [col for col in train.columns if col not in excluded]
    if feature_set == "manual":
        selected = [col for col in selected if "target" not in col]
    elif feature_set != "series_oof":
        raise ValueError(f"Unsupported AMEX LightGBM feature set: {feature_set}")
    _coerce_lgbm_feature_dtypes(train, test, selected)
    return train, test, selected


def load_series_lgb_data(
    raw_dir: str | Path,
    nrows: int | None = None,
    max_train_customers: int | None = None,
    max_test_customers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw_dir = Path(raw_dir)
    train, test, source = load_raw_amex(raw_dir, nrows=nrows)
    if source == "csv":
        train = denoise_amex_frame(train)
        test = denoise_amex_frame(test)
    train = limit_customers(train, max_train_customers)
    test = limit_customers(test, max_test_customers)
    labels = read_table(raw_dir / "train_labels.csv")[[ID_COLUMN, TARGET_COLUMN]]
    if max_train_customers is not None:
        labels = labels[labels[ID_COLUMN].isin(train[ID_COLUMN].drop_duplicates())]
    train = train.merge(labels, on=ID_COLUMN, how="left", validate="many_to_one")
    features = [
        col
        for col in train.columns
        if col not in {ID_COLUMN, TARGET_COLUMN, DATE_COLUMN}
        and col in test.columns
        and pd.api.types.is_numeric_dtype(train[col])
    ]
    return train, test, features


def _split_indices(
    train: pd.DataFrame,
    labels: np.ndarray,
    folds: int,
    seed: int,
    group_by_customer: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    if not group_by_customer:
        return [
            (np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64))
            for train_idx, valid_idx in splitter.split(np.zeros(len(labels)), labels)
        ]

    customer_labels = train[[ID_COLUMN, TARGET_COLUMN]].drop_duplicates(ID_COLUMN).reset_index(drop=True)
    splits = []
    for train_idx, valid_idx in splitter.split(np.zeros(len(customer_labels)), customer_labels[TARGET_COLUMN]):
        train_ids = set(customer_labels.iloc[train_idx][ID_COLUMN])
        valid_ids = set(customer_labels.iloc[valid_idx][ID_COLUMN])
        row_train = train.index[train[ID_COLUMN].isin(train_ids)].to_numpy(dtype=np.int64)
        row_valid = train.index[train[ID_COLUMN].isin(valid_ids)].to_numpy(dtype=np.int64)
        splits.append((row_train, row_valid))
    return splits


def train_amex_lgbm_cv(
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    features: list[str],
    output_dir: str | Path,
    params: dict,
    folds: int = 5,
    rounds: int = 4500,
    early_stopping_rounds: int = 100,
    log_period: int = 50,
    seed: int = 42,
    group_by_customer: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = train[TARGET_COLUMN].astype(int).to_numpy()
    oof = np.zeros(len(train), dtype=np.float64)
    test_pred = np.zeros(len(test), dtype=np.float64) if test is not None else None
    fold_scores: list[float] = []
    importances: list[pd.DataFrame] = []
    params = {**params, "seed": seed}
    splits = _split_indices(train, labels, folds, seed, group_by_customer)

    for fold, (train_idx, valid_idx) in enumerate(splits):
        train_set = lgb.Dataset(train.iloc[train_idx][features], label=labels[train_idx])
        valid_set = lgb.Dataset(train.iloc[valid_idx][features], label=labels[valid_idx])
        model = lgb.train(
            params,
            train_set=train_set,
            num_boost_round=rounds,
            valid_sets=[train_set, valid_set],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.record_evaluation({}),
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                lgb.log_evaluation(period=log_period),
            ],
        )
        model.save_model(str(output_dir / f"fold_{fold}.txt"))
        valid_pred = model.predict(train.iloc[valid_idx][features], num_iteration=_best_iteration(model))
        oof[valid_idx] = valid_pred
        fold_scores.append(score_binary_predictions(labels[valid_idx], valid_pred, metric="amex"))
        importances.append(
            pd.DataFrame(
                {
                    "feature_name": model.feature_name(),
                    "importance_gain": model.feature_importance(importance_type="gain"),
                    "importance_split": model.feature_importance(importance_type="split"),
                }
            )
        )
        if test is not None and test_pred is not None:
            test_pred += model.predict(test[features], num_iteration=_best_iteration(model)) / folds

    global_score = score_binary_predictions(labels, oof, metric="amex")
    pd.DataFrame({ID_COLUMN: train[ID_COLUMN].to_numpy(), TARGET_COLUMN: labels, "prediction": oof}).to_csv(
        output_dir / "oof.csv", index=False
    )
    if test is not None and test_pred is not None:
        pd.DataFrame({ID_COLUMN: test[ID_COLUMN].to_numpy(), "prediction": test_pred}).to_csv(
            output_dir / "submission.csv", index=False
        )
    if importances:
        importance = pd.concat(importances, axis=0)
        importance = importance.groupby("feature_name", as_index=False).mean()
        importance.sort_values("importance_gain", ascending=False).to_csv(
            output_dir / "feature_importance.csv", index=False
        )

    summary = {
        "metric": "amex",
        "fold_scores": fold_scores,
        "mean_fold_score": float(np.mean(fold_scores)),
        "global_oof_score": float(global_score),
        "features": features,
        "feature_count": len(features),
        "params": params,
        "training": {
            "folds": folds,
            "rounds": rounds,
            "early_stopping_rounds": early_stopping_rounds,
            "group_by_customer": group_by_customer,
        },
    }
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def train_lgbm_cv(
    data_dir: str | Path,
    output_dir: str | Path,
    feature_set: TabularFeatureSet,
    folds: int = 5,
    seed: int = 42,
    rounds: int = 4500,
    early_stopping_rounds: int = 100,
    num_threads: int | None = 24,
) -> dict:
    train, test, features = load_lgbm_data(data_dir, feature_set=feature_set)
    return train_amex_lgbm_cv(
        train=train,
        test=test,
        features=features,
        output_dir=output_dir,
        params=lgbm_params(seed=seed, num_threads=num_threads),
        folds=folds,
        rounds=rounds,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
        group_by_customer=False,
    )


def train_series_lgbm_cv(
    raw_dir: str | Path,
    output_dir: str | Path,
    folds: int = 5,
    seed: int = 42,
    rounds: int = 4500,
    early_stopping_rounds: int = 100,
    num_threads: int | None = 24,
    nrows: int | None = None,
    max_train_customers: int | None = None,
    max_test_customers: int | None = None,
) -> dict:
    train, test, features = load_series_lgb_data(
        raw_dir=raw_dir,
        nrows=nrows,
        max_train_customers=max_train_customers,
        max_test_customers=max_test_customers,
    )
    return train_amex_lgbm_cv(
        train=train,
        test=test,
        features=features,
        output_dir=output_dir,
        params=series_lgb_params(seed=seed, num_threads=num_threads),
        folds=folds,
        rounds=rounds,
        early_stopping_rounds=early_stopping_rounds,
        seed=seed,
        group_by_customer=True,
    )
