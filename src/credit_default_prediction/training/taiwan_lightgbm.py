from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from credit_default_prediction.data.sequence import ID_COLUMN, TARGET_COLUMN, read_table
from credit_default_prediction.metrics import score_binary_predictions


def train_taiwan_lightgbm_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    output_dir: str | Path,
    metric: str,
    label_col: str = TARGET_COLUMN,
    id_col: str = ID_COLUMN,
    folds: int = 5,
    seed: int = 42,
    rounds: int = 1500,
    early_stopping_rounds: int = 100,
    learning_rate: float = 0.035,
    feature_fraction: float = 0.05,
    bagging_fraction: float = 0.75,
    bagging_freq: int = 5,
    num_leaves: int = 64,
    boosting: str = "gbdt",
    max_depth: int | None = None,
    min_data_in_leaf: int | None = None,
    max_bin: int | None = None,
    min_data_in_bin: int | None = None,
    tree_learner: str | None = None,
    boost_from_average: bool | str | None = None,
    lambda_l1: float = 30.0,
    lambda_l2: float = 24.0,
    num_threads: int | None = None,
    verbosity: int = -1,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = [c for c in train_df.columns if c not in {id_col, label_col}]
    labels = train_df[label_col].astype(int).to_numpy()
    oof = np.zeros(len(train_df), dtype=np.float64)
    fold_scores = []
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting": boosting,
        "learning_rate": learning_rate,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "num_leaves": num_leaves,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "verbosity": verbosity,
        "seed": seed,
    }
    optional_params = {
        "max_depth": max_depth,
        "min_data_in_leaf": min_data_in_leaf,
        "max_bin": max_bin,
        "min_data_in_bin": min_data_in_bin,
        "tree_learner": tree_learner,
        "boost_from_average": boost_from_average,
        "num_threads": num_threads,
    }
    params.update({key: value for key, value in optional_params.items() if value is not None})

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    test_pred = np.zeros(len(test_df), dtype=np.float64) if test_df is not None else None
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(train_df[features], labels)):
        train_set = lgb.Dataset(train_df.iloc[train_idx][features], label=labels[train_idx])
        valid_set = lgb.Dataset(train_df.iloc[valid_idx][features], label=labels[valid_idx])
        model = lgb.train(
            params,
            train_set=train_set,
            num_boost_round=rounds,
            valid_sets=[train_set, valid_set],
            callbacks=[
                lgb.log_evaluation(50),
                lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            ],
        )
        model.save_model(str(output_dir / f"fold_{fold}.txt"))
        valid_pred = model.predict(train_df.iloc[valid_idx][features], num_iteration=model.best_iteration)
        oof[valid_idx] = valid_pred
        fold_scores.append(score_binary_predictions(labels[valid_idx], valid_pred, metric=metric))
        if test_df is not None:
            test_pred += model.predict(test_df[features], num_iteration=model.best_iteration) / folds

    global_score = score_binary_predictions(labels, oof, metric=metric)
    pd.DataFrame(
        {id_col: train_df[id_col].values, label_col: labels, "prediction": oof}
    ).to_csv(output_dir / "oof.csv", index=False)
    if test_df is not None:
        pd.DataFrame({id_col: test_df[id_col].values, "prediction": test_pred}).to_csv(
            output_dir / "submission.csv", index=False
        )
    summary = {
        "metric": metric,
        "fold_scores": fold_scores,
        "mean_fold_score": float(np.mean(fold_scores)),
        "global_oof_score": float(global_score),
        "features": features,
        "params": params,
    }
    (output_dir / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_taiwan_feature_branch_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train_features = read_table(data_dir / "df_nn_feature_train.feather")
    test_features = read_table(data_dir / "df_nn_feature_test.feather")
    labels = read_table(data_dir / "train_labels.csv")[[ID_COLUMN, TARGET_COLUMN]]
    train = train_features.merge(labels, on=ID_COLUMN, how="inner")
    return train, test_features
