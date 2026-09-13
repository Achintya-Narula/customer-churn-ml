from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .data import validate_churn_frame

TARGET = "churn"
ID_COLUMN = "customer_id"
NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges", "senior_citizen"]
CATEGORICAL_FEATURES = [
    "gender",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract_type",
    "paperless_billing",
    "payment_method",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class TrainingResult:
    best_model_name: str
    pipeline: Pipeline
    metrics: dict[str, dict[str, Any]]
    test_rows: int
    selected_threshold: float
    threshold_metrics: dict[str, Any]
    holdout_X: pd.DataFrame
    holdout_y: pd.Series
    threshold_curve: pd.DataFrame


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def _candidates(random_state: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        ),
    }


def _parameter_grids() -> dict[str, dict[str, list[Any]]]:
    return {
        "logistic_regression": {
            "model__C": [0.5, 3.0],
            "model__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "model__n_estimators": [300, 450],
            "model__max_depth": [8, None],
            "model__min_samples_leaf": [4],
            "model__max_features": ["sqrt"],
        },
        "xgboost": {
            "model__n_estimators": [250, 350],
            "model__max_depth": [2, 3],
            "model__learning_rate": [0.03],
            "model__min_child_weight": [3],
            "model__subsample": [0.85],
            "model__colsample_bytree": [0.85],
            "model__reg_lambda": [1],
        },
    }


def _evaluate(y_true: pd.Series, probability: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, prediction)
    tn, fp, fn, tp = matrix.ravel()
    return {
        "precision": round(float(precision_score(y_true, prediction, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, prediction, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, prediction, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probability)), 4),
        "confusion_matrix": [int(tn), int(fp), int(fn), int(tp)],
    }


def _best_f1_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    candidates = np.linspace(0.20, 0.80, 61)
    scored = [
        (float(f1_score(y_true, (probability >= threshold).astype(int), zero_division=0)), float(threshold))
        for threshold in candidates
    ]
    return round(max(scored, key=lambda pair: (pair[0], -abs(pair[1] - 0.5)))[1], 2)


def threshold_analysis(y_true: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for threshold in np.linspace(0.20, 0.80, 61):
        prediction = (probability >= threshold).astype(int)
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": round(float(precision_score(y_true, prediction, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, prediction, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, prediction, zero_division=0)), 4),
            }
        )
    return pd.DataFrame(rows)


def train_and_compare(
    frame: pd.DataFrame,
    artifact_dir: str | Path,
    random_state: int = 42,
    *,
    tune: bool = True,
    cv_folds: int = 5,
) -> TrainingResult:
    errors = validate_churn_frame(frame)
    if errors:
        raise ValueError("; ".join(errors))
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")

    X = frame[FEATURE_COLUMNS].copy()
    y = frame[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    metrics: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Pipeline] = {}
    grids = _parameter_grids()

    for name, estimator in _candidates(random_state).items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        if tune:
            search = GridSearchCV(
                pipeline,
                grids[name],
                scoring="roc_auc",
                cv=cv,
                n_jobs=1,
                refit=True,
            )
            search.fit(X_train, y_train)
            fitted_pipeline = search.best_estimator_
            cv_roc_auc = float(search.best_score_)
            best_params = {key.replace("model__", ""): value for key, value in search.best_params_.items()}
        else:
            cv_roc_auc = float(cross_val_score(pipeline, X_train, y_train, scoring="roc_auc", cv=cv, n_jobs=1).mean())
            fitted_pipeline = pipeline.fit(X_train, y_train)
            best_params = {}

        probability = fitted_pipeline.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)
        model_metrics = _evaluate(y_test, probability, prediction)
        model_metrics["cv_roc_auc"] = round(cv_roc_auc, 4)
        model_metrics["best_params"] = best_params
        metrics[name] = model_metrics
        fitted[name] = fitted_pipeline

    # Select by training-only cross-validation, not by the holdout test set.
    best_model_name = max(metrics, key=lambda name: metrics[name]["cv_roc_auc"])
    best_pipeline = fitted[best_model_name]

    oof_probability = cross_val_predict(
        clone(best_pipeline),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    selected_threshold = _best_f1_threshold(y_train, oof_probability)
    threshold_curve = threshold_analysis(y_train, oof_probability)
    holdout_probability = best_pipeline.predict_proba(X_test)[:, 1]
    threshold_prediction = (holdout_probability >= selected_threshold).astype(int)
    threshold_metrics = _evaluate(y_test, holdout_probability, threshold_prediction)

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, artifact_path / "churn_model.joblib")
    payload = {
        "best_model": best_model_name,
        "selection_metric": "mean_stratified_cv_roc_auc",
        "cv_folds": cv_folds,
        "test_rows": int(len(X_test)),
        "selected_threshold": round(selected_threshold, 2),
        "threshold_source": "out-of-fold predictions on the training split",
        "threshold_metrics_on_holdout": threshold_metrics,
        "models": metrics,
    }
    (artifact_path / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return TrainingResult(
        best_model_name=best_model_name,
        pipeline=best_pipeline,
        metrics=metrics,
        test_rows=int(len(X_test)),
        selected_threshold=selected_threshold,
        threshold_metrics=threshold_metrics,
        holdout_X=X_test,
        holdout_y=y_test,
        threshold_curve=threshold_curve,
    )
