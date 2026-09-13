from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def global_feature_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    return (
        pd.DataFrame({"feature": X.columns, "importance": result.importances_mean})
        .sort_values("importance", ascending=False, ignore_index=True)
    )


def shap_global_importance(pipeline: Pipeline, X: pd.DataFrame, sample_size: int = 200) -> pd.DataFrame:
    """Return mean absolute SHAP importance for transformed model features."""
    import shap

    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    sample = X.head(sample_size)
    transformed = preprocessor.transform(sample)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.Explainer(model, transformed)
    values = explainer(transformed)
    raw = np.asarray(values.values)
    if raw.ndim == 3:
        raw = raw[:, :, -1]
    importance = np.abs(raw).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    )
