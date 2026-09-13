from pathlib import Path

import joblib

from churn_ml.data import generate_synthetic_churn
from churn_ml.model import FEATURE_COLUMNS, train_and_compare


def test_train_and_compare_evaluates_three_models_and_returns_valid_metrics(tmp_path: Path):
    frame = generate_synthetic_churn(n_rows=500, seed=42)
    result = train_and_compare(frame, artifact_dir=tmp_path, random_state=42, tune=False, cv_folds=2)

    assert set(result.metrics) == {"logistic_regression", "random_forest", "xgboost"}
    for metrics in result.metrics.values():
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1
        assert 0 <= metrics["cv_roc_auc"] <= 1
        assert sum(metrics["confusion_matrix"]) > 0

    assert 0.2 <= result.selected_threshold <= 0.8
    assert not result.threshold_curve.empty
    assert result.selected_threshold in result.threshold_curve["threshold"].tolist()
    assert result.best_model_name in result.metrics
    assert (tmp_path / "churn_model.joblib").exists()
    assert (tmp_path / "metrics.json").exists()


def test_persisted_pipeline_predicts_probability_for_raw_features(tmp_path: Path):
    frame = generate_synthetic_churn(n_rows=350, seed=19)
    train_and_compare(frame, artifact_dir=tmp_path, random_state=19, tune=False, cv_folds=2)
    pipeline = joblib.load(tmp_path / "churn_model.joblib")
    sample = frame.loc[[0], FEATURE_COLUMNS]
    probability = float(pipeline.predict_proba(sample)[0, 1])
    assert 0 <= probability <= 1
