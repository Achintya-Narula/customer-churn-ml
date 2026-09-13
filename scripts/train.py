from pathlib import Path
import json

import pandas as pd

from churn_ml.data import eda_summary, load_ibm_telco
from churn_ml.explain import global_feature_importance, shap_global_importance
from churn_ml.model import train_and_compare


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "data" / "Telco-Customer-Churn.csv"
    frame = load_ibm_telco(source)
    artifact_dir = root / "artifacts"
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)

    (reports_dir / "eda_summary.json").write_text(
        json.dumps(eda_summary(frame), indent=2), encoding="utf-8"
    )

    result = train_and_compare(frame, artifact_dir=artifact_dir, random_state=42, tune=True, cv_folds=5)
    metrics = json.loads((artifact_dir / "metrics.json").read_text(encoding="utf-8"))
    comparison_rows = []
    for model_name, model_metrics in metrics["models"].items():
        comparison_rows.append(
            {
                "model": model_name,
                "cv_roc_auc": model_metrics["cv_roc_auc"],
                "holdout_roc_auc": model_metrics["roc_auc"],
                "holdout_pr_auc": model_metrics["pr_auc"],
                "precision_at_0_5": model_metrics["precision"],
                "recall_at_0_5": model_metrics["recall"],
                "f1_at_0_5": model_metrics["f1"],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(reports_dir / "model_comparison.csv", index=False)

    global_feature_importance(
        result.pipeline,
        result.holdout_X,
        result.holdout_y,
        n_repeats=5,
    ).to_csv(reports_dir / "permutation_importance.csv", index=False)
    shap_global_importance(result.pipeline, result.holdout_X, sample_size=200).to_csv(
        reports_dir / "shap_importance.csv", index=False
    )
    result.threshold_curve.to_csv(
        reports_dir / "threshold_analysis_oof_train.csv", index=False
    )

    print(
        json.dumps(
            {
                "best_model": result.best_model_name,
                "selected_threshold": round(result.selected_threshold, 2),
                "holdout_metrics_at_0_5": result.metrics[result.best_model_name],
                "holdout_metrics_at_selected_threshold": result.threshold_metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
