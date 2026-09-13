from pathlib import Path

from fastapi.testclient import TestClient

from churn_ml.api import create_app
from churn_ml.data import generate_synthetic_churn
from churn_ml.explain import global_feature_importance
from churn_ml.model import FEATURE_COLUMNS, train_and_compare


def _payload(frame):
    row = frame.loc[0, FEATURE_COLUMNS].to_dict()
    for key, value in row.items():
        if hasattr(value, "item"):
            row[key] = value.item()
    return row


def test_global_feature_importance_returns_ranked_features(tmp_path: Path):
    frame = generate_synthetic_churn(n_rows=320, seed=5)
    result = train_and_compare(frame, artifact_dir=tmp_path, random_state=5)
    importance = global_feature_importance(result.pipeline, frame[FEATURE_COLUMNS], frame["churn"], n_repeats=2)
    assert not importance.empty
    assert list(importance.columns) == ["feature", "importance"]
    assert importance.iloc[0]["importance"] >= importance.iloc[-1]["importance"]


def test_predict_endpoint_returns_probability_and_risk_band(tmp_path: Path):
    frame = generate_synthetic_churn(n_rows=320, seed=17)
    train_and_compare(frame, artifact_dir=tmp_path, random_state=17)
    app = create_app(model_path=tmp_path / "churn_model.joblib")
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/predict", json=_payload(frame))
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_band"] in {"low", "medium", "high"}
