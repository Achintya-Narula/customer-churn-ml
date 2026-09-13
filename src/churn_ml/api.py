from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=500)
    total_charges: float | None = Field(default=None, ge=0)
    senior_citizen: int = Field(ge=0, le=1)
    gender: str
    partner: str
    dependents: str
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract_type: str
    paperless_billing: str
    payment_method: str


def _risk_band(probability: float) -> str:
    if probability < 0.35:
        return "low"
    if probability < 0.65:
        return "medium"
    return "high"


def create_app(model_path: str | Path = "artifacts/churn_model.joblib") -> FastAPI:
    path = Path(model_path)
    app = FastAPI(title="Customer Churn Prediction API", version="2.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(payload: ChurnRequest) -> dict[str, float | str]:
        if not path.exists():
            raise HTTPException(status_code=503, detail="Model artifact is not available. Run training first.")
        model = joblib.load(path)
        frame = pd.DataFrame([payload.model_dump()])
        probability = float(model.predict_proba(frame)[0, 1])
        return {
            "churn_probability": round(probability, 4),
            "risk_band": _risk_band(probability),
        }

    return app


app = create_app()
