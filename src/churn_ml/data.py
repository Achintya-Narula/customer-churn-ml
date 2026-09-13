from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "customer_id",
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure_months",
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
    "monthly_charges",
    "total_charges",
    "churn",
]

IBM_COLUMN_MAP = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure_months",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract_type",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}

CATEGORICAL_COLUMNS = [
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


def _slug(value: Any) -> Any:
    if pd.isna(value):
        return value
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text


def load_ibm_telco(path: str | Path) -> pd.DataFrame:
    """Load IBM Telco Customer Churn data into the project's canonical schema."""
    raw = pd.read_csv(path)
    missing = [column for column in IBM_COLUMN_MAP if column not in raw.columns]
    if missing:
        raise ValueError(f"IBM dataset is missing columns: {', '.join(missing)}")

    frame = raw[list(IBM_COLUMN_MAP)].rename(columns=IBM_COLUMN_MAP).copy()
    frame["total_charges"] = pd.to_numeric(frame["total_charges"], errors="coerce")
    frame["monthly_charges"] = pd.to_numeric(frame["monthly_charges"], errors="coerce")
    frame["tenure_months"] = pd.to_numeric(frame["tenure_months"], errors="coerce")
    frame["senior_citizen"] = pd.to_numeric(frame["senior_citizen"], errors="coerce").astype("Int64")
    for column in CATEGORICAL_COLUMNS:
        frame[column] = frame[column].map(_slug)
    frame["churn"] = frame["churn"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    return frame[REQUIRED_COLUMNS]


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def generate_synthetic_churn(n_rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Create deterministic telecom-like data matching the IBM-style canonical schema."""
    if n_rows < 20:
        raise ValueError("n_rows must be at least 20")
    rng = np.random.default_rng(seed)

    gender = rng.choice(["female", "male"], size=n_rows)
    senior = rng.choice([0, 1], p=[0.83, 0.17], size=n_rows)
    partner = rng.choice(["yes", "no"], p=[0.48, 0.52], size=n_rows)
    dependents = rng.choice(["yes", "no"], p=[0.30, 0.70], size=n_rows)
    tenure = rng.integers(1, 73, size=n_rows)
    phone_service = rng.choice(["yes", "no"], p=[0.90, 0.10], size=n_rows)
    multiple_lines = np.where(
        phone_service == "no",
        "no-phone-service",
        rng.choice(["yes", "no"], p=[0.43, 0.57], size=n_rows),
    )
    internet = rng.choice(["fiber-optic", "dsl", "no"], p=[0.44, 0.39, 0.17], size=n_rows)

    def internet_addon(yes_probability: float) -> np.ndarray:
        draw = rng.choice(["yes", "no"], p=[yes_probability, 1 - yes_probability], size=n_rows)
        return np.where(internet == "no", "no-internet-service", draw)

    online_security = internet_addon(0.42)
    online_backup = internet_addon(0.46)
    device_protection = internet_addon(0.44)
    tech_support = internet_addon(0.41)
    streaming_tv = internet_addon(0.49)
    streaming_movies = internet_addon(0.49)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], p=[0.56, 0.25, 0.19], size=n_rows)
    paperless = rng.choice(["yes", "no"], p=[0.67, 0.33], size=n_rows)
    payment = rng.choice(
        ["electronic-check", "credit-card-automatic", "bank-transfer-automatic", "mailed-check"],
        p=[0.34, 0.25, 0.23, 0.18],
        size=n_rows,
    )

    addon_count = sum(
        (feature == "yes").astype(int)
        for feature in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
    )
    base_charge = np.where(internet == "fiber-optic", 72, np.where(internet == "dsl", 48, 20))
    monthly = np.clip(base_charge + addon_count * 5 + rng.normal(0, 8, size=n_rows) + senior * 2, 18, 125).round(2)
    total = np.clip(monthly * tenure + rng.normal(0, 75, size=n_rows), 0, None).round(2)

    logit = (
        -1.15
        + 1.25 * (contract == "month-to-month")
        - 0.65 * (contract == "two-year")
        + 0.60 * (internet == "fiber-optic")
        + 0.55 * (tech_support == "no")
        + 0.35 * (online_security == "no")
        + 0.45 * (payment == "electronic-check")
        + 0.35 * senior
        + 0.25 * (paperless == "yes")
        - 0.028 * tenure
        + 0.012 * (monthly - 60)
        - 0.30 * (partner == "yes")
        - 0.25 * (dependents == "yes")
    )
    churn_probability = _sigmoid(logit)
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{idx:05d}" for idx in range(1, n_rows + 1)],
            "gender": gender,
            "senior_citizen": senior,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract_type": contract,
            "paperless_billing": paperless,
            "payment_method": payment,
            "monthly_charges": monthly,
            "total_charges": total,
            "churn": churn,
        },
        columns=REQUIRED_COLUMNS,
    )


def validate_churn_frame(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        errors.append(f"Missing required columns: {', '.join(missing)}")
    if "customer_id" in frame and frame["customer_id"].duplicated().any():
        errors.append("customer_id must be unique")
    if "churn" in frame and not set(frame["churn"].dropna().unique()).issubset({0, 1}):
        errors.append("churn must contain only 0/1 values")
    numeric_columns = ["tenure_months", "monthly_charges", "total_charges"]
    for column in numeric_columns:
        if column in frame:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if (numeric < 0).any():
                errors.append(f"{column} cannot contain negative values")
    return errors


def eda_summary(frame: pd.DataFrame) -> dict[str, Any]:
    errors = validate_churn_frame(frame)
    if errors:
        raise ValueError("; ".join(errors))
    total_charges = pd.to_numeric(frame["total_charges"], errors="coerce")
    return {
        "rows": int(len(frame)),
        "churn_rate": round(float(frame["churn"].mean()), 4),
        "tenure_months_mean": round(float(pd.to_numeric(frame["tenure_months"], errors="coerce").mean()), 2),
        "monthly_charges_mean": round(float(pd.to_numeric(frame["monthly_charges"], errors="coerce").mean()), 2),
        "total_charges_mean": round(float(total_charges.mean()), 2),
        "missing_total_charges": int(total_charges.isna().sum()),
        "contract_mix": {str(k): int(v) for k, v in frame["contract_type"].value_counts().to_dict().items()},
        "internet_mix": {str(k): int(v) for k, v in frame["internet_service"].value_counts().to_dict().items()},
    }
