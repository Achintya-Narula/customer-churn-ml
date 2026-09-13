from pathlib import Path

import pandas as pd

from churn_ml.data import (
    REQUIRED_COLUMNS,
    eda_summary,
    generate_synthetic_churn,
    load_ibm_telco,
    validate_churn_frame,
)


def test_generate_synthetic_churn_is_deterministic_and_complete():
    first = generate_synthetic_churn(n_rows=120, seed=7)
    second = generate_synthetic_churn(n_rows=120, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == REQUIRED_COLUMNS
    assert len(first) == 120
    assert set(first["churn"].unique()).issubset({0, 1})
    assert first["churn"].nunique() == 2
    assert {"online_security", "online_backup", "device_protection", "streaming_tv"}.issubset(first.columns)


def test_load_ibm_telco_normalizes_schema_and_target(tmp_path: Path):
    raw = pd.DataFrame(
        [
            {
                "customerID": "A-1",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": " ",
                "Churn": "Yes",
            }
        ]
    )
    source = tmp_path / "telco.csv"
    raw.to_csv(source, index=False)

    frame = load_ibm_telco(source)

    assert list(frame.columns) == REQUIRED_COLUMNS
    assert frame.loc[0, "customer_id"] == "A-1"
    assert frame.loc[0, "contract_type"] == "month-to-month"
    assert frame.loc[0, "payment_method"] == "electronic-check"
    assert pd.isna(frame.loc[0, "total_charges"])
    assert frame.loc[0, "churn"] == 1


def test_validate_churn_frame_reports_missing_columns_and_invalid_target():
    frame = generate_synthetic_churn(n_rows=50, seed=3).drop(columns=["monthly_charges"])
    frame.loc[0, "churn"] = 4
    errors = validate_churn_frame(frame)
    assert any("monthly_charges" in error for error in errors)
    assert any("churn" in error.lower() for error in errors)


def test_eda_summary_contains_core_business_statistics():
    frame = generate_synthetic_churn(n_rows=80, seed=11)
    summary = eda_summary(frame)
    assert summary["rows"] == 80
    assert 0 < summary["churn_rate"] < 1
    assert "monthly_charges_mean" in summary
    assert "missing_total_charges" in summary
    assert set(summary["contract_mix"]) == set(frame["contract_type"].unique())
