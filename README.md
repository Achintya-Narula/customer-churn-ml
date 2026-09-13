# Customer Churn Prediction & Explainability

[![CI](https://github.com/Achintya-Narula/customer-churn-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/Achintya-Narula/customer-churn-ml/actions/workflows/ci.yml)

An end-to-end tabular machine-learning project that predicts customer churn using the public **IBM Telco Customer Churn sample dataset**, compares multiple classifiers with stratified cross-validation, explains model behavior, and exposes the selected model through a FastAPI endpoint.

## Project scope
This repository demonstrates a complete, reproducible tabular ML workflow: data cleaning, EDA, leakage-safe preprocessing, model comparison, hyperparameter tuning, holdout evaluation, threshold analysis, explainability, serialization, API inference, testing, and containerization.

## Stack
Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, FastAPI, joblib, pytest, Docker.

## Dataset
The primary benchmark is `data/Telco-Customer-Churn.csv` from IBM's public Telco Customer Churn sample repository. It contains **7,043 rows and 21 source columns**. The project converts `TotalCharges` to numeric, maps `Churn` to 0/1, and normalizes categorical values into a consistent canonical schema.

See `data/DATA_SOURCE.md` for source and license information.

A deterministic synthetic generator remains in `scripts/generate_data.py` as an optional fallback for learning and tests. **The benchmark metrics below are from the IBM sample dataset, not the synthetic data.**

## Architecture

```text
IBM Telco Customer Churn CSV
        |
        v
Schema normalization + validation + EDA
        |
        v
25% untouched stratified holdout
        |
        +---- 75% training split
                |
                v
        ColumnTransformer
        numeric: median impute -> scale
        categorical: mode impute -> one-hot
                |
                v
        5-fold stratified GridSearchCV
        Logistic Regression | Random Forest | XGBoost
                |
                v
        Select by mean CV ROC-AUC only
                |
                +--> untouched holdout evaluation
                +--> PR-AUC / precision / recall / F1 / confusion matrix
                +--> OOF-training threshold selection
                +--> permutation + SHAP importance
                +--> joblib pipeline -> FastAPI /predict
```

## Leakage controls
- The holdout set is split **before** model tuning.
- Hyperparameters and model family are selected using 5-fold stratified CV on the training split only.
- Numeric imputation/scaling and categorical one-hot encoding live inside the scikit-learn pipeline, so preprocessing is fitted inside each CV fold.
- The optional classification threshold is selected from out-of-fold training predictions rather than the holdout set.
- The holdout set is used only for final performance reporting and post-selection diagnostics.

## Verified benchmark results
Random seed: 42. Test size: 25% stratified holdout (**1,761 rows**). Model selection uses mean 5-fold training CV ROC-AUC.

| Model | CV ROC-AUC | Holdout ROC-AUC | Holdout PR-AUC | Precision @ 0.5 | Recall @ 0.5 | F1 @ 0.5 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8443 | 0.8463 | 0.6340 | 0.6616 | 0.5567 | 0.6047 |
| Random Forest | 0.8454 | 0.8448 | 0.6451 | 0.5389 | 0.7859 | 0.6394 |
| **XGBoost** | **0.8486** | **0.8486** | **0.6585** | 0.6648 | 0.5011 | 0.5714 |

**Selected model:** XGBoost, because it achieved the strongest mean 5-fold CV ROC-AUC. Its untouched holdout ROC-AUC was **0.8486**.

### Threshold analysis
The project also uses out-of-fold training probabilities to select a demonstration threshold that maximizes training OOF F1. That threshold was **0.34**. On the untouched holdout it produced:

- Precision: **0.5584**
- Recall: **0.7366**
- F1: **0.6353**
- ROC-AUC: **0.8486**
- PR-AUC: **0.6585**

Changing the threshold does not change ROC-AUC or PR-AUC because those evaluate ranking across thresholds. It changes the precision/recall trade-off. A production threshold would need business costs, retention capacity, and probability calibration rather than F1 alone.

## Explainability
Permutation importance on the holdout set identifies **contract type** and **tenure** as the strongest raw predictors, followed by internet service, total charges, online security, tech support, and monthly charges.

The SHAP report provides transformed-feature importance for the selected XGBoost model. High-impact transformed features include month-to-month contracts, tenure, fiber-optic internet, lack of online security, monthly charges, and lack of tech support.

## Included features
- IBM dataset loader and canonical-schema normalization.
- Deterministic synthetic fallback generator.
- Validation and compact EDA summary.
- Leakage-safe `ColumnTransformer` preprocessing.
- Logistic Regression, Random Forest, and XGBoost comparison.
- 5-fold stratified hyperparameter tuning with `GridSearchCV`.
- Model selection by CV ROC-AUC, not holdout performance.
- Holdout ROC-AUC, PR-AUC, precision, recall, F1, and confusion matrix.
- Out-of-fold threshold selection and threshold-analysis report.
- Permutation and SHAP explainability.
- Persisted preprocessing+model pipeline with joblib.
- FastAPI `/health` and `/predict` endpoints.
- Dockerfile and automated tests.

## Reproduce the project

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Primary benchmark: train/tune/evaluate on IBM sample data
PYTHONPATH=src python scripts/train.py

# Optional: regenerate synthetic fallback data
PYTHONPATH=src python scripts/generate_data.py

# Verify
pytest

# Run API
PYTHONPATH=src uvicorn churn_ml.api:app --reload
```

Windows PowerShell can set `$env:PYTHONPATH="src"` before the Python/Uvicorn commands.

## Example request

```json
{
  "tenure_months": 8,
  "monthly_charges": 91.5,
  "total_charges": 732.0,
  "senior_citizen": 0,
  "gender": "female",
  "partner": "no",
  "dependents": "no",
  "phone_service": "yes",
  "multiple_lines": "yes",
  "internet_service": "fiber-optic",
  "online_security": "no",
  "online_backup": "yes",
  "device_protection": "no",
  "tech_support": "no",
  "streaming_tv": "yes",
  "streaming_movies": "yes",
  "contract_type": "month-to-month",
  "paperless_billing": "yes",
  "payment_method": "electronic-check"
}
```

The API returns a churn probability plus a `low`, `medium`, or `high` demonstration risk band.

## Repository map

```text
data/Telco-Customer-Churn.csv  IBM public sample dataset
src/churn_ml/data.py           IBM loader, normalization, synthetic fallback, validation, EDA
src/churn_ml/model.py          preprocessing, CV tuning, evaluation, threshold logic, persistence
src/churn_ml/explain.py        permutation and SHAP importance
src/churn_ml/api.py            FastAPI inference boundary
scripts/train.py               complete IBM benchmark workflow
scripts/generate_data.py       optional deterministic synthetic fallback
artifacts/                     selected fitted model + metrics
reports/                       EDA, comparison, thresholds, explainability outputs
tests/                         automated behavior tests
```
