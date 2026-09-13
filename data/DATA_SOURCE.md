# Dataset source

Primary benchmark data: **Telco Customer Churn** from IBM's public sample repository:

- Repository: https://github.com/IBM/telco-customer-churn-on-icp4d
- Source file: `data/Telco-Customer-Churn.csv`
- Original file name: `Telco-Customer-Churn.csv`
- Included rows: 7,043
- Included columns: 21

The upstream IBM repository is distributed under the Apache License 2.0. A copy of that license is included as `IBM_REPOSITORY_LICENSE.txt` beside the dataset.

This portfolio project normalizes column names/category values for modeling but keeps the original CSV unchanged in `data/Telco-Customer-Churn.csv`.

The repository also contains an optional deterministic synthetic-data generator. Synthetic data is useful for learning and testing, but the reported benchmark metrics in the README come from the IBM sample dataset, not from the synthetic fallback.
