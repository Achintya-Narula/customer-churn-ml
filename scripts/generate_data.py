from pathlib import Path
import json

from churn_ml.data import eda_summary, generate_synthetic_churn


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    reports_dir = root / "reports"
    data_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    frame = generate_synthetic_churn(n_rows=2500, seed=42)
    frame.to_csv(data_dir / "synthetic_sample_churn.csv", index=False)
    (reports_dir / "synthetic_eda_summary.json").write_text(json.dumps(eda_summary(frame), indent=2), encoding="utf-8")
    print(f"Generated {len(frame)} rows at {data_dir / 'synthetic_sample_churn.csv'}")


if __name__ == "__main__":
    main()
