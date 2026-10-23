
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_m5_data(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load the three core M5 datasets."""

    filenames = {
        "sales": "sales_train_validation.csv",
        "calendar": "calendar.csv",
        "prices": "sell_prices.csv",
    }

    datasets = {}

    for name, filename in filenames.items():
        filepath = data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Missing file: {filepath}\n"
                "Download the M5 dataset from Kaggle "
                "and place the CSV files in data/raw/."
            )

        print(f"Loading {filename}...")
        datasets[name] = pd.read_csv(filepath)

    return datasets


def validate_m5_data(datasets: dict[str, pd.DataFrame]) -> None:
    """Validate essential columns and dataset integrity."""

    required_columns = {
        "sales": {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id"},
        "calendar": {"date", "wm_yr_wk", "d"},
        "prices": {"store_id", "item_id", "wm_yr_wk", "sell_price"},
    }

    for name, required in required_columns.items():
        df = datasets[name]
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{name}: missing required columns: {sorted(missing)}"
            )

        if df.empty:
            raise ValueError(f"{name}: dataset is empty")

        print(f"{name.upper()}: {df.shape[0]:,} rows, {df.shape[1]} columns")

    day_columns = [
        col for col in datasets["sales"].columns if col.startswith("d_")
    ]

    if not day_columns:
        raise ValueError("Sales dataset has no daily sales columns.")

    print(f"Daily sales columns: {len(day_columns)}")
    print("Data validation completed successfully.")


if __name__ == "__main__":
    datasets = load_m5_data()
    validate_m5_data(datasets)
