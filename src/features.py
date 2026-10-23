
from pathlib import Path

import pandas as pd

from src.load_data import load_m5_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def prepare_sales_data(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    store_id: str = "CA_1",
    category: str = "FOODS",
) -> pd.DataFrame:
    """Prepare daily product-level sales for forecasting."""

    filtered = sales.loc[
        (sales["store_id"] == store_id)
        & (sales["cat_id"] == category)
    ].copy()

    if filtered.empty:
        raise ValueError(
            f"No sales found for store={store_id}, category={category}"
        )

    day_columns = [
        col for col in filtered.columns
        if col.startswith("d_") and col[2:].isdigit()
    ]

    long_sales = filtered.melt(
        id_vars=["item_id", "store_id", "cat_id"],
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )

    calendar_dates = calendar[["d", "date"]].copy()

    result = long_sales.merge(
        calendar_dates,
        on="d",
        how="left",
        validate="many_to_one",
    )

    if result["date"].isna().any():
        raise ValueError("Some sales days have no calendar date.")

    result["date"] = pd.to_datetime(result["date"])
    result["sales"] = pd.to_numeric(result["sales"])

    result = result[
        ["date", "store_id", "cat_id", "item_id", "sales"]
    ].sort_values(["item_id", "date"])

    return result.reset_index(drop=True)


if __name__ == "__main__":
    datasets = load_m5_data()

    prepared = prepare_sales_data(
        sales=datasets["sales"],
        calendar=datasets["calendar"],
        store_id="CA_1",
        category="FOODS",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "sales_ca1_foods.parquet"
    prepared.to_parquet(output_path, index=False)

    print("\nPrepared dataset:")
    print(prepared.head())

    print("\nSummary:")
    print(f"Rows: {len(prepared):,}")
    print(f"Products: {prepared['item_id'].nunique():,}")
    print(f"Date range: {prepared['date'].min()} to {prepared['date'].max()}")
    print(f"Missing sales: {prepared['sales'].isna().sum():,}")
    print(f"Negative sales: {(prepared['sales'] < 0).sum():,}")
    print(f"\nSaved to: {output_path}")
