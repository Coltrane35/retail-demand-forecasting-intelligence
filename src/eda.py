
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "sales_ca1_foods.parquet"
OUTPUT_DIR = ROOT / "outputs" / "eda"


def run_eda() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PATH)

    print("\n=== DATASET SUMMARY ===")
    print(f"Rows: {len(df):,}")
    print(f"Products: {df['item_id'].nunique():,}")
    print(f"Total units sold: {df['sales'].sum():,}")
    print(f"Average daily units per product: {df['sales'].mean():.2f}")
    print(f"Zero-sales share: {(df['sales'] == 0).mean():.2%}")

    # 1. Total daily sales across products
    daily = df.groupby("date")["sales"].sum().sort_index()

    plt.figure(figsize=(13, 5))
    plt.plot(daily.index, daily.values, linewidth=1)
    plt.title("CA_1 FOODS — Daily Sales")
    plt.xlabel("Date")
    plt.ylabel("Units Sold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "daily_sales.png", dpi=150)
    plt.close()

    # 2. Average daily sales by weekday
    weekday = daily.groupby(daily.index.dayofweek).mean()
    weekday.index = [
        "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"
    ]

    plt.figure(figsize=(9, 5))
    weekday.plot(kind="bar")
    plt.title("Average Sales by Weekday")
    plt.xlabel("Weekday")
    plt.ylabel("Average Daily Units")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "weekday_seasonality.png", dpi=150)
    plt.close()

    # 3. Top 15 products by total sales
    top_products = (
        df.groupby("item_id")["sales"]
        .sum()
        .nlargest(15)
        .sort_values()
    )

    plt.figure(figsize=(10, 7))
    top_products.plot(kind="barh")
    plt.title("Top 15 Products by Total Units Sold")
    plt.xlabel("Total Units Sold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_products.png", dpi=150)
    plt.close()

    # 4. Zero-sales share per product
    zero_share = (
        df.assign(is_zero=df["sales"].eq(0))
        .groupby("item_id")["is_zero"]
        .mean()
    )

    plt.figure(figsize=(10, 5))
    plt.hist(zero_share, bins=30, edgecolor="black")
    plt.title("Distribution of Zero-Sales Share by Product")
    plt.xlabel("Share of Days with Zero Sales")
    plt.ylabel("Number of Products")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "zero_sales_distribution.png", dpi=150)
    plt.close()

    print("\n=== TOP 10 PRODUCTS ===")
    print(top_products.sort_values(ascending=False).head(10))

    print("\n=== ZERO-SALES ANALYSIS ===")
    print(f"Products with >50% zero-sales days: {(zero_share > 0.5).sum():,}")
    print(f"Products with >80% zero-sales days: {(zero_share > 0.8).sum():,}")

    print("\n=== WEEKDAY AVERAGE SALES ===")
    print(weekday.round(2))

    print(f"\nCharts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    run_eda()
