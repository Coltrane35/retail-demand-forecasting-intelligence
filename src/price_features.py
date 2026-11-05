
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SALES_PATH = (
    ROOT / "data" / "processed" / "sales_ca1_foods.parquet"
)

PRICES_PATH = (
    ROOT / "data" / "raw" / "sell_prices.csv"
)

CALENDAR_PATH = (
    ROOT / "data" / "raw" / "calendar.csv"
)


def inspect_price_data():
    print("Loading sales data...")
    sales = pd.read_parquet(SALES_PATH)

    print("Loading M5 calendar...")
    calendar = pd.read_csv(
        CALENDAR_PATH,
        usecols=["date", "wm_yr_wk"],
    )

    print("Loading selling prices...")
    prices = pd.read_csv(
        PRICES_PATH,
        usecols=[
            "store_id",
            "item_id",
            "wm_yr_wk",
            "sell_price",
        ],
    )

    # Our project focuses on store CA_1.
    prices = prices[
        prices["store_id"] == "CA_1"
    ].copy()

    sales["date"] = pd.to_datetime(
        sales["date"]
    )

    calendar["date"] = pd.to_datetime(
        calendar["date"]
    )

    print("\n=== PRICE DATA ===")
    print(prices.head())

    print(
        f"\nPrice records: {len(prices):,}"
    )

    print(
        f"Products: {prices['item_id'].nunique():,}"
    )

    print(
        "Missing prices:",
        f"{prices['sell_price'].isna().sum():,}",
    )

    # -----------------------------------------
    # STEP 1: Join sales with M5 calendar
    # -----------------------------------------

    print("\n=== JOIN VALIDATION ===")

    if calendar["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates in calendar."
        )

    sales_with_week = sales.merge(
        calendar,
        on="date",
        how="left",
        validate="many_to_one",
    )

    if sales_with_week["wm_yr_wk"].isna().any():
        raise ValueError(
            "Missing week identifiers."
        )

    # -----------------------------------------
    # STEP 2: Validate price table
    # -----------------------------------------

    price_keys = [
        "store_id",
        "item_id",
        "wm_yr_wk",
    ]

    if prices.duplicated(price_keys).any():
        raise ValueError(
            "Duplicate product-week prices."
        )

    # -----------------------------------------
    # STEP 3: Join sales with weekly prices
    # -----------------------------------------

    merged = sales_with_week.merge(
        prices,
        on=price_keys,
        how="left",
        validate="many_to_one",
    )

    if len(merged) != len(sales):
        raise ValueError(
            "Row count changed after price merge."
        )

    missing_count = (
        merged["sell_price"].isna().sum()
    )

    missing_pct = (
        100 * missing_count / len(merged)
    )

    print(
        f"Sales observations: {len(sales):,}"
    )

    print(
        f"Merged observations: {len(merged):,}"
    )

    print(
        f"Missing matched prices: {missing_count:,}"
    )

    print(
        f"Missing price share: {missing_pct:.2f}%"
    )

    # -----------------------------------------
    # STEP 4: Price distribution
    # -----------------------------------------

    print("\n=== PRICE DISTRIBUTION ===")

    print(
        merged["sell_price"]
        .describe()
        .round(3)
    )

    # -----------------------------------------
    # STEP 5: Missing price diagnostics
    # -----------------------------------------

    print("\n=== MISSING PRICE DIAGNOSTICS ===")

    # First date with an available price
    # for each product in our sales dataset.
    first_price_date = (
        merged.loc[
            merged["sell_price"].notna()
        ]
        .groupby("item_id")["date"]
        .min()
        .rename("first_price_date")
    )

    diagnostic = merged.join(
        first_price_date,
        on="item_id",
    )

    missing = diagnostic[
        diagnostic["sell_price"].isna()
    ].copy()

    before_first_price = (
        missing["date"]
        < missing["first_price_date"]
    )

    after_first_price = (
        missing["date"]
        >= missing["first_price_date"]
    )

    never_priced = (
        missing["first_price_date"].isna()
    )

    print(
        "Missing before first available price:",
        f"{int(before_first_price.sum()):,}",
    )

    print(
        "Missing after first available price:",
        f"{int(after_first_price.sum()):,}",
    )

    print(
        "Missing for products without any price:",
        f"{int(never_priced.sum()):,}",
    )

    # -----------------------------------------
    # STEP 6: Sales on missing-price days
    # -----------------------------------------

    print("\n=== SALES WITHOUT PRICE ===")

    missing_sales_units = (
        missing["sales"].sum()
    )

    positive_sales_days = (
        (missing["sales"] > 0).sum()
    )

    print(
        "Sales units on missing-price days:",
        f"{int(missing_sales_units):,}",
    )

    print(
        "Positive-sales days without price:",
        f"{int(positive_sales_days):,}",
    )

    # -----------------------------------------
    # STEP 7: Product-level diagnostics
    # -----------------------------------------

    print("\n=== PRODUCTS WITH MOST MISSING PRICES ===")

    missing_by_product = (
        missing.groupby("item_id")
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    print(missing_by_product)

    # -----------------------------------------
    # STEP 8: Final validation
    # -----------------------------------------

    total_missing_categories = (
        before_first_price.sum()
        + after_first_price.sum()
        + never_priced.sum()
    )

    if total_missing_categories != missing_count:
        raise ValueError(
            "Missing-price categories do not "
            "sum to total missing observations."
        )

    print("\n=== VALIDATION SUMMARY ===")
    print("Price merge: OK")
    print("Duplicate price keys: NONE")
    print("Row count preserved: YES")
    print("Missing-price classification: OK")

    print("\nPrice data inspection completed.")


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add historical price features for direct 28-day forecasts.

    The features use prices from at least 28 days
    before the predicted date.
    """
    calendar = pd.read_csv(
        CALENDAR_PATH,
        usecols=["date", "wm_yr_wk"],
        parse_dates=["date"],
    )

    prices = pd.read_csv(
        PRICES_PATH,
        usecols=[
            "store_id",
            "item_id",
            "wm_yr_wk",
            "sell_price",
        ],
    )

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])

    # Avoid duplicate columns if the function is called
    # on an already enriched DataFrame.
    if "wm_yr_wk" in result.columns:
        result = result.drop(columns=["wm_yr_wk"])

    if "sell_price" in result.columns:
        result = result.drop(columns=["sell_price"])

    result = result.merge(
        calendar,
        on="date",
        how="left",
        validate="many_to_one",
    )

    prices = prices[
        prices["store_id"].isin(
            result["store_id"].unique()
        )
        & prices["item_id"].isin(
            result["item_id"].unique()
        )
    ].copy()

    result = result.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )

    result = result.sort_values(
        ["store_id", "item_id", "date"]
    ).copy()

    grouped = result.groupby(
        ["store_id", "item_id"]
    )["sell_price"]

    result["price_lag_28"] = (
        grouped.shift(28)
    )

    result["price_lag_56"] = (
        grouped.shift(56)
    )

    result["price_rolling_mean_28"] = (
        grouped.transform(
            lambda s: s.shift(28).rolling(
                28, min_periods=1
            ).mean()
        )
    )

    result["price_change_28"] = (
        result["price_lag_28"]
        / result["price_lag_56"].replace(0, float("nan"))
        - 1
    )

    result["price_available_28"] = (
        result["price_lag_28"].notna().astype(int)
    )

    result = result.drop(
        columns=["wm_yr_wk", "sell_price"]
    )

    return result


if __name__ == "__main__":
    inspect_price_data()
