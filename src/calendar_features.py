
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CALENDAR_PATH = ROOT / "data" / "raw" / "calendar.csv"
OUTPUT_PATH = (
    ROOT / "data" / "processed" / "calendar_features.parquet"
)


def prepare_calendar_features() -> pd.DataFrame:
    print("Loading M5 calendar...")

    calendar = pd.read_csv(CALENDAR_PATH)

    required_columns = [
        "date",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "snap_CA",
    ]

    missing = [
        col for col in required_columns
        if col not in calendar.columns
    ]

    if missing:
        raise ValueError(
            f"Missing calendar columns: {missing}"
        )

    calendar["date"] = pd.to_datetime(
        calendar["date"]
    )

    calendar["has_event"] = (
        calendar["event_name_1"].notna()
        | calendar["event_name_2"].notna()
    ).astype(int)

    calendar["is_sporting_event"] = (
        (calendar["event_type_1"] == "Sporting")
        | (calendar["event_type_2"] == "Sporting")
    ).astype(int)

    calendar["is_national_event"] = (
        (calendar["event_type_1"] == "National")
        | (calendar["event_type_2"] == "National")
    ).astype(int)

    calendar["is_religious_event"] = (
        (calendar["event_type_1"] == "Religious")
        | (calendar["event_type_2"] == "Religious")
    ).astype(int)

    calendar["is_cultural_event"] = (
        (calendar["event_type_1"] == "Cultural")
        | (calendar["event_type_2"] == "Cultural")
    ).astype(int)

    calendar["snap_CA"] = (
        calendar["snap_CA"].fillna(0).astype(int)
    )

    features = calendar[
        [
            "date",
            "has_event",
            "is_sporting_event",
            "is_national_event",
            "is_religious_event",
            "is_cultural_event",
            "snap_CA",
        ]
    ].copy()

    if features["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates in calendar features."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== CALENDAR FEATURES ===")
    print(features.head())
    print(f"\nRows: {len(features):,}")
    print(f"Columns: {len(features.columns)}")
    print(f"Missing values: {features.isna().sum().sum()}")

    print(f"\nSaved: {OUTPUT_PATH}")

    return features


if __name__ == "__main__":
    prepare_calendar_features()
