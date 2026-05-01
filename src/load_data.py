import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/world_happiness_2005_2025.csv")
PROCESSED_PATH = Path("data/processed/happiness_raw_loaded.csv")


def load_data():
    df = pd.read_csv(RAW_PATH)
    print("Dataset loaded successfully")
    print("-" * 40)
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nPreview:")
    print(df.head())

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print(f"\nSaved loaded dataset to: {PROCESSED_PATH}")


if __name__ == "__main__":
    load_data()
