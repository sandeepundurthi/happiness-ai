import pandas as pd
from pathlib import Path

INPUT_PATH = Path("data/processed/happiness_raw_loaded.csv")
OUTPUT_PATH = Path("data/processed/happiness_clean.csv")


def clean_data():
    df = pd.read_csv(INPUT_PATH)

    print("Original Shape:", df.shape)

    # Remove duplicates
    df = df.drop_duplicates()

    # Standardize country names
    df["country"] = df["country"].str.strip()

    # Sort values
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    # Missing value report
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Fill predictor missing values by country median
    cols_to_fill = [
        "explained_log_gdp_per_capita",
        "explained_social_support",
        "explained_healthy_life_expectancy",
        "explained_freedom",
        "explained_generosity",
        "explained_corruption",
        "dystopia_plus_residual"
    ]

    for col in cols_to_fill:
        df[col] = df.groupby("country")[col].transform(
            lambda x: x.fillna(x.median())
        )

    # Fill remaining with global median
    df[cols_to_fill] = df[cols_to_fill].fillna(df[cols_to_fill].median())

    # Save cleaned file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nCleaned Shape:", df.shape)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    clean_data()
