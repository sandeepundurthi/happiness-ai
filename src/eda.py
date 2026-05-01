import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")

INPUT_PATH = Path("data/processed/happiness_clean.csv")
REPORT_PATH = Path("reports")
REPORT_PATH.mkdir(exist_ok=True)


def run_eda():
    df = pd.read_csv(INPUT_PATH)

    print("Loaded:", df.shape)

    # ---------------------------------------------------
    # 1 Global average happiness by year
    # ---------------------------------------------------
    yearly = df.groupby("year")["happiness_score"].mean()

    plt.figure(figsize=(12,6))
    yearly.plot(marker="o")
    plt.title("Global Average Happiness Score Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average Happiness Score")
    plt.tight_layout()
    plt.savefig("reports/global_happiness_trend.png")
    plt.show()

    # ---------------------------------------------------
    # 2 Top gainers 2011 vs latest
    # ---------------------------------------------------
    first_year = df["year"].min()
    last_year = df["year"].max()

    start = df[df["year"] == first_year][["country","happiness_score"]]
    end = df[df["year"] == last_year][["country","happiness_score"]]

    merged = start.merge(end, on="country", suffixes=("_start","_end"))
    merged["change"] = merged["happiness_score_end"] - merged["happiness_score_start"]

    print("\nTop 10 Gainers:")
    print(merged.sort_values("change", ascending=False)[["country","change"]].head(10))

    print("\nTop 10 Decliners:")
    print(merged.sort_values("change")[["country","change"]].head(10))

    # ---------------------------------------------------
    # 3 Latest Top 15 Happiest Countries
    # ---------------------------------------------------
    latest = df[df["year"] == last_year].sort_values("happiness_score", ascending=False).head(15)

    plt.figure(figsize=(10,7))
    sns.barplot(data=latest, y="country", x="happiness_score")
    plt.title(f"Top 15 Happiest Countries ({last_year})")
    plt.tight_layout()
    plt.savefig("reports/top15_latest.png")
    plt.show()

    # ---------------------------------------------------
    # 4 USA Trend
    # ---------------------------------------------------
    usa = df[df["country"].str.contains("United States", case=False, na=False)]

    if not usa.empty:
        plt.figure(figsize=(10,5))
        sns.lineplot(data=usa, x="year", y="happiness_score", marker="o")
        plt.title("United States Happiness Trend")
        plt.tight_layout()
        plt.savefig("reports/us_trend.png")
        plt.show()


if __name__ == "__main__":
    run_eda()
