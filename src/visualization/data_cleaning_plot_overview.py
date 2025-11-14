# src/visualization/data_cleaning_plot_overview.py
# Zweck: Visuelle Prüfung des Cleaning-Schritts
# Vergleich: train_aligned (vorher) vs. train_cleaned (nachher) für das Jahr 2020

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import INTERIM_DIR


def _prepare_daily_country(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregiert tägliche Verkaufszahlen je Land und filtert auf 2020."""
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df_2020 = df[df["date"].dt.year == 2020].copy()

    daily_country = (
        df_2020.groupby(["date", "country"], as_index=False)["num_sold"].sum()
    )
    return daily_country


def plot_cleaning_comparison(df_aligned: pd.DataFrame, df_cleaned: pd.DataFrame) -> None:
    """Erstellt eine Vergleichsgrafik: vor/nach Cleaning für 2020."""

    sns.set(style="whitegrid")

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(18, 8), sharex=True)

    # Vor Cleaning (train_aligned)
    sns.lineplot(
        data=df_aligned,
        x="date",
        y="num_sold",
        hue="country",
        ax=axes[0],
    )
    axes[0].set_title("Summe der Verkäufe pro Land – vor Cleaning (train_aligned)", fontsize=13)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Verkäufe (num_sold)")

    # Nach Cleaning (train_cleaned)
    sns.lineplot(
        data=df_cleaned,
        x="date",
        y="num_sold",
        hue="country",
        ax=axes[1],
        legend=False,  # Legende nur oben
    )
    axes[1].set_title("Summe der Verkäufe pro Land – nach Cleaning (train_cleaned)", fontsize=13)
    axes[1].set_xlabel("Datum (nur Jahr 2020)")
    axes[1].set_ylabel("Verkäufe (num_sold, bereinigt)")

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Lädt train_aligned & train_cleaned und visualisiert den Cleaning-Schritt."""
    aligned_path = INTERIM_DIR / "train_aligned.parquet"
    cleaned_path = INTERIM_DIR / "train_cleaned.parquet"

    if not aligned_path.exists():
        raise FileNotFoundError(
            f"Parquet-Datei nicht gefunden: {aligned_path}\n"
            "Bitte zuerst data_alignment.py ausführen."
        )

    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Parquet-Datei nicht gefunden: {cleaned_path}\n"
            "Bitte zuerst data_cleaning.py ausführen."
        )

    df_aligned = pd.read_parquet(aligned_path)
    df_cleaned = pd.read_parquet(cleaned_path)

    daily_aligned = _prepare_daily_country(df_aligned)
    daily_cleaned = _prepare_daily_country(df_cleaned)

    plot_cleaning_comparison(daily_aligned, daily_cleaned)


if __name__ == "__main__":
    # python -m src.visualization.data_cleaning_plot_overview
    main()

