# src/visualization/data_cleaning_plot.py
# Zweck: Visuelle Prüfung der bereinigten (cleaned) Verkaufsdaten
# Quelle: data/interim/train_cleaned.parquet

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_cleaned_sales(df: pd.DataFrame) -> None:
    """Erstellt einen Liniendiagramm-Plot der bereinigten Verkaufszahlen (täglich, je Land)."""
    # Tagesweise Aggregation je Land
    daily_country = (
        df.groupby(["date", "country"], as_index=False)["num_sold"].sum()
    )

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(18, 6))

    sns.lineplot(
        data=daily_country,
        x="date",
        y="num_sold",
        hue="country",
        ax=ax
    )

    ax.set_title("Summe der Verkäufe pro Land (bereinigt)", fontsize=14)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Verkäufe (num_sold, bereinigt)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Lädt die bereinigte Parquet-Datei und erstellt den Plot."""
    base_dir = Path(__file__).resolve().parents[2]
    parquet_path = base_dir / "data" / "interim" / "train_cleaned.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet-Datei nicht gefunden: {parquet_path}\n"
            "Bitte zuerst data_alignment.py und anschließend data_cleaning.py ausführen."
        )

    df = pd.read_parquet(parquet_path)

    # Defensive: Datums-Typ sicherstellen
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    plot_cleaned_sales(df)


if __name__ == "__main__":
    main()
