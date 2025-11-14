# src/visualization/data_alignment_plot.py
# Zweck: Darstellung der angeglichenen Verkaufszahlen (auf 2020-Niveau)
# Quelle: data/interim/train_aligned.parquet

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_aligned_sales(df: pd.DataFrame) -> None:
    """Erstellt einen Liniendiagramm-Plot der angeglichenen Verkaufszahlen."""
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

    ax.set_title("Summe der Verkäufe pro Land (auf 2020-Niveau skaliert)", fontsize=14)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Verkäufe (num_sold, skaliert)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Lädt die angeglichene Parquet-Datei und erstellt den Plot."""
    base_dir = Path(__file__).resolve().parents[2]
    parquet_path = base_dir / "data" / "interim" / "train_aligned.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet-Datei nicht gefunden: {parquet_path}\n"
            "Bitte zuerst data_alignment.py ausführen."
        )

    df = pd.read_parquet(parquet_path)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    plot_aligned_sales(df)


if __name__ == "__main__":
    main()

# python -m src.visualization.data_alignment_plot
