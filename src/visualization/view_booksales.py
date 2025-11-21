# src/visualization/view_data_raw.py
"""
Visualisiert tägliche Verkaufszahlen pro Produkt/Store/Land (Book Sales Dataset).
Erstellt 3 separate Plots.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import RAW_DIR, BASE_DIR
from src.utils.load_dataset_config import load_dataset_config

_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]

def load_raw_data() -> pd.DataFrame:
    data_path = BASE_DIR / "data" / "raw" / _dataset_name / "train.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Rohdaten nicht gefunden: {data_path}\n"
            "Bitte train.csv im korrekten Verzeichnis ablegen."
        )

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def plot_sales_by_product(df: pd.DataFrame) -> plt.Figure:
    """Plot: Verkäufe pro Produkt."""
    daily_product = df.groupby(["date", "product"], as_index=False)["num_sold"].sum()

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.lineplot(x="date", y="num_sold", hue="product", data=daily_product, ax=ax)
    ax.set_title("Daily total sales per product")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (num_sold)")
    fig.tight_layout()
    return fig


def plot_sales_by_store(df: pd.DataFrame) -> plt.Figure:
    """Plot: Verkäufe pro Store."""
    daily_store = df.groupby(["date", "store"], as_index=False)["num_sold"].sum()

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.lineplot(x="date", y="num_sold", hue="store", data=daily_store, ax=ax)
    ax.set_title("Daily total sales per store")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (num_sold)")
    fig.tight_layout()
    return fig


def plot_sales_by_country(df: pd.DataFrame) -> plt.Figure:
    """Plot: Verkäufe pro Land."""
    daily_country = df.groupby(["date", "country"], as_index=False)["num_sold"].sum()

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.lineplot(x="date", y="num_sold", hue="country", data=daily_country, ax=ax)
    ax.set_title("Daily total sales per country")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales (num_sold)")
    fig.tight_layout()
    return fig


def main() -> None:
    """Erstellt alle 3 Plots und speichert sie."""
    df = load_raw_data()

    output_dir = BASE_DIR / "results" / "tft" / "plots" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        ("raw_by_product.png", plot_sales_by_product),
        ("raw_by_store.png", plot_sales_by_store),
        ("raw_by_country.png", plot_sales_by_country),
    ]

    for filename, plot_func in plots:
        fig = plot_func(df)
        output_path = output_dir / filename

        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot gespeichert: {output_path}")

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()

# Aufruf:
#   python -m src.visualization.view_booksales
#
# Output:
#   results/tft/plots/data/raw_by_product.png
#   results/tft/plots/data/raw_by_store.png
#   results/tft/plots/data/raw_by_country.png