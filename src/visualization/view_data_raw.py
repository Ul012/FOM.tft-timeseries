# src/visualization/view_data_raw.py
"""
Visualisiert Rohdaten beliebiger Datasets.
Liest Schema aus Dataset-Config (z.B. booksales.yaml).
"""
from pathlib import Path
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import RAW_DIR, BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema


def load_raw_data(dataset_config: dict) -> pd.DataFrame:
    """Lädt Rohdaten basierend auf Dataset-Config."""
    dataset_name = dataset_config.get("name")
    if not dataset_name:
        raise ValueError("Dataset-Config muss 'name' enthalten.")

    data_path = RAW_DIR / "tabular-playground-series-sep-2022" / "train.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Rohdaten nicht gefunden: {data_path}\n"
            f"Erwartete Struktur: data/raw/tabular-playground-series-sep-2022/train.csv"
        )

    return pd.read_csv(data_path)


def prepare_timeseries(df: pd.DataFrame, time_col: str, target_col: str) -> pd.DataFrame:
    """Bereitet Zeitreihen-Daten vor."""
    # Validierung
    if time_col not in df.columns:
        raise KeyError(f"Spalte '{time_col}' nicht in Daten gefunden.")
    if target_col not in df.columns:
        raise KeyError(f"Spalte '{target_col}' nicht in Daten gefunden.")

    # Zeitstempel konvertieren
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    return df


def plot_by_dimension(
        df: pd.DataFrame,
        time_col: str,
        dimension: str,
        target_col: str,
        dataset_name: str
) -> plt.Figure:
    """Erstellt Plot für eine Dimension (z.B. product, store, country)."""
    if dimension not in df.columns:
        raise KeyError(f"Dimension '{dimension}' nicht in Daten gefunden.")

    # Tägliche Aggregation
    daily_agg = df.groupby([time_col, dimension], as_index=False)[target_col].sum()

    fig, ax = plt.subplots(figsize=(18, 6))
    sns.lineplot(
        data=daily_agg,
        x=time_col,
        y=target_col,
        hue=dimension,
        ax=ax
    )

    ax.set_title(f"Daily {target_col} by {dimension} ({dataset_name})", fontsize=14)
    ax.set_xlabel("Time")
    ax.set_ylabel(target_col)
    fig.tight_layout()
    return fig


def main(dataset_path: Optional[str] = None) -> None:
    """
    Erstellt Plots für alle ID-Dimensionen des Datasets.

    Args:
        dataset_path: Pfad zur Dataset-Config (z.B. "configs/datasets/booksales.yaml").
                      Falls None, wird Standard-Config verwendet.
    """
    # Config laden
    if dataset_path:
        import yaml
        with open(dataset_path) as f:
            dataset_config = yaml.safe_load(f)
    else:
        dataset_config = load_dataset_config()

    schema = get_schema(dataset_config)
    dataset_name = dataset_config.get("name")
    if not dataset_name:
        raise ValueError("Dataset-Config muss 'name' enthalten.")

    time_col = schema["time_col"]
    id_cols = schema["id_cols"]
    target_col = schema["target_col"]

    # Daten laden
    df = load_raw_data(dataset_config)
    df = prepare_timeseries(df, time_col, target_col)

    # Output-Verzeichnis
    output_dir = BASE_DIR / "results" / "tft" / "plots" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot für jede ID-Dimension
    for dimension in id_cols:
        fig = plot_by_dimension(df, time_col, dimension, target_col, dataset_name)

        output_filename = f"{dataset_name}_raw_by_{dimension}.png"
        output_path = output_dir / output_filename

        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot gespeichert: {output_path}")

        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualisiert Rohdaten eines Datasets (alle ID-Dimensionen)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Pfad zur Dataset-Config (z.B. configs/datasets/booksales.yaml). "
             "Standard: configs/datasets/booksales.yaml"
    )
    args = parser.parse_args()

    main(dataset_path=args.dataset)

# Aufruf:
#   python -m src.visualization.view_data_raw
#   python -m src.visualization.view_data_raw --dataset configs/datasets/booksales.yaml
#   python -m src.visualization.view_data_raw --dataset configs/datasets/other_dataset.yaml
#
# Output (Booksales):
#   results/tft/plots/data/booksales_raw_by_product.png
#   results/tft/plots/data/booksales_raw_by_store.png
#   results/tft/plots/data/booksales_raw_by_country.png