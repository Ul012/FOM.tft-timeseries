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

# Farbschema
COLORS = {
    'background': '#0f172a',  # Dunkler Navy-Hintergrund
    'card': '#1e293b',  # Karten-Container
    'history': '#38bdf8',  # Cyan-Blau für Historie
    'forecast': '#fb923c',  # Warmes Orange für Forecast
    'prediction': '#4ade80',  # Helles Grün für Prognose
    'text': '#e2e8f0',  # Heller Text
    'grid': '#334155'  # Grid-Linien
}


def load_raw_data(dataset_config: dict) -> pd.DataFrame:
    """Lädt Rohdaten basierend auf Dataset-Config."""
    dataset_name = dataset_config.get("name")
    if not dataset_name:
        raise ValueError("Dataset-Config muss 'name' enthalten.")

    data_path = BASE_DIR / "data" / "raw" / dataset_name / "train.csv"

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

    # Farbpalette für verschiedene Kategorien
    n_categories = daily_agg[dimension].nunique()
    palette = [COLORS['history'], COLORS['forecast'], COLORS['prediction']]
    if n_categories > 3:
        # Erweitere Palette mit Variationen
        palette = palette + sns.color_palette("husl", n_categories - 3)

    fig, ax = plt.subplots(figsize=(18, 6))

    # Setze Hintergrundfarben
    fig.patch.set_facecolor(COLORS['background'])
    ax.set_facecolor(COLORS['card'])

    # Plot mit Custom-Farben
    sns.lineplot(
        data=daily_agg,
        x=time_col,
        y=target_col,
        hue=dimension,
        ax=ax,
        palette=palette[:n_categories],
        linewidth=1
    )

    # Styling
    ax.set_title(
        f"Daily {target_col} by {dimension} ({dataset_name})",
        fontsize=16,
        color=COLORS['text'],
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel("Time", fontsize=12, color=COLORS['text'])
    ax.set_ylabel(target_col, fontsize=12, color=COLORS['text'])

    # Achsen und Grid
    ax.tick_params(colors=COLORS['text'], labelsize=10)
    ax.grid(True, alpha=0.2, color=COLORS['grid'], linestyle='--', linewidth=0.5)
    ax.spines['bottom'].set_color(COLORS['grid'])
    ax.spines['top'].set_color(COLORS['grid'])
    ax.spines['left'].set_color(COLORS['grid'])
    ax.spines['right'].set_color(COLORS['grid'])

    # Legende
    legend = ax.legend(
        frameon=True,
        facecolor=COLORS['background'],
        edgecolor=COLORS['grid'],
        labelcolor=COLORS['text'],
        fontsize=10
    )
    legend.get_frame().set_alpha(0.9)

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

    # Nur Germany filtern
    if "country" in df.columns:
        df = df[df["country"] == "Germany"].copy()
        print(f"✓ Gefiltert auf Germany: {len(df)} Zeilen")
    else:
        print("⚠ Spalte 'country' nicht gefunden - kein Filter angewendet")

    # Output-Verzeichnis
    output_dir = BASE_DIR / "results" / "plots" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot für jede ID-Dimension
    for dimension in id_cols:
        fig = plot_by_dimension(df, time_col, dimension, target_col, dataset_name)

        output_filename = f"{dataset_name}_raw_by_{dimension}_germany.png"
        output_path = output_dir / output_filename

        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=COLORS['background'])
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

# python -m src.visualization.view_data_raw_single-group --dataset configs/datasets/booksales.yaml
