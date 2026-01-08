# src/visualization/plot_prophet_optuna_study.py
"""
Visualisiert Prophet Optuna-Studien: Optimization History, Parameter Importance, etc.

Aufruf:
    Booksales (alle Plots):
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
    python -m src.visualization.plot_prophet_optuna_study --study-name prophet_booksales

    Walmart (nur bestimmte Plots):
    $env:DATASET_CONFIG="configs/datasets/walmart.yaml"
    python -m src.visualization.plot_prophet_optuna_study --study-name prophet_walmart --plots history importance
"""

import argparse
from pathlib import Path

import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
)

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config

# ============================================================================
# KONSTANTEN
# ============================================================================

_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]

OPTUNA_BASE_DIR = BASE_DIR / "results" / "prophet" / "optuna" / _dataset_name
STORAGE_PATH = OPTUNA_BASE_DIR / "prophet_studies.db"
OPTUNA_STORAGE = f"sqlite:///{STORAGE_PATH}"
PLOT_DIR = OPTUNA_BASE_DIR / "plots"


# ============================================================================
# FUNKTIONEN
# ============================================================================


def plot_study(study_name: str, plots: list[str]):
    """
    Erstellt Visualisierungen für eine Optuna Study.

    Args:
        study_name: Name der Study
        plots: Liste der zu erstellenden Plots ['history', 'importance', 'parallel', 'slice']

    Raises:
        FileNotFoundError: Wenn Datenbank nicht existiert
        ValueError: Wenn keine abgeschlossenen Trials vorhanden
    """
    if not STORAGE_PATH.exists():
        raise FileNotFoundError(f"Optuna-Datenbank nicht gefunden: {STORAGE_PATH}")

    study = optuna.load_study(study_name=study_name, storage=OPTUNA_STORAGE)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise ValueError(f"Study '{study_name}' hat keine abgeschlossenen Trials!")

    print(f"[plot_prophet_optuna_study] Study: {study_name}")
    print(f"[plot_prophet_optuna_study] Trials: {len(study.trials)} (davon {len(completed)} abgeschlossen)")
    print(f"[plot_prophet_optuna_study] Beste val_mae: {study.best_value:.4f}")
    print()

    plot_dir = PLOT_DIR / study_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_functions = {
        "history": (plot_optimization_history, "optimization_history"),
        "importance": (plot_param_importances, "param_importances"),
        "parallel": (plot_parallel_coordinate, "parallel_coordinate"),
        "slice": (plot_slice, "slice"),
    }

    for plot_name in plots:
        if plot_name not in plot_functions:
            print(f"[WARNING] Unbekannter Plot: {plot_name}")
            continue

        plot_func, filename = plot_functions[plot_name]

        try:
            fig = plot_func(study)
            html_path = plot_dir / f"{filename}.html"
            fig.write_html(str(html_path))
            print(f"[plot_prophet_optuna_study] ✓ {html_path}")

            # PNG nur wenn kaleido verfügbar
            try:
                png_path = plot_dir / f"{filename}.png"
                fig.write_image(str(png_path))
                print(f"[plot_prophet_optuna_study] ✓ {png_path}")
            except Exception:
                print(f"[plot_prophet_optuna_study] ⚠ PNG-Export übersprungen (kaleido nicht installiert)")

        except Exception as e:
            print(f"[plot_prophet_optuna_study] ✗ Fehler bei {plot_name}: {e}")

    print()
    print(f"[plot_prophet_optuna_study] Plots gespeichert in: {plot_dir}")


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Visualisiere Prophet Optuna Study")
    parser.add_argument(
        "--study-name",
        type=str,
        required=True,
        help="Name der Optuna Study (z.B. prophet_booksales)",
    )
    parser.add_argument(
        "--plots",
        type=str,
        nargs="+",
        default=["history", "importance", "parallel", "slice"],
        choices=["history", "importance", "parallel", "slice"],
        help="Plots zum Erstellen (default: alle)",
    )

    args = parser.parse_args()

    plot_study(args.study_name, args.plots)


if __name__ == "__main__":
    main()

# Aufruf:
#   Booksales:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
#   python -m src.visualization.plot_prophet_optuna_study --study-name prophet_booksales
#
#   Walmart:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"
#   python -m src.visualization.plot_prophet_optuna_study --study-name prophet_walmart