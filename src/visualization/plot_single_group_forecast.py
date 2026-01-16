# src/visualization/plot_single_group_forecast.py
"""
Single-Group Forecast Visualization

Zeigt Forecast-Vergleich für EINE repräsentative Gruppe pro Modell.
Perfekt für Seminararbeit - saubere, durchgängige Linien ohne Lücken.

Aufruf:
    python -m src.visualization.plot_single_group_forecast

Output:
    - results/plots/single_group_forecast_booksales_highest_mean.png
    - results/plots/single_group_forecast_walmart_highest_mean.png

Hinweis:
    Die Gruppe wird nach "Highest Average Sales" ausgewählt.
    Dies ist wissenschaftlich gut argumentierbar:
    "Zeigt typisches Verhalten eines erfolgreichen Stores"
"""

from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import BASE_DIR

# Dataset-Konfiguration
DATASET_CONFIG = {
    'Booksales': {
        'n_groups': 48,  # 48 Produkte
        'forecast_length': 30,  # 30 Tage
        'x_label': 'Days'
    },
    'Walmart': {
        'n_groups': 48,  # 48 Stores
        'forecast_length': 4,  # 4 Wochen
        'x_label': 'Weeks'
    }
}


def load_predictions(model: str, run_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Lädt Test-Predictions für ein Modell.

    Args:
        model: Modell-Name (TFT, Prophet, ARIMA)
        run_id: Run-ID

    Returns:
        (actuals, predictions) oder None
    """
    pred_dir = BASE_DIR / "results" / model.lower() / "runs" / run_id / "predictions"

    if not pred_dir.exists():
        return None

    pred_path = pred_dir / "predictions_test.npy"
    actual_path = pred_dir / "actuals_test.npy"

    if pred_path.exists() and actual_path.exists():
        predictions = np.load(pred_path)
        actuals = np.load(actual_path)

        # Flatten falls multidimensional
        if actuals.ndim > 1:
            actuals = actuals.flatten()
        if predictions.ndim > 1:
            predictions = predictions.flatten()

        return actuals, predictions

    return None


def select_best_group(
        actuals: np.ndarray,
        predictions: np.ndarray,
        n_groups: int,
        forecast_length: int
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    Wählt Gruppe mit höchstem durchschnittlichen Sales.

    Args:
        actuals: Flattened actuals (n_groups * forecast_length,)
        predictions: Flattened predictions (n_groups * forecast_length,)
        n_groups: Anzahl Gruppen
        forecast_length: Länge pro Gruppe

    Returns:
        (actuals_group, predictions_group, group_idx, group_mean)
    """
    total_length = len(actuals)
    expected_length = n_groups * forecast_length

    # Kürze auf erwartete Länge falls nötig
    if total_length > expected_length:
        actuals = actuals[:expected_length]
        predictions = predictions[:expected_length]
        total_length = expected_length

    # Prüfe ob Länge passt
    if total_length != expected_length:
        # Fallback: Nimm erste forecast_length Punkte
        return actuals[:forecast_length], predictions[:forecast_length], 0, actuals[:forecast_length].mean()

    # Reshape zu (n_groups, forecast_length)
    actuals_reshaped = actuals.reshape(n_groups, forecast_length)
    predictions_reshaped = predictions.reshape(n_groups, forecast_length)

    # Finde Gruppe mit höchstem Durchschnitt
    group_means = actuals_reshaped.mean(axis=1)
    best_idx = group_means.argmax()
    best_mean = group_means[best_idx]

    return actuals_reshaped[best_idx], predictions_reshaped[best_idx], best_idx, best_mean


def plot_single_group(
        dataset: str,
        predictions_data: Dict[str, Tuple[np.ndarray, np.ndarray, int, float]],
        metrics: Dict[str, Dict[str, float]],
        output_dir: Path
) -> None:
    """
    Erstellt Plot für eine repräsentative Gruppe.

    Args:
        dataset: Dataset-Name (Booksales oder Walmart)
        predictions_data: Dict[model] -> (actuals, predictions, group_idx, group_mean)
        metrics: Dict[model] -> {test_smape, test_mae}
        output_dir: Output-Verzeichnis
    """
    config = DATASET_CONFIG[dataset]

    # Setup Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {
        'Actual': '#2c3e50',
        'TFT': '#3498db',
        'Prophet': '#9b59b6',
        'ARIMA': '#e67e22'
    }

    # Plot Actual (von erstem Modell)
    first_model = list(predictions_data.keys())[0]
    actuals, _, group_idx, group_mean = predictions_data[first_model]
    x = np.arange(len(actuals))

    ax.plot(x, actuals,
            label='Actual',
            color=colors['Actual'],
            linewidth=2.5,
            alpha=0.8,
            zorder=10)

    # Plot Predictions für jedes Modell
    for model, (_, predictions, _, _) in predictions_data.items():
        smape = metrics[model]['test_smape']
        mae = metrics[model]['test_mae']

        # Clip negative predictions (Sales können nicht negativ sein)
        predictions = np.clip(predictions, 0, None)

        ax.plot(x, predictions,
                label=f'{model} (SMAPE: {smape:.1f}%, MAE: {mae:.1f})',
                color=colors[model],
                linewidth=2,
                alpha=0.7,
                linestyle='--' if model != 'TFT' else '-')

    # Achsen und Titel
    ax.set_xlabel(config['x_label'], fontsize=12, fontweight='bold')
    ax.set_ylabel('Sales', fontsize=12, fontweight='bold')
    ax.set_title(f'Multi-Model Forecast Comparison: {dataset}\n'
                 f'Selected Group: #{group_idx} (Highest Average Sales: {group_mean:.1f})',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(bottom=0)  # Y-Achse startet bei 0
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Save
    output_path = output_dir / f"single_group_forecast_{dataset.lower()}_highest_mean.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[single_group_forecast] Saved: {output_path}")
    print(
        f"  Group #{group_idx}, Avg Sales: {group_mean:.1f}, Forecast Length: {len(actuals)} {config['x_label'].lower()}")


def get_best_runs(dataset: str) -> Dict[str, pd.Series]:
    """
    Lädt beste Runs pro Modell aus model_comparison.csv.

    Args:
        dataset: Dataset-Name

    Returns:
        Dict[model] -> Run-Info (als pandas Series)
    """
    comparison_file = BASE_DIR / "results" / "eval" / "model_comparison.csv"

    if not comparison_file.exists():
        print(f"[ERROR] {comparison_file} nicht gefunden!")
        print("Führe zuerst aggregate_all_models_eval.py aus.")
        return {}

    df = pd.read_csv(comparison_file)

    # Filter für Dataset und Baseline/Optuna
    df_filtered = df[
        (df['dataset'] == dataset) &
        (df['type'].isin(['Baseline', 'Optuna']))
        ]

    if df_filtered.empty:
        return {}

    # Bester Run pro Modell (niedrigste SMAPE)
    best_runs = {}
    for model in ['TFT', 'Prophet', 'ARIMA']:
        model_runs = df_filtered[df_filtered['model'] == model]
        if not model_runs.empty:
            best_idx = model_runs['test_smape'].idxmin()
            best_runs[model] = model_runs.loc[best_idx]

    return best_runs


def main() -> None:
    """Hauptfunktion."""
    print("[single_group_forecast] Erstelle Single-Group Forecast Plots...")
    print()

    output_dir = BASE_DIR / "results" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Für jeden Dataset
    for dataset in ['Booksales', 'Walmart']:
        print(f"Processing {dataset}...")

        # Lade beste Runs
        best_runs = get_best_runs(dataset)

        if not best_runs:
            print(f"  ⚠ Keine Runs für {dataset} gefunden")
            continue

        print(f"  Best Runs: {list(best_runs.keys())}")

        # Lade Predictions
        config = DATASET_CONFIG[dataset]
        predictions_data = {}
        metrics = {}

        # SCHRITT 1: Lade alle Predictions
        all_predictions = {}
        for model, run_info in best_runs.items():
            run_id = run_info['run_id']

            result = load_predictions(model, run_id)
            if result is None:
                print(f"  ⚠ {model}: Keine Predictions gefunden (run: {run_id})")
                continue

            all_predictions[model] = result
            metrics[model] = {
                'test_smape': run_info['test_smape'],
                'test_mae': run_info['test_mae']
            }

        if not all_predictions:
            print(f"  ⚠ Keine Predictions für {dataset}")
            continue

        # SCHRITT 2: Wähle gemeinsame Gruppe basierend auf ERSTEM Modell
        first_model = list(all_predictions.keys())[0]
        actuals_first, predictions_first = all_predictions[first_model]

        _, _, best_group_idx, best_group_mean = select_best_group(
            actuals_first, predictions_first,
            config['n_groups'],
            config['forecast_length']
        )

        print(
            f"  Gemeinsame Gruppe gewählt: #{best_group_idx} (Avg: {best_group_mean:.1f}) basierend auf {first_model}")

        # SCHRITT 3: Extrahiere DIESE Gruppe für alle Modelle
        for model, (actuals, predictions) in all_predictions.items():
            total_length = len(actuals)
            expected_length = config['n_groups'] * config['forecast_length']

            # Kürze auf erwartete Länge falls nötig
            if total_length > expected_length:
                actuals = actuals[:expected_length]
                predictions = predictions[:expected_length]
                total_length = expected_length

            # Prüfe ob Länge passt
            if total_length != expected_length:
                print(f"  ⚠ {model}: Länge passt nicht ({total_length} != {expected_length}), überspringe")
                continue

            # Reshape und extrahiere die gemeinsame Gruppe
            actuals_reshaped = actuals.reshape(config['n_groups'], config['forecast_length'])
            predictions_reshaped = predictions.reshape(config['n_groups'], config['forecast_length'])

            actuals_group = actuals_reshaped[best_group_idx]
            predictions_group = predictions_reshaped[best_group_idx]

            predictions_data[model] = (actuals_group, predictions_group, best_group_idx, best_group_mean)

            print(f"    {model}: Group #{best_group_idx}")

        if not predictions_data:
            print(f"  ⚠ Keine Predictions für {dataset}")
            continue

        # Erstelle Plot
        plot_single_group(dataset, predictions_data, metrics, output_dir)
        print()

    print("=" * 80)
    print(f"✅ Single-Group Plots erstellt in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Aufruf:
#   python -m src.visualization.plot_single_group_forecast
#
# Output:
#   - results/plots/single_group_forecast_booksales_highest_mean.png
#   - results/plots/single_group_forecast_walmart_highest_mean.png
#
# Hinweis:
#   Zeigt EINE repräsentative Gruppe (Highest Average Sales) pro Dataset.
#   Perfekt für Seminararbeit - saubere, durchgängige Linien ohne Lücken.
#
#   Booksales: 30 Tage Forecast, 48 Gruppen
#   Walmart: 4 Wochen Forecast, 48 Stores
#
# Voraussetzung:
#   - model_comparison.csv muss existieren (aggregate_all_models_eval.py)
#   - Predictions müssen mit --save-predictions erstellt worden sein