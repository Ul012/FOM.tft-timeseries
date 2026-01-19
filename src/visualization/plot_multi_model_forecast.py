# src/visualization/plot_multi_model_forecast.py
"""
Multi-Model Forecast Visualization

Erstellt Forecast-Plots die alle Modelle (TFT, Prophet, ARIMA) vergleichen.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import BASE_DIR


def load_tft_predictions(run_id: str) -> Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Lädt TFT Predictions aus predictions.npz oder predictions_test.npy
    """
    pred_dir = BASE_DIR / "results" / "tft" / "runs" / run_id / "predictions"

    if not pred_dir.exists():
        return None

    # Versuche predictions.npz (combined)
    npz_path = pred_dir / "predictions.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        return data['actuals'], data['predictions'], data.get('timestamps', None)

    # Versuche predictions_test.npy
    test_pred = pred_dir / "predictions_test.npy"
    test_actual = pred_dir / "actuals_test.npy"

    if test_pred.exists() and test_actual.exists():
        predictions = np.load(test_pred)
        actuals = np.load(test_actual)
        return actuals, predictions, None

    return None


def load_prophet_predictions(run_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Lädt Prophet Predictions aus predictions/predictions_test.npy
    """
    pred_dir = BASE_DIR / "results" / "prophet" / "runs" / run_id / "predictions"

    if not pred_dir.exists():
        return None

    pred_path = pred_dir / "predictions_test.npy"
    actual_path = pred_dir / "actuals_test.npy"

    if pred_path.exists() and actual_path.exists():
        predictions = np.load(pred_path)
        actuals = np.load(actual_path)
        return actuals, predictions

    return None


def load_arima_predictions(run_id: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Lädt ARIMA Predictions aus predictions/predictions_test.npy
    """
    pred_dir = BASE_DIR / "results" / "arima" / "runs" / run_id / "predictions"

    if not pred_dir.exists():
        return None

    pred_path = pred_dir / "predictions_test.npy"
    actual_path = pred_dir / "actuals_test.npy"

    if pred_path.exists() and actual_path.exists():
        predictions = np.load(pred_path)
        actuals = np.load(actual_path)
        return actuals, predictions

    return None


def load_and_normalize_predictions(
        best_runs: Dict[str, pd.Series],
        max_points: Optional[int] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Zentrale Funktion: Lädt alle Predictions und normalisiert sie.

    Args:
        best_runs: Dict mit Modell -> Run-Info
        max_points: Optional - Downsampling auf max Punkte

    Returns:
        Dict mit Modell -> (actuals, predictions) - alle gleiche Länge!
    """
    predictions_data = {}

    # 1. Lade Predictions
    for model, row in best_runs.items():
        run_id = row['run_id']

        try:
            if model == 'TFT':
                result = load_tft_predictions(run_id)
                if result:
                    actuals, predictions, _ = result
                    predictions_data[model] = (actuals, predictions)
            elif model == 'Prophet':
                result = load_prophet_predictions(run_id)
                if result:
                    actuals, predictions = result
                    predictions_data[model] = (actuals, predictions)
            elif model == 'ARIMA':
                result = load_arima_predictions(run_id)
                if result:
                    actuals, predictions = result
                    predictions_data[model] = (actuals, predictions)
        except Exception as e:
            print(f"[WARNING] {model} predictions nicht geladen: {e}")

    if not predictions_data:
        return {}

    # 2. Flatten alle Arrays
    for model in predictions_data:
        actuals, predictions = predictions_data[model]

        if actuals.ndim > 1:
            actuals = actuals.flatten()
        if predictions.ndim > 1:
            predictions = predictions.flatten()

        predictions_data[model] = (actuals, predictions)

    # 3. Finde minimale Länge über alle Modelle
    min_length = min(len(preds[1]) for preds in predictions_data.values())

    # 4. Kürze alle auf minimale Länge
    for model in predictions_data:
        actuals, predictions = predictions_data[model]
        predictions_data[model] = (actuals[:min_length], predictions[:min_length])

    # 5. Optional: Downsampling für bessere Performance
    if max_points and min_length > max_points:
        step = min_length // max_points
        for model in predictions_data:
            actuals, predictions = predictions_data[model]
            predictions_data[model] = (actuals[::step], predictions[::step])

    return predictions_data


def plot_metric_comparison_fallback(
        dataset: str,
        best_runs: Dict[str, pd.Series],
        output_dir: Path
) -> None:
    """
    Zeigt Metrik-Vergleich zwischen allen Modellen.
    Gruppiert nach Metriken (jede Metrik hat 3 Balken für TFT, Prophet, ARIMA).
    Wird immer erstellt, unabhängig davon ob Predictions vorhanden sind.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    models = list(best_runs.keys())
    metrics = ['test_mae', 'test_rmse', 'test_smape']
    metric_labels = ['MAE', 'RMSE', 'SMAPE (%)']

    # Farben für Modelle (nicht Metriken!)
    model_colors = {
        'TFT': '#3498db',
        'Prophet': '#9b59b6',
        'ARIMA': '#e67e22'
    }

    x = np.arange(len(metrics))  # X-Achse = Metriken
    width = 0.25

    # Iteriere über Modelle, jedes Modell bekommt einen Balken pro Metrik
    for i, model in enumerate(models):
        values = [best_runs[model][metric] for metric in metrics]
        ax.bar(x + i * width, values, width,
               label=model,
               color=model_colors.get(model, '#95a5a6'),
               alpha=0.8)

    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Performance Comparison: {dataset}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='best', fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path = output_dir / f"multi_model_comparison_{dataset.lower()}_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[plot_multi_model_forecast] Saved metric comparison: {output_path}")


def plot_multi_model_forecast(
        dataset: str,
        best_runs: Dict[str, pd.Series],
        output_dir: Path,
        max_points: int = 500
) -> None:
    """
    Erstellt Line Plot mit Forecast-Vergleich.
    """
    # Lade und normalisiere Predictions (DRY!)
    predictions_data = load_and_normalize_predictions(best_runs, max_points)

    if not predictions_data:
        return

    # Setup Plot
    fig, ax = plt.subplots(figsize=(16, 8))

    colors = {
        'Actual': '#2c3e50',
        'TFT': '#3498db',
        'Prophet': '#9b59b6',
        'ARIMA': '#e67e22'
    }

    # Plot Actual (von erstem Modell)
    first_model = list(predictions_data.keys())[0]
    actuals, _ = predictions_data[first_model]
    x = np.arange(len(actuals))

    ax.plot(x, actuals,
            label='Actual',
            color=colors['Actual'],
            linewidth=2.5,
            alpha=0.8,
            zorder=10)

    # Plot Predictions für jedes Modell
    for model, (_, predictions) in predictions_data.items():
        smape = best_runs[model]['test_smape']
        mae = best_runs[model]['test_mae']

        ax.plot(x, predictions,
                label=f'{model} (SMAPE: {smape:.1f}%, MAE: {mae:.1f})',
                color=colors[model],
                linewidth=2,
                alpha=0.7,
                linestyle='--' if model != 'TFT' else '-')

    ax.set_xlabel('Time Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Multi-Model Forecast Comparison: {dataset}',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path = output_dir / f"multi_model_forecast_{dataset.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[plot_multi_model_forecast] Saved: {output_path}")


def plot_multi_model_forecast_aggregated(
        dataset: str,
        best_runs: Dict[str, pd.Series],
        output_dir: Path,
        n_samples: int = 100
) -> None:
    """
    Erstellt Scatter Plot mit gesampelten Punkten.
    """
    # Lade und normalisiere Predictions (DRY!)
    predictions_data = load_and_normalize_predictions(best_runs)

    if not predictions_data:
        return

    # Setup Plot
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {
        'Actual': '#2c3e50',
        'TFT': '#3498db',
        'Prophet': '#9b59b6',
        'ARIMA': '#e67e22'
    }

    # Sample Points
    first_model = list(predictions_data.keys())[0]
    actuals, _ = predictions_data[first_model]
    total_points = len(actuals)

    if total_points > n_samples:
        idx = np.random.choice(total_points, n_samples, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(total_points)

    x = np.arange(len(idx))

    # Plot sampled actuals
    sampled_actuals = actuals[idx]
    ax.scatter(x, sampled_actuals,
               label='Actual (sampled)',
               color=colors['Actual'],
               alpha=0.6, s=30, zorder=10)

    # Plot sampled predictions
    for model, (_, predictions) in predictions_data.items():
        sampled_preds = predictions[idx]
        smape = best_runs[model]['test_smape']

        ax.plot(x, sampled_preds,
                label=f'{model} (SMAPE: {smape:.1f}%)',
                color=colors[model],
                linewidth=2,
                alpha=0.7)

    ax.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Multi-Model Forecast Comparison: {dataset} (Sampled Points)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path = output_dir / f"multi_model_forecast_{dataset.lower()}_sampled.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_best_run_per_model(df: pd.DataFrame, dataset: str) -> Dict[str, pd.Series]:
    """
    Findet besten Run pro Modell für Dataset.
    """
    df_filtered = df[
        (df['dataset'] == dataset) &
        (df['type'].isin(['Baseline', 'Optuna']))
        ]

    if df_filtered.empty:
        return {}

    best_runs = {}
    for model in ['TFT', 'Prophet', 'ARIMA']:
        model_runs = df_filtered[df_filtered['model'] == model]
        if not model_runs.empty:
            best_idx = model_runs['test_smape'].idxmin()
            best_runs[model] = model_runs.loc[best_idx]

    return best_runs


def main() -> None:
    """Hauptfunktion."""
    print("[plot_multi_model_forecast] Erstelle Multi-Model Forecast Plots...")

    # Lade model_comparison.csv
    comparison_file = BASE_DIR / "results" / "eval" / "model_comparison.csv"

    if not comparison_file.exists():
        print(f"[ERROR] {comparison_file} nicht gefunden!")
        print("Führe zuerst aggregate_all_models_eval.py aus.")
        return

    df = pd.read_csv(comparison_file)
    output_dir = BASE_DIR / "results" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Für jeden Dataset
    for dataset in ['Booksales', 'Walmart']:
        print(f"Processing {dataset}...")

        best_runs = get_best_run_per_model(df, dataset)

        if not best_runs:
            print(f"  ⚠ Keine Runs für {dataset}")
            continue

        # DEBUG: Zeige welche Modelle gefunden wurden
        print(f"  [DEBUG] Best Runs gefunden: {list(best_runs.keys())}")
        for model, row in best_runs.items():
            print(f"    {model}: {row['run_id']}")

        # Check ob Predictions verfügbar
        predictions_data = load_and_normalize_predictions(best_runs)
        print(f"  [DEBUG] Predictions geladen: {list(predictions_data.keys())}")

        # DEBUG: Zeige min/max Werte
        for model, (actuals, predictions) in predictions_data.items():
            print(f"    {model}: actuals=[{actuals.min():.1f}, {actuals.max():.1f}], "
                  f"predictions=[{predictions.min():.1f}, {predictions.max():.1f}]")

        # Metrik-Vergleich IMMER erstellen
        print("  ✓ Erstelle Metrik-Vergleich")
        plot_metric_comparison_fallback(dataset, best_runs, output_dir)

        # Forecast-Plots nur wenn Predictions vorhanden
        if predictions_data:
            print("  ✓ Predictions gefunden - erstelle Forecast-Plots")
            plot_multi_model_forecast(dataset, best_runs, output_dir, max_points=500)
            plot_multi_model_forecast_aggregated(dataset, best_runs, output_dir, n_samples=100)
        else:
            print("  ℹ Keine Predictions gefunden - nur Metrik-Vergleich erstellt")

    print("=" * 80)
    print(f"✅ Multi-Model Plots erstellt in: {output_dir}")
    print("=" * 80)
    print("HINWEIS:")
    print("Für bessere Forecast-Visualisierungen können Predictions gespeichert werden:")
    print("  python -m src.evaluation.evaluate_tft --run-id <run_id> --split test --save-predictions")
    print("  python -m src.evaluation.evaluate_prophet --run-id <run_id> --split test --save-predictions")
    print("  python -m src.evaluation.evaluate_arima --run-id <run_id> --split test --save-predictions")


if __name__ == "__main__":
    main()

# Aufruf:
#   python -m src.visualization.plot_multi_model_forecast
#
# Output (immer):
#   - results/plots/multi_model_comparison_booksales_metrics.png
#   - results/plots/multi_model_comparison_walmart_metrics.png
#
# Output (zusätzlich, wenn Predictions vorhanden):
#   - results/plots/multi_model_forecast_booksales.png
#   - results/plots/multi_model_forecast_walmart.png
#   - results/plots/multi_model_forecast_booksales_sampled.png
#   - results/plots/multi_model_forecast_walmart_sampled.png