# src/visualization/plot_multi_model_forecast.py
# Multi-Model Forecast Comparison: Actual vs. TFT vs. Prophet vs. ARIMA
#
# Zeigt tatsächliche Werte + Forecasts von allen 3 Modellen (Best Runs)
# in einem Plot für visuellen Vergleich
#
# Nutzung:
#   python -m src.visualization.plot_multi_model_forecast

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import BASE_DIR

# Plot-Konfiguration
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'


def load_model_comparison() -> pd.DataFrame:
    """Lädt model_comparison.csv"""
    path = BASE_DIR / "results" / "eval" / "model_comparison.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"model_comparison.csv nicht gefunden: {path}\n"
            f"Bitte zuerst ausführen: python -m src.evaluation.aggregate_all_models_eval"
        )

    return pd.read_csv(path)


def get_best_run_per_model(df: pd.DataFrame, dataset: str) -> Dict[str, pd.Series]:
    """
    Gibt Best Run pro Modell für ein Dataset zurück.
    """
    dataset_df = df[df['dataset'] == dataset]
    best_runs = dataset_df[dataset_df['type'].isin(['Baseline', 'Optuna'])].copy()

    result = {}
    for model in ['TFT', 'Prophet', 'ARIMA']:
        model_df = best_runs[best_runs['model'] == model]
        if len(model_df) > 0:
            best_idx = model_df['test_smape'].idxmin()
            result[model] = model_df.loc[best_idx]

    return result


def load_tft_predictions(run_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Lädt TFT Predictions aus predictions.npz oder predictions_{split}.npy

    Returns:
        (actuals, predictions, timestamps) oder None
    """
    # TFT predictions sind in results/tft/runs/{run_id}/predictions/
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


def load_prophet_predictions(run_id: str) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Lädt Prophet Predictions aus predictions/predictions_test.npy

    Returns:
        (actuals, predictions) oder None
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


def load_arima_predictions(run_id: str) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Lädt ARIMA Predictions aus predictions/predictions_test.npy

    Returns:
        (actuals, predictions) oder None
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


def plot_metric_comparison_fallback(
        dataset: str,
        best_runs: Dict[str, pd.Series],
        output_dir: Path
) -> None:
    """
    Fallback: Zeigt Metric Comparison wenn Predictions fehlen.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        'TFT': '#2E86AB',
        'Prophet': '#A23B72',
        'ARIMA': '#F18F01'
    }

    models = list(best_runs.keys())
    x = np.arange(len(models))
    width = 0.25

    # Metriken zum Plotten
    metrics = ['test_mae', 'test_rmse', 'test_smape']
    metric_labels = ['MAE', 'RMSE', 'SMAPE (%)']

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [best_runs[model][metric] for model in models]
        offset = (i - 1) * width

        bars = ax.bar(x + offset, values, width,
                      label=label, alpha=0.8, edgecolor='black', linewidth=1)

        # Annotate bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.02,
                    f'{value:.1f}',
                    ha='center', va='bottom', fontsize=9)

    # Styling
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Model Performance Comparison: {dataset} Dataset (Test Set)\n[Predictions not available - showing metrics only]',
        fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(title='Metrics', fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save
    output_path = output_dir / f"multi_model_comparison_{dataset.lower()}_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[plot_multi_model_forecast] Saved fallback plot: {output_path}")

    plt.close()


def plot_multi_model_forecast(
        dataset: str,
        best_runs: Dict[str, pd.Series],
        output_dir: Path,
        max_points: int = 200
) -> None:
    """
    Erstellt Multi-Model Forecast Plot.

    Args:
        dataset: Dataset name (Booksales oder Walmart)
        best_runs: Dict mit best run per model
        output_dir: Output directory
        max_points: Max Anzahl Punkte für bessere Lesbarkeit
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    colors = {
        'Actual': '#2C3E50',  # Dunkelgrau/Schwarz
        'TFT': '#2E86AB',  # Blau
        'Prophet': '#A23B72',  # Lila
        'ARIMA': '#F18F01'  # Orange
    }

    actuals_plotted = False
    predictions_data = {}

    # Lade Predictions für jedes Modell
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
            print(f"[plot_multi_model_forecast] Warnung: Konnte {model} predictions nicht laden: {e}")

    if not predictions_data:
        # Wird bereits in main() gehandelt, hier nur silent fail
        plt.close()
        return

    # Flatten predictions (falls multidimensional)
    for model in list(predictions_data.keys()):
        actuals, predictions = predictions_data[model]

        # Flatten falls nötig
        if actuals.ndim > 1:
            actuals = actuals.flatten()
        if predictions.ndim > 1:
            predictions = predictions.flatten()

        # Limit points für bessere Darstellung
        if len(actuals) > max_points:
            step = len(actuals) // max_points
            actuals = actuals[::step]
            predictions = predictions[::step]

        predictions_data[model] = (actuals, predictions)

    # Plot Actual values (nur einmal, von erstem verfügbaren Modell)
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
    for model, (actuals, predictions) in predictions_data.items():
        smape = best_runs[model]['test_smape']
        mae = best_runs[model]['test_mae']

        ax.plot(x, predictions,
                label=f'{model} (SMAPE: {smape:.1f}%, MAE: {mae:.1f})',
                color=colors[model],
                linewidth=2,
                alpha=0.7,
                linestyle='--' if model != 'TFT' else '-')

    # Styling
    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Multi-Model Forecast Comparison: {dataset} Dataset (Test Set)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save
    output_path = output_dir / f"multi_model_forecast_{dataset.lower()}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[plot_multi_model_forecast] Saved: {output_path}")

    plt.close()


def plot_multi_model_forecast_aggregated(
        dataset: str,
        best_runs: Dict[str, pd.Series],
        output_dir: Path,
        n_samples: int = 100
) -> None:
    """
    Alternative: Zeigt aggregierte Statistik (mean ± std) statt einzelne Werte.
    Nützlich bei vielen Zeitserien (z.B. Walmart mit 3050 groups).
    """
    fig, ax = plt.subplots(figsize=(16, 8))

    colors = {
        'Actual': '#2C3E50',
        'TFT': '#2E86AB',
        'Prophet': '#A23B72',
        'ARIMA': '#F18F01'
    }

    predictions_data = {}

    # Lade Predictions
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
            print(f"[plot_multi_model_forecast] Warnung: {model} predictions nicht geladen: {e}")

    if not predictions_data:
        # Wird bereits in main() gehandelt
        plt.close()
        return

    # Sample random points falls zu viele
    first_model = list(predictions_data.keys())[0]
    actuals, _ = predictions_data[first_model]

    if actuals.ndim > 1:
        actuals = actuals.flatten()

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

    # Plot predictions
    for model, (actuals_full, predictions) in predictions_data.items():
        if predictions.ndim > 1:
            predictions = predictions.flatten()

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
    print(f"[plot_multi_model_forecast] Saved: {output_path}")

    plt.close()


def main() -> None:
    # Load model comparison
    df = load_model_comparison()

    # Output directory für übergreifende Cross-Model Plots
    output_dir = BASE_DIR / "results" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[plot_multi_model_forecast] Erstelle Multi-Model Forecast Plots...")
    print()

    # Für jedes Dataset
    for dataset in ['Booksales', 'Walmart']:
        print(f"Processing {dataset}...")

        # Get best runs
        best_runs = get_best_run_per_model(df, dataset)

        if not best_runs:
            print(f"  Warnung: Keine Best Runs für {dataset} gefunden.")
            continue

        # Check if ANY predictions are available
        has_any_predictions = False
        for model, row in best_runs.items():
            run_id = row['run_id']

            try:
                if model == 'TFT':
                    result = load_tft_predictions(run_id)
                    if result:
                        has_any_predictions = True
                        break
                elif model == 'Prophet':
                    result = load_prophet_predictions(run_id)
                    if result:
                        has_any_predictions = True
                        break
                elif model == 'ARIMA':
                    result = load_arima_predictions(run_id)
                    if result:
                        has_any_predictions = True
                        break
            except Exception:
                pass

        if has_any_predictions:
            # Create plots with predictions
            print(f"  ✓ Predictions gefunden - erstelle Forecast-Plots")
            plot_multi_model_forecast(dataset, best_runs, output_dir, max_points=200)
            plot_multi_model_forecast_aggregated(dataset, best_runs, output_dir, n_samples=100)
        else:
            # Fallback: Metric comparison
            print(f"  ℹ Keine Predictions gefunden - erstelle Metrik-Vergleich (Fallback)")
            plot_metric_comparison_fallback(dataset, best_runs, output_dir)

        print()

    print("=" * 80)
    print(f"✅ Multi-Model Plots erstellt in: {output_dir}")
    print("=" * 80)
    print()
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
# Output (mit Predictions):
#   - results/plots/multi_model_forecast_booksales.png
#   - results/plots/multi_model_forecast_walmart.png
#   - results/plots/multi_model_forecast_booksales_sampled.png
#   - results/plots/multi_model_forecast_walmart_sampled.png
#
# Output (ohne Predictions - Fallback):
#   - results/plots/multi_model_comparison_booksales_metrics.png
#   - results/plots/multi_model_comparison_walmart_metrics.png