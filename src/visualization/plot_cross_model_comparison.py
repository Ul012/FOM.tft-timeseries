# src/visualization/plot_cross_model_comparison.py
# Erstellt Cross-Model Comparison Plots (TFT vs Prophet vs ARIMA)
#
# Zeigt Best Performer pro Modell & Dataset als Grouped Bar Chart
# Primär: SMAPE, Sekundär: MAE als Annotation
#
# Nutzung:
#   python -m src.visualization.plot_cross_model_comparison

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import BASE_DIR

# Plot-Konfiguration
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10


def load_model_comparison() -> pd.DataFrame:
    """Lädt model_comparison.csv"""
    path = BASE_DIR / "results" / "eval" / "model_comparison.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"model_comparison.csv nicht gefunden: {path}\n"
            f"Bitte zuerst ausführen: python -m src.evaluation.aggregate_all_models_eval"
        )

    return pd.read_csv(path)


def get_best_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrahiert Best Run pro Modell & Dataset (niedrigste test_smape).
    Bevorzugt Optuna, sonst Baseline.
    """
    # Filtere nur Baseline & Optuna (Exploration ignorieren)
    best_df = df[df['type'].isin(['Baseline', 'Optuna'])].copy()

    # Für jede Kombination von (dataset, model): nehme Run mit niedrigstem test_smape
    idx = best_df.groupby(['dataset', 'model'])['test_smape'].idxmin()
    result = best_df.loc[idx].copy()

    return result.sort_values(['dataset', 'test_smape'])


def plot_cross_model_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Erstellt Grouped Bar Chart: Best Run pro Modell & Dataset.
    Separate Subplots für Booksales und Walmart.
    """
    best_runs = get_best_runs(df)

    # Datasets
    datasets = sorted(best_runs['dataset'].unique())

    # Figure mit 2 Subplots (nebeneinander)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Cross-Model Performance Comparison (Best Runs)',
                 fontsize=14, fontweight='bold', y=0.98)

    colors = {
        'TFT': '#2E86AB',  # Blau
        'Prophet': '#A23B72',  # Lila
        'ARIMA': '#F18F01'  # Orange
    }

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        dataset_df = best_runs[best_runs['dataset'] == dataset]

        # Sortiere nach SMAPE (beste zuerst)
        dataset_df = dataset_df.sort_values('test_smape')

        models = dataset_df['model'].tolist()
        smape_values = dataset_df['test_smape'].tolist()
        mae_values = dataset_df['test_mae'].tolist()
        run_types = dataset_df['type'].tolist()

        # Bar positions
        x_pos = np.arange(len(models))

        # Create bars
        bars = ax.bar(x_pos, smape_values,
                      color=[colors[m] for m in models],
                      alpha=0.8, edgecolor='black', linewidth=1.2)

        # Annotate bars with values
        for i, (bar, smape, mae, run_type) in enumerate(zip(bars, smape_values, mae_values, run_types)):
            height = bar.get_height()

            # SMAPE value on top of bar
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                    f'{smape:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)

            # MAE value inside/below bar
            y_pos = height / 2 if height > 5 else -2
            text_color = 'white' if height > 5 else 'black'
            ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                    f'MAE: {mae:.1f}',
                    ha='center', va='center', fontsize=8,
                    color=text_color, style='italic')

            # Run type (Baseline/Optuna) as small label
            ax.text(bar.get_x() + bar.get_width() / 2., -0.5,
                    run_type, ha='center', va='top', fontsize=7, color='gray')

        # Styling
        ax.set_xlabel('Model', fontsize=11, fontweight='bold')
        ax.set_ylabel('Test SMAPE (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{dataset} Dataset', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # Y-axis starts at 0
        ax.set_ylim(bottom=0, top=max(smape_values) * 1.15)

    plt.tight_layout()

    # Save
    output_path = output_dir / "cross_model_comparison_best.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[plot_cross_model_comparison] Saved: {output_path}")

    plt.close()


def plot_cross_model_comparison_combined(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Alternative: Single plot mit allen Datasets gruppiert.
    """
    best_runs = get_best_runs(df)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {
        'TFT': '#2E86AB',
        'Prophet': '#A23B72',
        'ARIMA': '#F18F01'
    }

    datasets = sorted(best_runs['dataset'].unique())
    models = ['TFT', 'Prophet', 'ARIMA']

    # Grouped bar chart
    x = np.arange(len(datasets))
    width = 0.25

    for i, model in enumerate(models):
        model_data = []
        for dataset in datasets:
            model_df = best_runs[(best_runs['dataset'] == dataset) &
                                 (best_runs['model'] == model)]
            if len(model_df) > 0:
                model_data.append(model_df.iloc[0]['test_smape'])
            else:
                model_data.append(0)

        offset = (i - 1) * width
        bars = ax.bar(x + offset, model_data, width,
                      label=model, color=colors[model],
                      alpha=0.8, edgecolor='black', linewidth=1)

        # Annotate
        for bar, value in zip(bars, model_data):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                        f'{value:.1f}%',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test SMAPE (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Model Performance Comparison (Best Runs per Model)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(title='Model', title_fontsize=11, fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    output_path = output_dir / "cross_model_comparison_combined.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[plot_cross_model_comparison] Saved: {output_path}")

    plt.close()


def print_summary(df: pd.DataFrame) -> None:
    """Gibt Zusammenfassung der Best Runs aus."""
    best_runs = get_best_runs(df)

    print()
    print("=" * 80)
    print("CROSS-MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print()

    for dataset in sorted(best_runs['dataset'].unique()):
        dataset_df = best_runs[best_runs['dataset'] == dataset].sort_values('test_smape')

        print(f"{dataset}:")
        print("-" * 80)
        for _, row in dataset_df.iterrows():
            print(f"  {row['model']:8} ({row['type']:10}) | "
                  f"SMAPE: {row['test_smape']:6.2f}% | "
                  f"MAE: {row['test_mae']:8.2f} | "
                  f"RMSE: {row['test_rmse']:8.2f}")
        print()

    # Winner pro Dataset
    print("🏆 WINNER:")
    print("-" * 80)
    for dataset in sorted(best_runs['dataset'].unique()):
        winner = best_runs[best_runs['dataset'] == dataset].nsmallest(1, 'test_smape').iloc[0]
        print(f"  {dataset:12} : {winner['model']} ({winner['type']}) - "
              f"SMAPE {winner['test_smape']:.2f}%")
    print()


def main() -> None:
    # Load data
    df = load_model_comparison()

    # Output directory für übergreifende Cross-Model Plots
    output_dir = BASE_DIR / "results" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[plot_cross_model_comparison] Erstelle Cross-Model Comparison Plots...")

    # Create plots
    plot_cross_model_comparison(df, output_dir)  # Side-by-side subplots
    plot_cross_model_comparison_combined(df, output_dir)  # Grouped bar chart

    # Print summary
    print_summary(df)

    print()
    print("=" * 80)
    print(f"✅ Plots erstellt in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Aufruf:
#   python -m src.visualization.plot_cross_model_comparison
#
# Output:
#   - results/plots/cross_model_comparison_best.png
#   - results/plots/cross_model_comparison_combined.png