# src/evaluation/analyze_optuna_tft_trials.py
"""
Analysiert alle Optuna Trials und erstellt detaillierte Berichte.

Aufruf:
    Booksales:
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.evaluation.analyze_optuna_tft_trials --study-name tft_newyear

    Walmart:
    $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.evaluation.analyze_optuna_tft_trials --study-name walmart --top-n 10
"""

import argparse
from pathlib import Path

import optuna
import pandas as pd

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config

# ============================================================================
# KONSTANTEN
# ============================================================================

_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]

OPTUNA_BASE_DIR = BASE_DIR / "results" / "tft" / "optuna" / _dataset_name
STORAGE_PATH = OPTUNA_BASE_DIR / "tft_studies.db"
OPTUNA_STORAGE = f"sqlite:///{STORAGE_PATH}"


# ============================================================================
# FUNKTIONEN
# ============================================================================


def analyze_all_trials(study_name: str, top_n: int = 10):
    """
    Lädt alle Trials aus einer Optuna Study und erstellt detaillierte Analysen.

    Args:
        study_name: Name der Study
        top_n: Anzahl Top-Trials zum Anzeigen
    """
    if not STORAGE_PATH.exists():
        raise FileNotFoundError(f"Datenbank nicht gefunden: {STORAGE_PATH}")

    study = optuna.load_study(study_name=study_name, storage=OPTUNA_STORAGE)

    print("=" * 80)
    print(f"OPTUNA TRIAL-ANALYSE: {study_name}")
    print("=" * 80)
    print()

    df = study.trials_dataframe()

    print("📊 ÜBERSICHT:")
    print("-" * 80)
    print(f"Gesamt Trials:        {len(df)}")
    print(f"Abgeschlossene:       {len(df[df['state'] == 'COMPLETE'])}")
    print(f"Gepruned:             {len(df[df['state'] == 'PRUNED'])}")
    print(f"Fehlgeschlagen:       {len(df[df['state'] == 'FAIL'])}")
    print()

    df_complete = df[df["state"] == "COMPLETE"].copy()

    if len(df_complete) == 0:
        print("Keine abgeschlossenen Trials gefunden!")
        return

    print(f"🏆 TOP {top_n} TRIALS (nach val_mae):")
    print("-" * 80)

    df_top = df_complete.nsmallest(top_n, "value")

    cols_to_show = [
        "number",
        "value",
        "params_learning_rate",
        "params_batch_size",
        "params_hidden_size",
        "params_dropout",
        "duration",
    ]

    cols_to_show = [c for c in cols_to_show if c in df_top.columns]

    print(df_top[cols_to_show].to_string(index=False))
    print()

    print("📈 PARAMETER-STATISTIK (nur abgeschlossene Trials):")
    print("-" * 80)

    param_cols = [c for c in df_complete.columns if c.startswith("params_")]

    for col in param_cols:
        param_name = col.replace("params_", "")

        if df_complete[col].dtype in ["float64", "int64"]:
            print(f"\n{param_name}:")
            print(f"  Min:    {df_complete[col].min():.4f}")
            print(f"  Max:    {df_complete[col].max():.4f}")
            print(f"  Mean:   {df_complete[col].mean():.4f}")
            print(f"  Median: {df_complete[col].median():.4f}")
        else:
            print(f"\n{param_name}:")
            value_counts = df_complete[col].value_counts()
            for val, count in value_counts.items():
                print(f"  {val}: {count} Trials")

    print()

    print("🔗 KORRELATIONEN (Parameter vs. val_mae):")
    print("-" * 80)

    correlations = []
    for col in param_cols:
        if df_complete[col].dtype in ["float64", "int64"]:
            corr = df_complete[col].corr(df_complete["value"])
            correlations.append((col.replace("params_", ""), corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    for param, corr in correlations:
        direction = "↑" if corr > 0 else "↓"
        print(f"  {param:<25} {direction} {corr:>7.4f}")

    print()
    print("Hinweis: Positive Korrelation = höherer Wert → höhere val_mae (schlechter)")
    print("         Negative Korrelation = höherer Wert → niedrigere val_mae (besser)")
    print()

    if len(df[df["state"] == "PRUNED"]) > 0:
        print("✂️ PRUNING-ANALYSE:")
        print("-" * 80)

        df_pruned = df[df["state"] == "PRUNED"]

        print(
            f"Gepruned: {len(df_pruned)} von {len(df)} Trials ({100 * len(df_pruned) / len(df):.1f}%)"
        )

        if "duration" in df.columns:
            avg_duration_complete = df_complete["duration"].mean()
            avg_duration_pruned = df_pruned["duration"].mean()

            # FIX: Timedelta zu Sekunden konvertieren
            avg_complete_sec = avg_duration_complete.total_seconds()
            avg_pruned_sec = avg_duration_pruned.total_seconds()

            print(f"Durchschn. Dauer (abgeschlossen): {avg_complete_sec / 60:.1f} min")
            print(f"Durchschn. Dauer (gepruned):      {avg_pruned_sec / 60:.1f} min")
            print(
                f"Zeitersparnis durch Pruning:      {100 * (1 - avg_pruned_sec / avg_complete_sec):.1f}%"
            )

        print()

    output_dir = OPTUNA_BASE_DIR / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{study_name}_all_trials.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Alle Trials exportiert: {csv_path}")

    top_csv_path = output_dir / f"{study_name}_top{top_n}.csv"
    df_top.to_csv(top_csv_path, index=False)
    print(f"✅ Top {top_n} Trials exportiert: {top_csv_path}")

    print()
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Analysiere alle Optuna Trials")
    parser.add_argument(
        "--study-name", type=str, required=True, help="Name der Optuna Study"
    )
    parser.add_argument(
        "--top-n", type=int, default=10, help="Anzahl Top-Trials (default: 10)"
    )

    args = parser.parse_args()

    analyze_all_trials(args.study_name, args.top_n)


if __name__ == "__main__":
    main()

# Booksales:
# $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.evaluation.analyze_optuna_tft_trials --study-name tft_newyear
#
# Walmart:
# $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.evaluation.analyze_optuna_tft_trials --study-name walmart_full --top-n 10






























