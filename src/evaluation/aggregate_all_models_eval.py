# src/evaluation/aggregate_all_models_eval.py
# Kombiniert TFT, Prophet und ARIMA Evaluierungen zu einer Master-Tabelle.
#
# Workflow:
#   1. python -m src.evaluation.aggregate_tft_eval
#   2. python -m src.evaluation.aggregate_prophet_eval
#   3. python -m src.evaluation.aggregate_arima_eval
#   4. python -m src.evaluation.aggregate_all_models_eval
#
# Nutzung:
#   python -m src.evaluation.aggregate_all_models_eval

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import BASE_DIR, EVALUATION_METRICS


def _parse_run_metadata(run_id: str, dataset_hint: str | None = None) -> Dict[str, str]:
    """
    Extrahiert Metadaten aus run_id.

    Returns:
        Dict mit 'model', 'dataset', 'type' (baseline/exploration/optuna)
    """
    run_lower = run_id.lower()

    # Model
    if 'prophet' in run_lower:
        model = 'Prophet'
    elif 'arima' in run_lower:
        model = 'ARIMA'
    else:
        model = 'TFT'  # Default für TFT-Runs

    # Dataset (aus run_id oder Hint)
    if 'walmart' in run_lower:
        dataset = 'Walmart'
    elif 'booksales' in run_lower:
        dataset = 'Booksales'
    elif dataset_hint:
        dataset = dataset_hint.capitalize()
    else:
        dataset = 'Unknown'

    # Type
    if 'optuna' in run_lower or 'best' in run_lower:
        run_type = 'Optuna'
    elif 'baseline' in run_lower:
        run_type = 'Baseline'
    elif 'exploration' in run_lower:
        run_type = 'Exploration'
    else:
        run_type = 'Other'

    return {
        'model': model,
        'dataset': dataset,
        'type': run_type,
    }


def _load_tft_overview() -> pd.DataFrame:
    """Lädt TFT eval_overview.csv."""
    path = BASE_DIR / "results" / "tft" / "eval" / "eval_overview.csv"

    if not path.exists():
        print(f"[aggregate_all_models] Warnung: TFT overview nicht gefunden: {path}")
        print("                        Führe zuerst aus: python -m src.evaluation.aggregate_tft_eval")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Parse Metadaten
    metadata = df['run_id'].apply(lambda x: pd.Series(_parse_run_metadata(x)))

    # Füge Metadaten hinzu (überschreibe falls vorhanden, vermeidet doppelte Spalten)
    for col in metadata.columns:
        df[col] = metadata[col]

    return df


def _load_prophet_overview() -> pd.DataFrame:
    """Lädt Prophet eval_overview.csv."""
    path = BASE_DIR / "results" / "prophet" / "eval_overview.csv"

    if not path.exists():
        print(f"[aggregate_all_models] Warnung: Prophet overview nicht gefunden: {path}")
        print("                        Führe zuerst aus: python -m src.evaluation.aggregate_prophet_eval")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Parse Metadaten (dataset ist in Prophet bereits vorhanden)
    metadata = df.apply(
        lambda row: pd.Series(_parse_run_metadata(row['run_id'], row.get('dataset'))),
        axis=1
    )

    # Füge Metadaten hinzu (überschreibe falls vorhanden, vermeidet doppelte Spalten)
    for col in metadata.columns:
        df[col] = metadata[col]

    return df


def _load_arima_overview() -> pd.DataFrame:
    """Lädt ARIMA eval_overview.csv."""
    path = BASE_DIR / "results" / "arima" / "eval_overview.csv"

    if not path.exists():
        print(f"[aggregate_all_models] Warnung: ARIMA overview nicht gefunden: {path}")
        print("                        Führe zuerst aus: python -m src.evaluation.aggregate_arima_eval")
        return pd.DataFrame()

    df = pd.read_csv(path)

    # Parse Metadaten (dataset ist in ARIMA bereits vorhanden)
    metadata = df.apply(
        lambda row: pd.Series(_parse_run_metadata(row['run_id'], row.get('dataset'))),
        axis=1
    )

    # Füge Metadaten hinzu (überschreibe falls vorhanden, vermeidet doppelte Spalten)
    for col in metadata.columns:
        df[col] = metadata[col]

    return df


def _standardize_columns(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Standardisiert Spalten für einheitliches Format.

    Alle Modelle sollen haben:
    - model, dataset, type, run_id
    - val_mae, val_rmse, val_mape, val_smape, val_r2
    - test_mae, test_rmse, test_mape, test_smape, test_r2
    """
    if df.empty:
        return df

    # Basis-Spalten die alle haben sollten
    base_cols = ['model', 'dataset', 'type', 'run_id']

    # Metrik-Spalten
    metric_cols = []
    for split in ['val', 'test']:
        for metric in EVALUATION_METRICS:
            col = f"{split}_{metric}"
            metric_cols.append(col)
            # Füge Spalte hinzu wenn sie fehlt (z.B. r2 bei ARIMA/Prophet)
            if col not in df.columns:
                df[col] = None

    # Optional: checkpoint_path (nur TFT hat das)
    optional_cols = []
    if 'checkpoint_path' in df.columns:
        optional_cols.append('checkpoint_path')

    # Finale Spalten-Reihenfolge
    final_cols = base_cols + metric_cols + optional_cols

    # Nur existierende Spalten nehmen
    existing_cols = [c for c in final_cols if c in df.columns]

    return df[existing_cols].copy()


def aggregate_all_models() -> pd.DataFrame:
    """
    Kombiniert alle Modell-Evaluierungen zu einer Master-Tabelle.

    Returns:
        DataFrame mit allen Evaluierungen in einheitlichem Format
    """
    print("[aggregate_all_models] Lade Evaluierungen...")

    # Lade einzelne Modelle
    tft_df = _load_tft_overview()
    prophet_df = _load_prophet_overview()
    arima_df = _load_arima_overview()

    # Standardisiere
    tft_df = _standardize_columns(tft_df, 'TFT')
    prophet_df = _standardize_columns(prophet_df, 'Prophet')
    arima_df = _standardize_columns(arima_df, 'ARIMA')

    # Kombiniere
    dfs = [df for df in [tft_df, prophet_df, arima_df] if not df.empty]

    if not dfs:
        raise FileNotFoundError(
            "Keine Evaluierungen gefunden. Bitte führe zuerst aus:\n"
            "  - python -m src.evaluation.aggregate_tft_eval\n"
            "  - python -m src.evaluation.aggregate_prophet_eval\n"
            "  - python -m src.evaluation.aggregate_arima_eval"
        )

    combined_df = pd.concat(dfs, ignore_index=True)

    # Sortiere: erst nach dataset, dann model, dann type
    combined_df.sort_values(['dataset', 'model', 'type', 'run_id'], inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    return combined_df


def main() -> None:
    df = aggregate_all_models()

    # Output
    output_dir = BASE_DIR / "results" / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "model_comparison.csv"
    json_path = output_dir / "model_comparison.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print()
    print("[aggregate_all_models] Aggregation abgeschlossen.")
    print("=" * 60)
    print("ZUSAMMENFASSUNG:")
    print("=" * 60)
    print(f"Gesamt Runs:  {len(df)}")
    print()

    # Aufschlüsselung nach Modell
    print("Nach Modell:")
    for model in df['model'].unique():
        count = len(df[df['model'] == model])
        print(f"  {model:8} : {count:3} Runs")
    print()

    # Aufschlüsselung nach Dataset
    print("Nach Dataset:")
    for dataset in df['dataset'].unique():
        count = len(df[df['dataset'] == dataset])
        print(f"  {dataset:10} : {count:3} Runs")
    print()

    # Best Performers (Test SMAPE)
    print("🏆 BEST PERFORMERS (Test SMAPE):")
    print("-" * 60)

    test_df = df.copy()
    test_df = test_df[test_df['type'].isin(['Baseline', 'Optuna'])]  # Nur Best Runs

    for dataset in sorted(df['dataset'].unique()):
        dataset_df = test_df[test_df['dataset'] == dataset]
        if dataset_df.empty:
            continue

        print(f"\n{dataset}:")
        best_runs = dataset_df.nsmallest(5, 'test_smape')

        for _, row in best_runs.iterrows():
            print(f"  {row['model']:8} {row['type']:12} | "
                  f"SMAPE: {row['test_smape']:6.2f}% | "
                  f"MAE: {row['test_mae']:8.2f}")

    print()
    print("=" * 60)
    print(f"CSV  : {csv_path}")
    print(f"JSON : {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Workflow:
#   1. python -m src.evaluation.aggregate_tft_eval
#   2. python -m src.evaluation.aggregate_prophet_eval
#   3. python -m src.evaluation.aggregate_arima_eval
#   4. python -m src.evaluation.aggregate_all_models_eval
#
# Output: results/eval/model_comparison.{csv,json}