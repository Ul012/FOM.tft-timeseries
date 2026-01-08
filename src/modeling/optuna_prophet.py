# src/modeling/optuna_prophet.py
"""
Hyperparameter-Optimierung für Prophet mit Optuna.

METRIKEN:
- Primäre Metrik (Optimierungsziel): val_mae (Mean Absolute Error)
- Geloggte Metriken: MAE, RMSE, MAPE, SMAPE
- Ziel: val_mae minimieren

KONFIGURATION:
- Alle Parameter sind im Script hardcodiert
- Search Space basiert auf Prophet Best Practices
- Training-Parameter entsprechen baseline.yaml

Aufrufbeispiele:
    # Test-Run
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
    python -m src.modeling.optuna_prophet --study-name prophet_test --n-trials 2

    # Einfacher Run (10 Trials)
    python -m src.modeling.optuna_prophet --n-trials 10

    # Mit Custom Study Name
    python -m src.modeling.optuna_prophet --study-name prophet_full --n-trials 30

    # Fortsetzen einer existierenden Study
    python -m src.modeling.optuna_prophet --study-name prophet_full --n-trials 10
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from prophet import Prophet

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema

# ============================================================================
# GLOBALE KONFIGURATION
# ============================================================================

# Dataset-Config laden
_dataset_config = load_dataset_config()
_schema = get_schema(_dataset_config)
_dataset_name = _dataset_config["name"]

# Pfade (dataset-spezifisch)
OPTUNA_BASE_DIR = BASE_DIR / "results" / "prophet" / "optuna" / _dataset_name
OPTUNA_STORAGE = f"sqlite:///{OPTUNA_BASE_DIR}/prophet_studies.db"

# ============================================================================
# HARDCODIERTE PARAMETER
# ============================================================================

# Search Space (Prophet Hyperparameter)
SEARCH_SPACE = {
    "changepoint_prior_scale": {"min": 0.001, "max": 0.5, "log": True},
    "seasonality_prior_scale": {"min": 0.01, "max": 10.0, "log": True},
    "holidays_prior_scale": {"min": 0.01, "max": 10.0, "log": True},
    "seasonality_mode": {"choices": ["multiplicative", "additive"]},
    "growth": {"choices": ["linear"]},  # logistic braucht cap/floor
}

# Feste Parameter
FIXED_CONFIG = {
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": False,
    "interval_width": 0.95,
    "mcmc_samples": 0,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _load_prophet_spec(dataset_name: str) -> Dict[str, Any]:
    """Lädt prophet_spec.json."""
    spec_path = BASE_DIR / "data" / "processed" / dataset_name / "prophet_spec.json"

    if not spec_path.exists():
        raise FileNotFoundError(f"prophet_spec.json nicht gefunden: {spec_path}")

    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_prophet_dataframe(
        df: pd.DataFrame,
        time_col: str,
        target_col: str,
        regressors: List[str]
) -> pd.DataFrame:
    """Konvertiert zu Prophet-Format."""
    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(df[time_col]),
        "y": df[target_col].astype("float64")
    })

    for reg in regressors:
        if reg in df.columns:
            prophet_df[reg] = pd.to_numeric(df[reg], errors='coerce').astype("float64").fillna(0.0)
        else:
            prophet_df[reg] = 0.0

    return prophet_df


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Berechnet Evaluation-Metriken."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "mape": None, "smape": None}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    mape = None
    if not (y_true == 0).any():
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = None
    if not (denominator == 0).any():
        smape = float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)

    return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape}


# ============================================================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================================================


def objective(trial: optuna.Trial) -> float:
    """
    Optuna Objective Function für Prophet-Hyperparameter-Optimierung.

    Args:
        trial: Optuna Trial-Objekt

    Returns:
        val_mae: Validation MAE (zu minimierende Metrik)

    Raises:
        optuna.TrialPruned: Wenn Trial abgebrochen wird
    """
    # -------------------------------------------------------------------------
    # 1. Hyperparameter vorschlagen
    # -------------------------------------------------------------------------
    changepoint_prior_scale = trial.suggest_float(
        "changepoint_prior_scale",
        SEARCH_SPACE["changepoint_prior_scale"]["min"],
        SEARCH_SPACE["changepoint_prior_scale"]["max"],
        log=SEARCH_SPACE["changepoint_prior_scale"]["log"]
    )

    seasonality_prior_scale = trial.suggest_float(
        "seasonality_prior_scale",
        SEARCH_SPACE["seasonality_prior_scale"]["min"],
        SEARCH_SPACE["seasonality_prior_scale"]["max"],
        log=SEARCH_SPACE["seasonality_prior_scale"]["log"]
    )

    holidays_prior_scale = trial.suggest_float(
        "holidays_prior_scale",
        SEARCH_SPACE["holidays_prior_scale"]["min"],
        SEARCH_SPACE["holidays_prior_scale"]["max"],
        log=SEARCH_SPACE["holidays_prior_scale"]["log"]
    )

    seasonality_mode = trial.suggest_categorical(
        "seasonality_mode",
        SEARCH_SPACE["seasonality_mode"]["choices"]
    )

    growth = trial.suggest_categorical(
        "growth",
        SEARCH_SPACE["growth"]["choices"]
    )

    # -------------------------------------------------------------------------
    # 2. Datasets laden
    # -------------------------------------------------------------------------
    prophet_spec = _load_prophet_spec(_dataset_name)

    time_col = prophet_spec["time_col"]
    target_col = prophet_spec["target_col"]
    group_cols = prophet_spec["group_cols"]
    regressors = prophet_spec["regressors"]
    country_holidays = prophet_spec.get("country_holidays")

    processed_dir = BASE_DIR / "data" / "processed" / _dataset_name
    df_train = pd.read_parquet(processed_dir / "train.parquet")
    df_val = pd.read_parquet(processed_dir / "val.parquet")

    # -------------------------------------------------------------------------
    # 3. Training über alle Gruppen
    # -------------------------------------------------------------------------
    all_mae = []

    trial_id = f"trial_{trial.number:04d}"
    trial_dir = OPTUNA_BASE_DIR / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)

    if not group_cols:
        # Keine Gruppen
        groups = [("all", df_train, df_val)]
    else:
        # Gruppierung
        groups = []
        for group_values, group_train_df in df_train.groupby(group_cols):
            group_id = "_".join(str(v) for v in group_values) if isinstance(group_values, tuple) else str(group_values)

            # Entsprechende Val-Gruppe
            if isinstance(group_values, tuple):
                mask = True
                for col, val in zip(group_cols, group_values):
                    mask = mask & (df_val[col] == val)
                group_val_df = df_val[mask]
            else:
                group_val_df = df_val[df_val[group_cols[0]] == group_values]

            if len(group_val_df) > 0:
                groups.append((group_id, group_train_df, group_val_df))

    print(f"\n[Trial {trial.number}] Training {len(groups)} Gruppen...")
    print(f"  Hyperparameter: changepoint={changepoint_prior_scale:.4f}, "
          f"seasonality={seasonality_prior_scale:.2f}, "
          f"holidays={holidays_prior_scale:.2f}, "
          f"mode={seasonality_mode}, growth={growth}")

    for group_id, group_train_df, group_val_df in groups:
        # Prophet-Daten vorbereiten
        train_prophet = _prepare_prophet_dataframe(group_train_df, time_col, target_col, regressors)
        val_prophet = _prepare_prophet_dataframe(group_val_df, time_col, target_col, regressors)

        # Modell erstellen
        model = Prophet(
            growth=growth,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=FIXED_CONFIG["yearly_seasonality"],
            weekly_seasonality=FIXED_CONFIG["weekly_seasonality"],
            daily_seasonality=FIXED_CONFIG["daily_seasonality"],
            interval_width=FIXED_CONFIG["interval_width"],
            mcmc_samples=FIXED_CONFIG["mcmc_samples"]
        )

        # Regressoren hinzufügen
        for reg in regressors:
            model.add_regressor(reg)

        # Country Holidays
        if country_holidays:
            model.add_country_holidays(country_name=country_holidays)

        # Training (suppress output)
        try:
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

            model.fit(train_prophet)

            # Validation Forecast
            forecast = model.predict(val_prophet)

            # Metriken
            y_true = val_prophet["y"].values
            y_pred = forecast["yhat"].values

            metrics = _calculate_metrics(y_true, y_pred)

            if metrics["mae"] is not None:
                all_mae.append(metrics["mae"])

        except Exception as e:
            print(f"  ⚠ Fehler bei Gruppe {group_id}: {e}")
            continue

    # -------------------------------------------------------------------------
    # 4. Durchschnittliche val_mae berechnen
    # -------------------------------------------------------------------------
    if len(all_mae) == 0:
        print(f"  ✗ Trial {trial.number} FAILED (keine validen Gruppen)")
        raise optuna.TrialPruned()

    val_mae = float(np.mean(all_mae))

    print(f"  ✓ Trial {trial.number} abgeschlossen: val_mae={val_mae:.2f} (über {len(all_mae)} Gruppen)")

    # Speichere Trial-Zusammenfassung
    trial_summary = {
        "trial_number": trial.number,
        "val_mae": val_mae,
        "n_groups": len(all_mae),
        "hyperparameters": {
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_prior_scale": seasonality_prior_scale,
            "holidays_prior_scale": holidays_prior_scale,
            "seasonality_mode": seasonality_mode,
            "growth": growth
        }
    }

    summary_path = trial_dir / "trial_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(trial_summary, f, indent=2)

    return val_mae


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Prophet Hyperparameter-Optimierung mit Optuna"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="prophet_hpo",
        help="Name der Optuna Study (default: prophet_hpo)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Anzahl Trials (default: 20)"
    )

    args = parser.parse_args()

    # Optuna-Verzeichnis erstellen
    OPTUNA_BASE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PROPHET HYPERPARAMETER-OPTIMIERUNG MIT OPTUNA")
    print("=" * 80)
    print(f"Dataset:     {_dataset_name}")
    print(f"Study Name:  {args.study_name}")
    print(f"N Trials:    {args.n_trials}")
    print(f"Storage:     {OPTUNA_STORAGE}")
    print("=" * 80)

    # Study erstellen oder laden
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
        pruner=pruner
    )

    print(f"\n[Optuna] Study '{args.study_name}' gestartet/fortgesetzt")
    print(f"[Optuna] Bereits abgeschlossene Trials: {len(study.trials)}")
    print()

    # Optimization starten
    import time
    start_time = time.time()

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    elapsed_time = time.time() - start_time

    # Ergebnisse
    print("\n" + "=" * 80)
    print("OPTIMIZATION ABGESCHLOSSEN")
    print("=" * 80)
    print(f"Anzahl Trials:      {len(study.trials)}")
    print(f"Geprunte Trials:    {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Gesamt-Zeit:        {elapsed_time / 60:.1f} Minuten")
    print()
    print(f"Beste val_mae:      {study.best_value:.4f}")
    print(f"Bestes Trial:       {study.best_trial.number}")
    print()
    print("Beste Hyperparameter:")
    print("-" * 80)
    for key, value in study.best_params.items():
        print(f"  {key:<30} = {value}")
    print()

    # -------------------------------------------------------------------------
    # Ergebnisse speichern (JSON + CSV)
    # -------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON Export
    results_json = {
        "study_name": args.study_name,
        "timestamp": timestamp,
        "dataset": _dataset_name,
        "n_trials": len(study.trials),
        "elapsed_time_sec": elapsed_time,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "fixed_config": FIXED_CONFIG,
        "search_space": SEARCH_SPACE,
    }

    json_file = OPTUNA_BASE_DIR / f"study_{args.study_name}_{timestamp}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"✅ Ergebnisse gespeichert: {json_file}")

    # CSV Export (alle Trials)
    df_trials = study.trials_dataframe()
    csv_file = OPTUNA_BASE_DIR / f"study_{args.study_name}_{timestamp}.csv"
    df_trials.to_csv(csv_file, index=False)

    print(f"✅ Alle Trials als CSV:  {csv_file}")
    print()

    # -------------------------------------------------------------------------
    # Top 5 Trials anzeigen
    # -------------------------------------------------------------------------
    print("TOP 5 TRIALS:")
    print("-" * 80)
    df_top = df_trials.nsmallest(5, "value")[
        ["number", "value", "params_changepoint_prior_scale", "params_seasonality_mode", "state"]
    ]
    print(df_top.to_string(index=False))
    print()

    print("=" * 80)
    print("NÄCHSTE SCHRITTE:")
    print("-" * 80)
    print(f"1. Beste Config exportieren:")
    print(f"   python -m src.modeling.optuna_prophet_export_best --study-name {args.study_name}")
    print()
    print(f"2. Mit bester Config trainieren:")
    print(
        f"   python -m src.modeling.trainer_prophet --config configs/models/prophet/{_dataset_name}/optuna_{args.study_name}_best.yaml")
    print()
    print(f"3. Visualisierungen erstellen:")
    print(f"   python -m src.visualization.plot_prophet_optuna_study --study-name {args.study_name}")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Aufruf:
#   Booksales:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
#   python -m src.modeling.optuna_prophet --study-name prophet_booksales --n-trials 20
#
#   Walmart:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"
#   python -m src.modeling.optuna_prophet --study-name prophet_walmart --n-trials 20