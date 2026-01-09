# src/modeling/optuna_arima.py
"""
Hyperparameter-Optimierung für ARIMA mit Optuna.

Optimiert max_p, max_q, max_d (non-seasonal) und max_P, max_Q, max_D (seasonal)
für minimale val_mae.

Aufruf:
    Booksales:
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
    python -m src.modeling.optuna_arima --study-name arima_booksales --n-trials 50

    Walmart:
    $env:DATASET_CONFIG="configs/datasets/walmart.yaml"
    python -m src.modeling.optuna_arima --study-name arima_walmart --n-trials 20
"""

# ============================================================================
# IMPORTS
# ============================================================================
import argparse
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from pmdarima import auto_arima

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config

# Warnings filtern
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ============================================================================
# KONFIGURATION
# ============================================================================
_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]

OPTUNA_BASE_DIR = BASE_DIR / "results" / "arima" / "optuna" / _dataset_name
OPTUNA_STORAGE = f"sqlite:///{OPTUNA_BASE_DIR}/arima_studies.db"

# ============================================================================
# HYPERPARAMETER-KONFIGURATION (wissenschaftlich begründet)
# ============================================================================

# Search Space (zu optimierende Parameter)
SEARCH_SPACE = {
    "max_p": {"min": 0, "max": 5},  # Non-seasonal AR order
    "max_q": {"min": 0, "max": 5},  # Non-seasonal MA order
    "max_P": {"min": 0, "max": 2},  # Seasonal AR order
    "max_Q": {"min": 0, "max": 2},  # Seasonal MA order
}

# Fixe Parameter (theoretisch begründet, nicht aus empirischen Runs)
# - max_d=1: Box-Jenkins Methodik zeigt, dass d>1 selten benötigt wird.
#            Augmented Dickey-Fuller Tests bestätigen meist Stationarität nach d=1.
#            d=2 führt häufig zu Over-Differencing (siehe Hyndman & Athanasopoulos, 2021).
# - max_D=1: In der Literatur wird für saisonale ARIMA fast ausschließlich D=1 verwendet.
#            Höhere seasonal differencing orders sind in der Praxis extrem selten nötig.
FIXED_ARIMA_PARAMS = {
    "max_d": 1,
    "max_D": 1,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _load_arima_spec(dataset_name: str) -> Dict:
    """Lade arima_spec.json"""
    spec_path = BASE_DIR / "data" / "processed" / dataset_name / "arima_spec.json"
    with open(spec_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _prepare_arima_data(df: pd.DataFrame, time_col: str, target_col: str, exog_vars: list):
    """Konvertiert DataFrame zu ARIMA-Format"""
    df = df.sort_values(time_col).reset_index(drop=True)
    endog = df[target_col].astype("float64")

    if not exog_vars:
        return endog, None

    exog = df[exog_vars].copy()
    for col in exog.columns:
        exog[col] = pd.to_numeric(exog[col], errors='coerce').fillna(0).astype("float64")

    return endog, exog


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Berechnet MAE"""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"mae": None}

    return {"mae": float(np.mean(np.abs(y_true - y_pred)))}


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def objective(trial: optuna.Trial) -> float:
    """Optuna Objective für ARIMA HPO"""

    # Lade Spec
    arima_spec = _load_arima_spec(_dataset_name)
    time_col = arima_spec["time_col"]
    target_col = arima_spec["target_col"]
    group_cols = arima_spec["group_cols"]
    exog_vars = arima_spec["exog_vars"]
    seasonal_period = arima_spec["seasonal_period"]

    # Hyperparameter
    params = {
        # Variable Parameter (von Optuna optimiert)
        "max_p": trial.suggest_int("max_p", SEARCH_SPACE["max_p"]["min"], SEARCH_SPACE["max_p"]["max"]),
        "max_q": trial.suggest_int("max_q", SEARCH_SPACE["max_q"]["min"], SEARCH_SPACE["max_q"]["max"]),
        "max_P": trial.suggest_int("max_P", SEARCH_SPACE["max_P"]["min"], SEARCH_SPACE["max_P"]["max"]),
        "max_Q": trial.suggest_int("max_Q", SEARCH_SPACE["max_Q"]["min"], SEARCH_SPACE["max_Q"]["max"]),
        # Fixe Parameter (theoretisch begründet)
        "max_d": FIXED_ARIMA_PARAMS["max_d"],
        "max_D": FIXED_ARIMA_PARAMS["max_D"],
    }

    # Lade Daten
    processed_dir = BASE_DIR / "data" / "processed" / _dataset_name
    df_train = pd.read_parquet(processed_dir / "train.parquet")
    df_val = pd.read_parquet(processed_dir / "val.parquet")

    # Gruppenbildung
    if not group_cols:
        groups = [("all", df_train, df_val)]
    else:
        groups = []
        for group_values, group_train_df in df_train.groupby(group_cols):
            group_id = "_".join(str(v) for v in group_values) if isinstance(group_values, tuple) else str(group_values)

            if isinstance(group_values, tuple):
                mask = True
                for col, val in zip(group_cols, group_values):
                    mask = mask & (df_val[col] == val)
                group_val_df = df_val[mask]
            else:
                group_val_df = df_val[df_val[group_cols[0]] == group_values]

            if len(group_val_df) > 0:
                groups.append((group_id, group_train_df, group_val_df))

    # Training
    print(f"\n[Trial {trial.number}] Training {len(groups)} Gruppen...")
    print(
        f"  Variable: max_p={params['max_p']}, max_q={params['max_q']}, max_P={params['max_P']}, max_Q={params['max_Q']}")
    print(f"  Fix:      max_d={params['max_d']}, max_D={params['max_D']}, m={seasonal_period}")

    total_groups = len(groups)
    all_mae = []
    success_count = 0
    error_count = 0

    trial_dir = OPTUNA_BASE_DIR / f"trial_{trial.number:04d}"
    trial_models_dir = trial_dir / "models"
    trial_models_dir.mkdir(parents=True, exist_ok=True)

    for idx, (group_id, group_train_df, group_val_df) in enumerate(groups, start=1):
        try:
            # ARIMA-Daten
            train_endog, train_exog = _prepare_arima_data(group_train_df, time_col, target_col, exog_vars)
            val_endog, val_exog = _prepare_arima_data(group_val_df, time_col, target_col, exog_vars)

            if len(train_endog) < 10:
                error_count += 1
                continue

            # Auto-ARIMA mit dynamischen start_* (verhindert max_* < start_* Fehler)
            model = auto_arima(
                train_endog,
                exogenous=train_exog,
                seasonal=seasonal_period > 1,
                m=seasonal_period,
                max_p=params["max_p"],
                max_d=params["max_d"],
                max_q=params["max_q"],
                max_P=params["max_P"],
                max_D=params["max_D"],
                max_Q=params["max_Q"],
                start_p=min(1, params["max_p"]),
                start_q=min(1, params["max_q"]),
                start_P=min(1, params["max_P"]) if seasonal_period > 1 else 0,
                start_Q=min(1, params["max_Q"]) if seasonal_period > 1 else 0,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False
            )

            # Forecast
            n_periods = len(val_endog)
            forecast = model.predict(n_periods=n_periods, exogenous=val_exog)

            # Metriken
            metrics = _calculate_metrics(val_endog.values, forecast)

            if metrics["mae"] is not None:
                all_mae.append(metrics["mae"])
                success_count += 1

                model_path = trial_models_dir / f"arima_{group_id}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

        except Exception:
            error_count += 1
            continue

        # Progress
        if idx % 50 == 0 or idx == total_groups:
            pct = int(100 * idx / total_groups)
            bar_len = 25
            filled = int(bar_len * idx / total_groups)
            bar = "=" * filled + (">" if filled < bar_len else "")
            bar = bar.ljust(bar_len)
            print(f"  [{bar}] {idx}/{total_groups} ({pct}%) | OK: {success_count} | Fehler: {error_count}")

    # Ergebnis
    if len(all_mae) == 0:
        print(f"\n  ✗ Trial {trial.number}: Alle Gruppen fehlgeschlagen!\n")
        return float('inf')

    val_mae = float(np.mean(all_mae))
    print(f"\n  ✓ Trial {trial.number}: val_mae={val_mae:.2f} ({success_count}/{total_groups} Gruppen)\n")

    # Speichere Summary
    with open(trial_dir / "trial_summary.json", "w") as f:
        json.dump({
            "trial_number": trial.number,
            "val_mae": val_mae,
            "n_groups": success_count,
            "hyperparameters": params,
            "seasonal_period": seasonal_period
        }, f, indent=2)

    return val_mae


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ARIMA Hyperparameter-Optimierung")
    parser.add_argument("--study-name", type=str, default="arima_hpo")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in Sekunden")
    args = parser.parse_args()

    OPTUNA_BASE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ARIMA HYPERPARAMETER-OPTIMIERUNG MIT OPTUNA")
    print("=" * 80)
    print(f"Dataset:     {_dataset_name}")
    print(f"Study Name:  {args.study_name}")
    print(f"N Trials:    {args.n_trials}")
    print(f"Storage:     {OPTUNA_STORAGE}")
    print("=" * 80)

    # Optuna Study (ohne Pruning)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=None
    )

    print(f"\n[Optuna] Study '{args.study_name}' gestartet/fortgesetzt")
    print(f"[Optuna] Bereits abgeschlossene Trials: {len(study.trials)}\n")

    # Optimization
    import time
    start_time = time.time()

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=True)

    elapsed_time = time.time() - start_time

    # Ergebnisse
    completed_trials = [t for t in study.trials if
                        t.state == optuna.trial.TrialState.COMPLETE and t.value != float('inf')]

    print("\n" + "=" * 80)
    print("OPTIMIZATION ABGESCHLOSSEN")
    print("=" * 80)
    print(f"Anzahl Trials:      {len(study.trials)}")
    print(f"Abgeschlossene:     {len(completed_trials)}")
    print(f"Gesamt-Zeit:        {elapsed_time / 60:.1f} Minuten")
    print()

    if len(completed_trials) == 0:
        print("⚠️  WARNUNG: Keine erfolgreichen Trials!")
        print("=" * 80)
        return

    print(f"Beste val_mae:      {study.best_value:.4f}")
    print(f"Bestes Trial:       {study.best_trial.number}")
    print()
    print("Beste Hyperparameter:")
    print("-" * 80)
    for key, value in study.best_params.items():
        print(f"  {key:<30} = {value}")
    print()

    # Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_json = {
        "study_name": args.study_name,
        "timestamp": timestamp,
        "dataset": _dataset_name,
        "n_trials": len(study.trials),
        "n_completed": len(completed_trials),
        "elapsed_time_sec": elapsed_time,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "search_space": SEARCH_SPACE,
        "fixed_params": FIXED_ARIMA_PARAMS,
    }

    json_file = OPTUNA_BASE_DIR / f"study_{args.study_name}_{timestamp}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    df_trials = study.trials_dataframe()
    csv_file = OPTUNA_BASE_DIR / f"study_{args.study_name}_{timestamp}.csv"
    df_trials.to_csv(csv_file, index=False)

    print(f"✅ Ergebnisse gespeichert: {json_file}")
    print(f"✅ Alle Trials als CSV:  {csv_file}")
    print()

    # Top 5
    print("TOP 5 TRIALS:")
    print("-" * 80)
    cols_to_show = ["number", "value", "params_max_p", "params_max_d", "params_max_q", "state"]
    if "params_max_P" in df_trials.columns:
        cols_to_show.extend(["params_max_P", "params_max_Q", "params_max_D"])
    cols_to_show = [c for c in cols_to_show if c in df_trials.columns]

    df_top = df_trials[df_trials['value'] != float('inf')].nsmallest(5, "value")[cols_to_show]
    print(df_top.to_string(index=False))
    print()

    print("=" * 80)
    print("NÄCHSTE SCHRITTE:")
    print("-" * 80)
    print(f"1. Analyse:")
    print(f"   python -m src.evaluation.analyze_optuna_arima_trials --study-name {args.study_name}")
    print()
    print(f"2. Visualisierungen:")
    print(f"   python -m src.visualization.plot_arima_optuna_study --study-name {args.study_name}")
    print()
    print(f"3. Beste Config exportieren:")
    print(f"   python -m src.modeling.optuna_arima_export_best --study-name {args.study_name}")
    print("=" * 80)


if __name__ == "__main__":
    main()

# ============================================================================
# AUFRUF-BEISPIELE & HINWEISE
# ============================================================================
#
# WICHTIG - Datenbank löschen bei Neustart:
#   Remove-Item results/arima/optuna/booksales/arima_studies.db -Force
#   Remove-Item results/arima/optuna/walmart/arima_studies.db -Force
#
# Booksales (wissenschaftlich begründeter Search Space):
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
#   python -m src.modeling.optuna_arima --study-name arima_booksales --n-trials 20
#
# Walmart:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"
#   python -m src.modeling.optuna_arima --study-name arima_walmart --n-trials 50
#
# Hinweis: Pruning ist für ARIMA deaktiviert, da gruppenweises Training
#          keine vergleichbaren Intermediate Values liefert.