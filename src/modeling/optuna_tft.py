# src/modeling/optuna_tft.py
"""
Hyperparameter-Optimierung für TFT mit Optuna.

METRIKEN:
- Primäre Metrik (Optimierungsziel): val_mae (Mean Absolute Error)
- Geloggte Metriken: MAE, RMSE, MAPE, SMAPE
- Early Stopping basiert auf: val_mae
- Model Checkpoint basiert auf: val_mae

KONFIGURATION:
- Alle Parameter sind im Script hardcodiert
- Search Space basiert auf Vorwissen aus baseline-Experimenten
- Training-Parameter entsprechen baseline.yaml

Aufrufbeispiele:
    # Test-Run
    python -m src.modeling.optuna_tft --study-name tft_test --n-trials 1
    (Epochen vorher auf 2 setzen)

    # Einfacher Run
    python -m src.modeling.optuna_tft --n-trials 10

    # Mit Custom Study Name
    python -m src.modeling.optuna_tft --study-name tft_day --n-trials 20

    # Fortsetzen einer existierenden Study
    python -m src.modeling.optuna_tft --study-name tft_seminar --n-trials 5
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import os
import yaml

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE, QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer

from src.config import BASE_DIR
from src.utils.load_dataset_config import get_schema, load_dataset_config

# ============================================================================
# GLOBALE KONFIGURATION
# ============================================================================

# Dataset-Config laden
_dataset_config = load_dataset_config()
_schema = get_schema(_dataset_config)
TARGET_COL = _schema["target_col"]
ID_COLS = _schema["id_cols"]
TIME_COL = _schema["time_col"]

# Dataset-Name für Pfade
_dataset_name = _dataset_config["name"]

# Pfade (dataset-spezifisch)
OPTUNA_BASE_DIR = BASE_DIR / "results" / "tft" / "optuna" / _dataset_name
OPTUNA_STORAGE = f"sqlite:///{OPTUNA_BASE_DIR}/tft_studies.db"

# ============================================================================
# HARDCODIERTE PARAMETER
# ============================================================================

# Search Space (basierend auf baseline-Experimenten)
SEARCH_SPACE = {
    "learning_rate": {"min": 0.0003, "max": 0.003, "log": True},
    "batch_size": {"choices": [64, 128, 256]},
    "hidden_size": {"choices": [32, 64, 96]},
    "attention_head_size": {"min": 2, "max": 4},
    "dropout": {"min": 0.05, "max": 0.20},
    "hidden_continuous_size": {"choices": [16, 24, 32, 48]},
    "gradient_clip_val": {"min": 0.08, "max": 0.15},
}

# Training-Parameter (feste Werte für alle Trials)
TRAINING_CONFIG = {
    "seed": 42,
    "max_epochs": 30, # für Testing 2, sonst 30
    "num_workers": 4,
    "accelerator": "gpu",  # Auf "cpu" ändern falls keine GPU
    "devices": 1,
    "log_every_n_steps": 50,
    "early_stopping_patience": 4,
}

# Optuna-Einstellungen
OPTUNA_CONFIG = {
    "study_name_default": "tft_hpo",
    "n_trials_default": 10,
    "pruner": {
        "n_startup_trials": 3,  # Erste 3 Trials komplett durchlaufen
        "n_warmup_steps": 5,  # Erste 5 Epochen nicht prunen
        "interval_steps": 1,  # Jede Epoche prüfen
    },
}


def save_search_space_config(study_name: str) -> Path:
    """
    Speichert Search Space und Training Config als YAML für Reproduzierbarkeit.
    """
    config = {
        "study_name": study_name,
        "dataset": _dataset_name,
        "timestamp": datetime.now().isoformat(),
        "search_space": SEARCH_SPACE,
        "training_config": TRAINING_CONFIG,
        "optuna_config": OPTUNA_CONFIG,
    }

    OPTUNA_BASE_DIR.mkdir(parents=True, exist_ok=True)
    config_path = OPTUNA_BASE_DIR / f"{study_name}_search_space.yaml"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[optuna] Search Space gespeichert: {config_path}")
    return config_path

# ============================================================================
# DATASET-FUNKTIONEN
# ============================================================================


def _load_dataset_from_spec(processed_dir: Path) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
    """
    Lädt train/val Parquet anhand der dataset_spec.json.

    Args:
        processed_dir: Pfad zum processed-Verzeichnis

    Returns:
        (train_ds, val_ds)

    Raises:
        FileNotFoundError: Wenn dataset_spec.json oder Parquet-Dateien fehlen
        KeyError: Wenn erforderliche Keys fehlen
    """
    spec_path = processed_dir / "dataset_spec.json"

    if not spec_path.exists():
        raise FileNotFoundError(f"dataset_spec.json nicht gefunden: {spec_path}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    train_pq = Path(spec["paths"]["train"])
    val_pq = Path(spec["paths"]["val"])

    if not train_pq.exists():
        raise FileNotFoundError(f"Train-Parquet nicht gefunden: {train_pq}")
    if not val_pq.exists():
        raise FileNotFoundError(f"Val-Parquet nicht gefunden: {val_pq}")

    max_encoder_length = spec["lengths"]["max_encoder_length"]
    max_prediction_length = spec["lengths"]["max_prediction_length"]

    feature_lists = spec["feature_lists"]
    static_categoricals = feature_lists["static_categoricals"]
    time_varying_known_reals = feature_lists["time_varying_known_reals"]
    time_varying_unknown_reals = feature_lists["time_varying_unknown_reals"]
    time_varying_known_categoricals = feature_lists["time_varying_known_categoricals"]

    df_train = pd.read_parquet(train_pq)
    df_val = pd.read_parquet(val_pq)

    # NaN-Check für Lag-Features (Imputation erfolgt in lag_features.py)
    lag_cols = [col for col in df_train.columns if col.startswith("lag_")]
    if lag_cols:
        train_nans = df_train[lag_cols].isna().sum().sum()
        val_nans = df_val[lag_cols].isna().sum().sum()
        if train_nans > 0 or val_nans > 0:
            raise ValueError(
                f"NaN in Lag-Features gefunden (Preprocessing-Bug)! "
                f"Train: {train_nans}, Val: {val_nans}"
            )

    # Zielvariable auf float32 casten
    for df in (df_train, df_val):
        if TARGET_COL in df.columns:
            df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("float32")

    # time_idx muss vorhanden sein
    if "time_idx" not in df_train.columns:
        raise KeyError("Spalte 'time_idx' fehlt in Train-Daten")
    time_idx_col = "time_idx"

    train_ds = TimeSeriesDataSet(
        df_train,
        time_idx=time_idx_col,
        target=TARGET_COL,
        group_ids=ID_COLS,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        target_normalizer=GroupNormalizer(groups=ID_COLS, transformation="softplus"),
        allow_missing_timesteps=True,
    )

    val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_val, predict=False)

    return train_ds, val_ds


# ============================================================================
# OPTUNA OBJECTIVE FUNCTION
# ============================================================================


def objective(trial: optuna.Trial) -> float:
    """
    Optuna Objective Function für TFT-Hyperparameter-Optimierung.

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
    learning_rate = trial.suggest_float(
        "learning_rate",
        SEARCH_SPACE["learning_rate"]["min"],
        SEARCH_SPACE["learning_rate"]["max"],
        log=SEARCH_SPACE["learning_rate"]["log"]
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        SEARCH_SPACE["batch_size"]["choices"]
    )

    hidden_size = trial.suggest_categorical(
        "hidden_size",
        SEARCH_SPACE["hidden_size"]["choices"]
    )

    attention_head_size = trial.suggest_int(
        "attention_head_size",
        SEARCH_SPACE["attention_head_size"]["min"],
        SEARCH_SPACE["attention_head_size"]["max"]
    )

    dropout = trial.suggest_float(
        "dropout",
        SEARCH_SPACE["dropout"]["min"],
        SEARCH_SPACE["dropout"]["max"]
    )

    hidden_continuous_size = trial.suggest_categorical(
        "hidden_continuous_size",
        SEARCH_SPACE["hidden_continuous_size"]["choices"]
    )

    gradient_clip_val = trial.suggest_float(
        "gradient_clip_val",
        SEARCH_SPACE["gradient_clip_val"]["min"],
        SEARCH_SPACE["gradient_clip_val"]["max"]
    )

    # Feste Parameter
    max_epochs = TRAINING_CONFIG["max_epochs"]
    seed = TRAINING_CONFIG["seed"]
    num_workers = TRAINING_CONFIG["num_workers"]
    accelerator = TRAINING_CONFIG["accelerator"]
    devices = TRAINING_CONFIG["devices"]

    # -------------------------------------------------------------------------
    # 2. Reproduzierbarkeit
    # -------------------------------------------------------------------------
    pl.seed_everything(seed, workers=True)

    # Best-Effort Reproduzierbarkeit (GPU-Training ist nie 100% deterministisch)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    # -------------------------------------------------------------------------
    # 3. Datasets laden
    # -------------------------------------------------------------------------
    dataset_name = _dataset_config["name"]
    dataset_processed_dir = BASE_DIR / "data" / "processed" / dataset_name

    train_ds, val_ds = _load_dataset_from_spec(dataset_processed_dir)

    train_loader = train_ds.to_dataloader(
        train=True, batch_size=batch_size, num_workers=num_workers
    )
    val_loader = val_ds.to_dataloader(
        train=False, batch_size=batch_size, num_workers=num_workers
    )

    # -------------------------------------------------------------------------
    # 4. Modell erstellen
    # -------------------------------------------------------------------------
    loss_fn = QuantileLoss()
    output_size = 7

    logging_metrics = [MAE(), RMSE(), MAPE(), SMAPE()]

    model = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_continuous_size,
        output_size=output_size,
        loss=loss_fn,
        log_interval=10,
        reduce_on_plateau_patience=4,
        logging_metrics=logging_metrics,
    )

    # -------------------------------------------------------------------------
    # 5. Callbacks
    # -------------------------------------------------------------------------
    trial_id = f"trial_{trial.number:04d}"
    study_name = trial.study.study_name

    ckpt_dir = OPTUNA_BASE_DIR / trial_id / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = BASE_DIR / "logs" / "tft" / "optuna" / _dataset_name / study_name / trial_id
    logs_dir.mkdir(parents=True, exist_ok=True)

    early_stop = EarlyStopping(
        monitor="val_MAE",
        patience=TRAINING_CONFIG["early_stopping_patience"],
        mode="min",
        verbose=False,
    )

    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_MAE",
        mode="min",
        save_top_k=1,
        save_last=False,
        verbose=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Optuna Pruning Callback
    pruning_callback = optuna.integration.PyTorchLightningPruningCallback(
        trial, monitor="val_MAE"
    )

    # -------------------------------------------------------------------------
    # 6. Logger
    # -------------------------------------------------------------------------
    logger = CSVLogger(
        save_dir=logs_dir.parent,
        name=trial_id,
        version="",
    )

    # -------------------------------------------------------------------------
    # 7. Trainer
    # -------------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=gradient_clip_val,
        callbacks=[early_stop, checkpoint, lr_monitor, pruning_callback],
        accelerator=accelerator,
        devices=devices,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        log_every_n_steps=TRAINING_CONFIG["log_every_n_steps"],
        logger=logger,
        enable_progress_bar=True,  # Progress Bar aktiviert
        enable_model_summary=True,  # Zeigt Modell-Info
    )

    # -------------------------------------------------------------------------
    # 8. Training
    # -------------------------------------------------------------------------
    print(f"\n[Trial {trial.number}] Training startet mit:")
    print(f"  - learning_rate: {learning_rate:.6f}")
    print(f"  - batch_size: {batch_size}")
    print(f"  - hidden_size: {hidden_size}")
    print(f"  - dropout: {dropout:.3f}")
    print()

    trainer.fit(model, train_loader, val_loader)

    # Beste val_MAE extrahieren
    if "val_MAE" not in trainer.callback_metrics:
        raise KeyError("val_MAE nicht in callback_metrics gefunden")

    val_mae = float(trainer.callback_metrics["val_MAE"].item())

    # Trial-Metadaten speichern
    trial_meta = {
        "trial_id": trial_id,
        "trial_number": trial.number,
        "val_mae": val_mae,
        "val_loss": float(trainer.callback_metrics["val_loss"].item()),
        "epochs_trained": trainer.current_epoch + 1,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "hidden_size": hidden_size,
            "attention_head_size": attention_head_size,
            "dropout": dropout,
            "hidden_continuous_size": hidden_continuous_size,
            "gradient_clip_val": gradient_clip_val,
        },
    }

    trial_json_path = ckpt_dir.parent / "trial_summary.json"
    with open(trial_json_path, "w") as f:
        json.dump(trial_meta, f, indent=2)

    return val_mae


# ============================================================================
# MAIN
# ============================================================================


def main():
    # -------------------------------------------------------------------------
    # CLI-Argumente
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="TFT Hyperparameter-Optimierung mit Optuna"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=OPTUNA_CONFIG["study_name_default"],
        help=f"Name der Optuna Study (default: {OPTUNA_CONFIG['study_name_default']})",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=OPTUNA_CONFIG["n_trials_default"],
        help=f"Anzahl Optuna Trials (default: {OPTUNA_CONFIG['n_trials_default']})",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Anzahl paralleler Trials (1 = sequentiell)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Maximale Zeit in Sekunden (None = kein Limit)",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Optuna Study erstellen/laden
    # -------------------------------------------------------------------------
    OPTUNA_BASE_DIR.mkdir(parents=True, exist_ok=True)
    storage_url = OPTUNA_STORAGE

    print("=" * 80)
    print("OPTUNA HYPERPARAMETER-OPTIMIERUNG FÜR TFT")
    print("=" * 80)
    print()
    print(f"Study Name:    {args.study_name}")
    print(f"Trials:        {args.n_trials}")
    print(f"Parallele Jobs: {args.n_jobs}")
    print(f"Storage:       {storage_url}")
    print()
    print("KONFIGURATION:")
    print(f"  - Seed:               {TRAINING_CONFIG['seed']}")
    print(f"  - Max Epochs/Trial:   {TRAINING_CONFIG['max_epochs']}")
    print(f"  - Accelerator:        {TRAINING_CONFIG['accelerator']}")
    print(f"  - Early Stop Patience: {TRAINING_CONFIG['early_stopping_patience']}")
    print()
    print("PRIMÄRE METRIK:")
    print("  - val_MAE (Mean Absolute Error)")
    print()
    print("GELOGGTE METRIKEN:")
    print("  - MAE, RMSE, MAPE, SMAPE")
    print()

    # Search Space speichern für Reproduzierbarkeit
    save_search_space_config(args.study_name)

    # Study erstellen
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",  # Minimiere val_mae
        sampler=TPESampler(seed=TRAINING_CONFIG["seed"]),
        pruner=MedianPruner(
            n_startup_trials=OPTUNA_CONFIG["pruner"]["n_startup_trials"],
            n_warmup_steps=OPTUNA_CONFIG["pruner"]["n_warmup_steps"],
            interval_steps=OPTUNA_CONFIG["pruner"]["interval_steps"],
        ),
    )

    print("Study-Status:")
    print(f"  - Bisherige Trials: {len(study.trials)}")

    # Nur beste val_MAE anzeigen wenn es erfolgreich abgeschlossene Trials gibt
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        print(f"  - Abgeschlossene Trials: {len(completed_trials)}")
        print(f"  - Beste val_MAE:    {study.best_value:.4f}")

    print()
    print("=" * 80)
    print()

    # -------------------------------------------------------------------------
    # Optimierung starten
    # -------------------------------------------------------------------------
    start_time = time.perf_counter()

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    elapsed_time = time.perf_counter() - start_time

    # -------------------------------------------------------------------------
    # Ergebnisse ausgeben
    # -------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("OPTIMIERUNG ABGESCHLOSSEN")
    print("=" * 80)
    print()
    print(f"Gesamt-Zeit:        {elapsed_time / 60:.1f} Minuten")
    print(f"Durchgeführte Trials: {len(study.trials)}")
    print(f"Geprunte Trials:    {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print()

    print("BESTE HYPERPARAMETER:")
    print("-" * 80)
    for key, value in study.best_params.items():
        print(f"  {key:<25} = {value}")
    print()
    print(f"Beste val_MAE: {study.best_value:.4f}")
    print()

    # -------------------------------------------------------------------------
    # Ergebnisse speichern
    # -------------------------------------------------------------------------
    OPTUNA_BASE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = OPTUNA_BASE_DIR / f"study_{args.study_name}_{timestamp}.json"

    results = {
        "study_name": args.study_name,
        "timestamp": timestamp,
        "n_trials": len(study.trials),
        "elapsed_time_sec": elapsed_time,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "training_config": TRAINING_CONFIG,
        "search_space": SEARCH_SPACE,
    }

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Ergebnisse gespeichert: {results_file}")
    print()

    # -------------------------------------------------------------------------
    # Top 5 Trials anzeigen
    # -------------------------------------------------------------------------
    print("TOP 5 TRIALS:")
    print("-" * 80)
    df_trials = study.trials_dataframe()
    df_top = df_trials.nsmallest(5, "value")[
        ["number", "value", "params_learning_rate", "params_batch_size", "params_hidden_size", "state"]
    ]
    print(df_top.to_string(index=False))
    print()

    # CSV Export
    csv_file = OPTUNA_BASE_DIR / f"study_{args.study_name}_{timestamp}.csv"
    df_trials.to_csv(csv_file, index=False)
    print(f"Alle Trials als CSV: {csv_file}")
    print()

    print("=" * 80)
    print()
    print("NÄCHSTE SCHRITTE:")
    print("-" * 80)
    print("1. Beste Config als YAML exportieren:")
    print(f"   python -m src.modeling.optuna_export_best --study-name {args.study_name}")
    print()
    print("2. Visualisierungen erstellen:")
    print("   python -m src.visualization.plot_optuna_study")
    print()
    print("3. Finales Training mit besten Parametern:")
    print("   python -m src.pipeline --model configs/models/tft/optuna_tft_day_best.yaml")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()

# Aufruf:
# $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.modeling.optuna_tft --study-name tft_quicktest --n-trials 1
# $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.modeling.optuna_tft --study-name tft_newyear --n-trials 20
# $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.modeling.optuna_tft --study-name walmart_full --n-trials 20