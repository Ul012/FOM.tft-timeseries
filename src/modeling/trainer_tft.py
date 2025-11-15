# src/modeling/trainer_tft.py
"""
Trainiert einen Temporal Fusion Transformer (TFT) ausschließlich gesteuert über eine YAML-Konfiguration.
Keinerlei Hyperparameter-Fallbacks im Code – alles kommt aus der YAML.

Aufrufbeispiele:
    python -m src.modeling.trainer_tft --config configs/trainer_tft_baseline.yaml
    python -m src.modeling.trainer_tft  # nutzt Default-Pfad unten
"""

from __future__ import annotations

import time
import yaml

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE, MAPE, SMAPE

# Projektweite, statische Konstanten (Pfade, Spalten) – NICHT Hyperparameter:
from src.config import (
    PROCESSED_DIR,
    TARGET_COL,
    ID_COLS,
    TIME_COL,
)

# Strikter YAML-Loader (liefert typisierte cfg ohne Fallbacks)
from src.utils.config_loader import load_trainer_cfg
from src.utils.json_results import export_run_jsons_from_metrics


def _load_dataset_from_spec(processed_dir: Path):
    """
    Lädt train/val Parquet anhand der dataset_spec.json und baut TimeSeriesDataSet-Objekte.
    Nutzt die Pfade aus der JSON-Spezifikation.
    """
    spec_path = processed_dir / "dataset_spec.json"

    if not spec_path.exists():
        raise FileNotFoundError(f"dataset_spec.json nicht gefunden: {spec_path}")

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    train_pq = Path(spec["paths"]["train"])
    val_pq = Path(spec["paths"]["val"])

    if not train_pq.exists() or not val_pq.exists():
        raise FileNotFoundError(f"Parquet-Dateien nicht gefunden: {train_pq} oder {val_pq}")

    max_encoder_length = spec["lengths"]["max_encoder_length"]
    max_prediction_length = spec["lengths"]["max_prediction_length"]

    df_train = pd.read_parquet(train_pq)
    df_val = pd.read_parquet(val_pq)

    # Zielvariable auf float32 casten
    for df in (df_train, df_val):
        if TARGET_COL in df.columns:
            df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("float32")

    time_idx_col = "time_idx" if "time_idx" in df_train.columns else TIME_COL
    train_ds = TimeSeriesDataSet(
        df_train,
        time_idx=time_idx_col,
        target=TARGET_COL,
        group_ids=ID_COLS,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        target_normalizer=GroupNormalizer(groups=ID_COLS, transformation="softplus"),
    )

    val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_val, predict=False)
    return train_ds, val_ds


def main():
    # -----------------------------
    # CLI: Pfad zur YAML
    # -----------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=str,
        default="configs/trainer_tft_baseline.yaml",
        help="Pfad zur YAML-Konfiguration (ohne Fallbacks).",
    )
    args = ap.parse_args()

    # -----------------------------
    # YAML laden (strikt, ohne Fallbacks)
    # -----------------------------
    cfg = load_trainer_cfg(args.config)

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    # -----------------------------
    # Determinismus / Reproduzierbarkeit
    # -----------------------------
    pl.seed_everything(cfg.seed, workers=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # -----------------------------
    # Datasets + Dataloader
    # -----------------------------
    train_ds, val_ds = _load_dataset_from_spec(PROCESSED_DIR)

    train_loader = train_ds.to_dataloader(
        train=True, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )
    val_loader = val_ds.to_dataloader(
        train=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers
    )

    # -----------------------------
    # Modell aus YAML-Parametern
    # -----------------------------
    # Loss bestimmen
    if cfg.model.loss == "mse":
        loss_fn = torch.nn.MSELoss()
        output_size = 1
    else:
        loss_fn = QuantileLoss()  # Quantile werden intern am output_size festgelegt
        output_size = cfg.model.output_size

    from pytorch_forecasting.models import TemporalFusionTransformer

    logging_metrics = [MAE(), RMSE(), MAPE(), SMAPE()]

    model = TemporalFusionTransformer.from_dataset(
        train_ds,
        learning_rate=cfg.learning_rate,
        loss=loss_fn,
        logging_metrics=logging_metrics,
        hidden_size=cfg.model.hidden_size,
        attention_head_size=cfg.model.attention_head_size,
        dropout=cfg.model.dropout,
        hidden_continuous_size=cfg.model.hidden_continuous_size,
        output_size=output_size,
        reduce_on_plateau_patience=cfg.model.reduce_on_plateau_patience,
    )

    # -----------------------------
    # Run-ID, Checkpoints und Logger
    # -----------------------------
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Konfigurationsname als Suffix
    cfg_stem = config_path.stem
    suffix = cfg_stem.replace("trainer_tft_", "") or cfg_stem

    run_id = f"run_{ts_str}_{suffix}"

    # NEU: ein gemeinsamer Run-Ordner
    run_dir = Path("results") / "tft" / run_id  # <<< geändert
    run_dir.mkdir(parents=True, exist_ok=True)

    # NEU: Checkpoints im Run-Ordner
    ckpt_dir = run_dir / "checkpoints"  # <<< geändert
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=cfg.early_stopping_patience,
        mode="min",
    )

    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),  # <<< geändert
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    logger = CSVLogger(
        save_dir="logs",
        name="tft",
        version=run_id,
    )

    print(f"[trainer_tft] Run-ID: {run_id}")
    print(f"[trainer_tft] Checkpoints: {ckpt_dir}")
    print(f"[trainer_tft] Logs: {(Path('logs') / 'tft' / run_id).resolve()}")

    # -----------------------------
    # Trainer (alle Werte aus cfg)
    # -----------------------------
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        callbacks=[early_stop, checkpoint, lr_monitor],
        accelerator=cfg.accelerator,   # "cpu" | "gpu" – explizit aus YAML
        devices=cfg.devices,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        log_every_n_steps=50,
        enable_progress_bar=True,
        logger=logger,
    )

    # Optionale Protokollierung der Hyperparameter im Logger
    logger.log_hyperparams({
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "gradient_clip_val": cfg.gradient_clip_val,
        "num_workers": cfg.num_workers,
        "accelerator": cfg.accelerator,
        "devices": cfg.devices,
        "limit_train_batches": cfg.limit_train_batches,
        "limit_val_batches": cfg.limit_val_batches,
        **{f"model.{k}": v for k, v in vars(cfg.model).items()},
    })

    # -----------------------------
    # Zeitmessung
    # -----------------------------
    tfit_start = time.perf_counter()

    # -----------------------------
    # Training
    # -----------------------------
    trainer.fit(model, train_loader, val_loader)

    # Fit-Zeit stoppen
    fit_time_sec = round(time.perf_counter() - tfit_start, 2)
    epochs_trained = int(trainer.current_epoch) + 1  # 0-basiert -> +1

    # Meta-Infos für summary.json zusammenstellen
    meta = {
        "seed": cfg_dict.get("seed"),
        "config_file": str(config_path),
        "config_values": cfg_dict,  # komplette YAML als normales Dict
        "fit_time_sec": fit_time_sec,
        "epochs_trained": epochs_trained,
        "avg_epoch_time_sec": round(fit_time_sec / max(1, epochs_trained), 2),

        # wichtige Hparams aus dem YAML-Dict (primitives!)
        "learning_rate": cfg_dict.get("learning_rate"),
        "batch_size": cfg_dict.get("batch_size"),
        "max_epochs": cfg_dict.get("max_epochs"),
        "early_stopping_patience": cfg_dict.get("early_stopping_patience"),
        "gradient_clip_val": cfg_dict.get("gradient_clip_val"),
        "accelerator": cfg_dict.get("accelerator"),
        "devices": cfg_dict.get("devices"),
        "model": cfg_dict.get("model"),  # <-- jetzt als Dict, nicht als ModelCfg-Objekt
    }
    try:
        meta["best_checkpoint_path"] = str(checkpoint.best_model_path)
    except NameError:
        pass

    logs_run_dir = Path(logger.log_dir)  # z. B. logs/tft/run_YYYYMMDD_HHMMSS

    # NEU: Evaluation + summary.json im selben Run-Ordner
    results_dir = run_dir

    results_path, summary_path = export_run_jsons_from_metrics(
        run_id=run_id,
        logs_run_dir=logs_run_dir,
        results_dir=results_dir,
        meta=meta,
    )

    print("[trainer_tft] JSON-Export abgeschlossen:")
    print(f"  - {results_path}")
    print(f"  - {summary_path}")

    # Hinweis auf bestes Checkpoint
    if checkpoint.best_model_path:
        print(f"[trainer_tft] Bestes Checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    # python -m src.modeling.trainer_tft --config configs/trainer_tft_baseline.yaml
    # python -m src.modeling.trainer_tft --config configs/trainer_tft_bs32.yaml
    # python -m src.modeling.trainer_tft --config configs/trainer_tft_lr0007.yaml
    main()
