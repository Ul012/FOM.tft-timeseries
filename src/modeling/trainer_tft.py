# src/modeling/trainer_tft.py
"""
Trainiert einen Temporal Fusion Transformer (TFT) ausschließlich gesteuert über eine YAML-Konfiguration.
Keinerlei Hyperparameter-Fallbacks im Code – alles kommt aus der YAML.

Aufrufbeispiele:
    python -m src.modeling.trainer_tft --config configs/trainer_tft_baseline.yaml
    python -m src.modeling.trainer_tft  # nutzt Default-Pfad unten
"""

from __future__ import annotations

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


def _load_dataset_from_spec(processed_dir: Path):
    """
    Lädt train/val Parquet anhand der dataset_spec.json und baut TimeSeriesDataSet-Objekte.
    Nutzt – wie die alte x_-Version – die tatsächlichen Pfade aus der JSON.
    """
    base = processed_dir / "model_dataset" / "tft"
    spec_path = base / "dataset_spec.json"

    if not spec_path.exists():
        raise FileNotFoundError(f"dataset_spec.json nicht gefunden: {spec_path}")

    # <-- Änderung: Pfade aus der JSON nutzen
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    train_pq = Path(spec["paths"]["train"])
    val_pq   = Path(spec["paths"]["val"])

    if not train_pq.exists() or not val_pq.exists():
        raise FileNotFoundError(f"Parquet-Dateien nicht gefunden: {train_pq} oder {val_pq}")

    max_encoder_length = spec["lengths"]["max_encoder_length"]
    max_prediction_length = spec["lengths"]["max_prediction_length"]

    df_train = pd.read_parquet(train_pq)
    df_val   = pd.read_parquet(val_pq)

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

    loss_fn = QuantileLoss()
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
    run_id = f"run_{datetime.now():%Y%m%d_%H%M%S}"

    ckpt_dir = PROCESSED_DIR / "model_dataset" / "tft" / "checkpoints" / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    early_stop = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=cfg.early_stopping_patience,
        mode="min",
    )

    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="tft-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,                  # nur bester pro Lauf; alternativ: -1 + every_n_epochs=1
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")  # oder "epoch", wenn dir das lieber ist

    logger = CSVLogger(
        save_dir="logs",               # EIN Log-Wurzelordner
        name="tft",
        version=run_id,                # logs/tft/run_YYYYMMDD_HHMMSS/
    )

    print(f"[trainer_tft] Run-ID: {run_id}")
    print(f"[trainer_tft] Checkpoints: {ckpt_dir}")
    print(f"[trainer_tft] Logs: {(Path('logs') / 'tft' / run_id).resolve()}")
    print(f"[trainer_tft] Logs: {Path('logs') / 'tft' / run_id}")

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
    # Training
    # -----------------------------
    trainer.fit(model, train_loader, val_loader)

    # Hinweis auf bestes Checkpoint
    if checkpoint.best_model_path:
        print(f"[trainer_tft] Bestes Checkpoint: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()

# python -m src.modeling.trainer_tft --config configs/trainer_tft_baseline.yaml