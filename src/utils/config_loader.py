# src/utils/config_loader.py
# Schlanker, strikter YAML-Loader ohne Fallbacks.

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Dict, Any
import yaml


@dataclass(frozen=True)
class ModelCfg:
    loss: Literal["quantile", "mse"]
    hidden_size: int
    attention_head_size: int
    dropout: float
    hidden_continuous_size: int
    output_size: int
    reduce_on_plateau_patience: int


@dataclass(frozen=True)
class TrainerCfg:
    seed: int
    max_epochs: int
    batch_size: int
    learning_rate: float
    gradient_clip_val: float
    early_stopping_patience: int
    num_workers: int
    accelerator: Literal["cpu", "gpu"]
    devices: int
    limit_train_batches: float | int
    limit_val_batches: float | int
    model: ModelCfg


def _fail_if_extra_keys(loaded: Dict[str, Any], schema_keys: set[str], ctx: str) -> None:
    extras = set(loaded.keys()) - schema_keys
    if extras:
        raise KeyError(f"Unerwartete Schlüssel in {ctx}: {sorted(extras)}")


def load_trainer_cfg(path: str | Path) -> TrainerCfg:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {p}")

    try:
        cfg: Dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Konfiguration konnte nicht geladen werden: {e}")

    # Harte Validierung: nur erlaubte Keys
    allowed_top = {
        "seed", "max_epochs", "batch_size", "learning_rate", "gradient_clip_val",
        "early_stopping_patience", "num_workers", "accelerator", "devices",
        "limit_train_batches", "limit_val_batches", "model"
    }
    _fail_if_extra_keys(cfg, allowed_top, "trainer-config")

    if "model" not in cfg:
        raise KeyError("trainer-config: Schlüssel 'model' fehlt.")
    m = cfg["model"]
    allowed_model = {
        "loss", "hidden_size", "attention_head_size", "dropout",
        "hidden_continuous_size", "output_size", "reduce_on_plateau_patience"
    }
    _fail_if_extra_keys(m, allowed_model, "trainer-config.model")

    # Typisiertes Objekt bauen
    return TrainerCfg(
        seed=int(cfg["seed"]),
        max_epochs=int(cfg["max_epochs"]),
        batch_size=int(cfg["batch_size"]),
        learning_rate=float(cfg["learning_rate"]),
        gradient_clip_val=float(cfg["gradient_clip_val"]),
        early_stopping_patience=int(cfg["early_stopping_patience"]),
        num_workers=int(cfg["num_workers"]),
        accelerator=str(cfg["accelerator"]),
        devices=int(cfg["devices"]),
        limit_train_batches=cfg["limit_train_batches"],
        limit_val_batches=cfg["limit_val_batches"],
        model=ModelCfg(
            loss=str(m["loss"]),
            hidden_size=int(m["hidden_size"]),
            attention_head_size=int(m["attention_head_size"]),
            dropout=float(m["dropout"]),
            hidden_continuous_size=int(m["hidden_continuous_size"]),
            output_size=int(m["output_size"]),
            reduce_on_plateau_patience=int(m["reduce_on_plateau_patience"]),
        ),
    )

# python -m src.modeling.trainer_tft --config configs/trainer_tft_baseline.yaml