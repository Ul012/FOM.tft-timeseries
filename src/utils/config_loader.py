# src/utils/config_loader.py
"""
Lädt und validiert YAML-Konfigurationen für Trainer.

Unterstützt BEIDE Formate:
1. ALT (flach): batch_size, learning_rate, model.hidden_size, etc.
2. NEU (verschachtelt): training.batch_size, model.hidden_size, type, name, etc.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class ModelCfg:
    """Modell-spezifische Hyperparameter."""
    loss: str
    output_size: int
    hidden_size: int
    attention_head_size: int
    hidden_continuous_size: int
    dropout: float
    reduce_on_plateau_patience: int


@dataclass
class TrainerCfg:
    """Vollständige Trainer-Konfiguration."""
    # Training
    seed: int
    max_epochs: int
    batch_size: int
    learning_rate: float
    gradient_clip_val: float
    early_stopping_patience: int

    # Hardware
    accelerator: str
    devices: int

    # Dataloader
    num_workers: int
    limit_train_batches: float
    limit_val_batches: float

    # Modell
    model: ModelCfg


def load_trainer_cfg(config_path: str | Path) -> TrainerCfg:
    """
    Lädt Trainer-Config aus YAML.

    Unterstützt beide Formate:
    - ALT: Flache Struktur (batch_size, learning_rate, ...)
    - NEU: Verschachtelt (training.batch_size, type, name, ...)

    Args:
        config_path: Pfad zur YAML-Datei

    Returns:
        TrainerCfg mit allen Hyperparametern

    Raises:
        FileNotFoundError: Wenn Config nicht existiert
        KeyError: Wenn erforderliche Keys fehlen
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config nicht gefunden: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Erkenne Format
    is_new_format = "training" in raw and "type" in raw

    if is_new_format:
        # NEUES Format: training.*, model.*, type, name, etc.
        training = raw["training"]
        model_dict = raw["model"]

        cfg = TrainerCfg(
            # Training
            seed=training["seed"],
            max_epochs=training["max_epochs"],
            batch_size=training["batch_size"],
            learning_rate=training["learning_rate"],
            gradient_clip_val=training["gradient_clip_val"],
            early_stopping_patience=training["early_stopping_patience"],

            # Hardware
            accelerator=training["accelerator"],
            devices=training["devices"],

            # Dataloader
            num_workers=training["num_workers"],
            limit_train_batches=training["limit_train_batches"],
            limit_val_batches=training["limit_val_batches"],

            # Modell
            model=ModelCfg(
                loss=model_dict["loss"],
                output_size=model_dict["output_size"],
                hidden_size=model_dict["hidden_size"],
                attention_head_size=model_dict["attention_head_size"],
                hidden_continuous_size=model_dict["hidden_continuous_size"],
                dropout=model_dict["dropout"],
                reduce_on_plateau_patience=model_dict["reduce_on_plateau_patience"],
            ),
        )
    else:
        # ALTES Format: flache Struktur
        model_dict = raw["model"]

        cfg = TrainerCfg(
            # Training
            seed=raw["seed"],
            max_epochs=raw["max_epochs"],
            batch_size=raw["batch_size"],
            learning_rate=raw["learning_rate"],
            gradient_clip_val=raw["gradient_clip_val"],
            early_stopping_patience=raw["early_stopping_patience"],

            # Hardware
            accelerator=raw["accelerator"],
            devices=raw["devices"],

            # Dataloader
            num_workers=raw["num_workers"],
            limit_train_batches=raw["limit_train_batches"],
            limit_val_batches=raw["limit_val_batches"],

            # Modell
            model=ModelCfg(
                loss=model_dict["loss"],
                output_size=model_dict["output_size"],
                hidden_size=model_dict["hidden_size"],
                attention_head_size=model_dict["attention_head_size"],
                hidden_continuous_size=model_dict["hidden_continuous_size"],
                dropout=model_dict["dropout"],
                reduce_on_plateau_patience=model_dict["reduce_on_plateau_patience"],
            ),
        )

    return cfg