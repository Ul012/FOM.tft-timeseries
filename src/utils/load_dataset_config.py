# src/utils/load_dataset_config.py
"""Lädt Dataset-Config aus YAML - für Pipeline UND Standalone."""

import os
from pathlib import Path
import yaml
from typing import Any


def load_dataset_config(yaml_path: str | Path = None) -> dict[str, Any]:
    """
    Lädt Dataset-Config aus YAML.

    Reihenfolge:
    1. Übergebener yaml_path Parameter
    2. Umgebungsvariable DATASET_CONFIG
    3. Fehler (kein Default!)

    Args:
        yaml_path: Pfad zur Dataset-YAML (optional)

    Returns:
        Dict mit allen Config-Werten
    """
    # Priorität 1: Übergebener Parameter
    if yaml_path is not None:
        path = Path(yaml_path)
    # Priorität 2: Umgebungsvariable
    elif "DATASET_CONFIG" in os.environ:
        path = Path(os.environ["DATASET_CONFIG"])
    # Kein Default!
    else:
        raise ValueError(
            "Keine Dataset-Config angegeben!\n"
            "Bitte setzen:\n"
            "  1. Umgebungsvariable: $env:DATASET_CONFIG='configs/datasets/walmart.yaml'\n"
            "  2. Oder in Pipeline:   --dataset configs/datasets/walmart.yaml"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset-Config nicht gefunden: {path}\n"
            f"Gesucht in: {path.resolve()}"
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_schema(config: dict) -> dict:
    """Extrahiert Schema-Werte."""
    return config["schema"]


def get_preprocessing_params(config: dict, step_name: str) -> dict:
    """Holt Parameter für einen Preprocessing-Step."""
    for step in config["preprocessing"]:
        if step["step"] == step_name:
            return step.get("params", {})
    return {}


def get_tft_config(config: dict) -> dict:
    """Extrahiert TFT-spezifische Config."""
    return config["tft"]


def get_split_config(config: dict) -> dict:
    """Extrahiert Split-Config."""
    return config["split"]