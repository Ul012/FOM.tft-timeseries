# src/utils/load_dataset_config.py
"""Lädt Dataset-Config aus YAML - für Pipeline UND Standalone."""

from pathlib import Path
import yaml
from typing import Any


def load_dataset_config(yaml_path: str | Path = "configs/datasets/booksales.yaml") -> dict[str, Any]:
    """
    Lädt Dataset-Config aus YAML.

    Args:
        yaml_path: Pfad zur Dataset-YAML (default: booksales.yaml)

    Returns:
        Dict mit allen Config-Werten
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset-Config nicht gefunden: {path}")

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