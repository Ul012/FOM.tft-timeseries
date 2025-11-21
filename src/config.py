# src/config.py
# ============================================================================
# Projekt-übergreifende Konfiguration
# ============================================================================
# Enthält NUR Parameter, die für ALLE Datensätze gleich sind:
# - Verzeichnis-Struktur
# - Evaluation-Metriken
#
# Dataset-spezifische Parameter (Schema, Features, Split) gehören in:
# configs/datasets/<dataset_name>.yaml
# ============================================================================

from pathlib import Path

# -----------------------------------------------------------------------------
# Verzeichnisse (Projekt-Struktur)
# -----------------------------------------------------------------------------
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# -----------------------------------------------------------------------------
# Evaluation-Metriken (modell-übergreifend)
# -----------------------------------------------------------------------------
# Diese Metriken werden für ALLE Modelle gleich berechnet
EVALUATION_METRICS: list[str] = ["mae", "rmse", "mape", "smape", "rs"]
EVALUATION_SPLITS: list[str] = ["val", "test"]

# Beschreibung der Metriken (optional, für Dokumentation)
METRIC_DESCRIPTIONS: dict[str, str] = {
    "mae": "Mean Absolute Error",
    "rmse": "Root Mean Squared Error",
    "mape": "Mean Absolute Percentage Error",
    "smape": "Symmetric Mean Absolute Percentage Error",
    "r2": "R² Score (Coefficient of Determination)",
}