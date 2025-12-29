"""
Konvertiert TFT-formatierte Daten zu Prophet-Format und erstellt eine
Spezifikation für das Training.

Prophet benötigt:
- 'ds' Spalte (Datetime)
- 'y' Spalte (Target)
- Optional: Regressoren (zusätzliche Features)
- Optional: cap/floor (für logistic growth)

Input: train/val/test.parquet (aus model_dataset.py)
Output: prophet_spec.json

Aufruf:
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
    python -m src.modeling.dataset_prophet
"""

import json
from typing import List, Dict, Any

import pandas as pd

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema

_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]
_schema = get_schema(_dataset_config)

TIME_COL = _schema["time_col"]
TARGET_COL = _schema["target_col"]
ID_COLS = _schema["id_cols"]


class ProphetDatasetSpec:
    """Erstellt Prophet-Datensatz-Spezifikation."""

    def __init__(self, dataset_config: dict):
        self.config = dataset_config
        self.schema = get_schema(dataset_config)
        self.prophet_config = dataset_config.get("prophet", {})

    def identify_regressors(self, df: pd.DataFrame) -> List[str]:
        """
        Identifiziert Regressoren aus dem DataFrame.
        """
        # Start mit expliziten Regressoren aus Config
        regressors = self.prophet_config.get("regressors", [])

        # Verfügbare numerische/kategoriale Spalten
        available_cols = set(df.columns)
        numeric_cols = set(df.select_dtypes(include=["number"]).columns)

        # Ausschlussliste
        exclude = {
            TIME_COL,
            TARGET_COL,
            "time_idx",  # TFT-spezifisch
        }
        exclude.update(ID_COLS)

        # Zyklische Encodings ausschließen (cyc_*)
        exclude.update([col for col in available_cols if col.startswith("cyc_")])

        # Lag-Features ausschließen (lag_*, *_missing)
        exclude.update([col for col in available_cols if col.startswith("lag_") or col.endswith("_missing")])

        # Automatische Regressor-Erkennung
        calendar_cols = ["year", "month", "day", "dayofweek", "weekofyear", "is_weekend"]
        holiday_cols = [col for col in available_cols if col.startswith("is_holiday")]
        date_flag_cols = [
            col for col in available_cols
            if col.startswith("is_") and col not in exclude and col not in calendar_cols + holiday_cols
        ]

        # Externe Features
        external_cols = [
            col for col in numeric_cols
            if col not in exclude
               and col not in calendar_cols
               and col not in holiday_cols
               and col not in date_flag_cols
        ]

        # Kombiniere alle
        auto_regressors = calendar_cols + holiday_cols + date_flag_cols + external_cols

        # Nur die tatsächlich vorhandenen
        auto_regressors = [col for col in auto_regressors if col in available_cols]

        all_regressors = list(dict.fromkeys(regressors + auto_regressors))

        # Nochmal Ausschlussliste anwenden
        all_regressors = [col for col in all_regressors if col not in exclude]

        return all_regressors

    def create_spec(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Erstellt Prophet-Spezifikation.

        Returns:
            Dict mit:
            - time_col: Original-Zeitstempel-Spalte
            - target_col: Original-Target-Spalte
            - group_cols: ID-Spalten für Gruppen-Iteration
            - regressors: Liste der Regressor-Spalten
            - country_holidays: Ländercode (aus Config)
            - growth: "linear" oder "logistic"
            - seasonality_mode: "additive" oder "multiplicative"
            - n_groups: Anzahl der Gruppen
            - prediction_length: Vorhersagehorizont (aus Config)
        """
        regressors = self.identify_regressors(train_df)

        # Gruppen-Anzahl
        if ID_COLS:
            n_groups = train_df.groupby(ID_COLS).ngroups
        else:
            n_groups = 1

        spec = {
            "time_col": TIME_COL,
            "target_col": TARGET_COL,
            "group_cols": ID_COLS,
            "regressors": regressors,
            "country_holidays": self.prophet_config.get("country_holidays", "DE"),
            "growth": self.prophet_config.get("growth", "linear"),
            "seasonality_mode": self.prophet_config.get("seasonality_mode", "multiplicative"),
            "n_groups": n_groups,
            "prediction_length": self.prophet_config.get("prediction_length", 7),
        }

        return spec

    def validate_spec(self, spec: Dict[str, Any], df: pd.DataFrame) -> None:
        """Validiert die Spezifikation gegen DataFrame."""
        # Prüfe ob alle Regressoren vorhanden sind
        missing_regressors = set(spec["regressors"]) - set(df.columns)
        if missing_regressors:
            raise ValueError(
                f"Regressoren in Spec aber nicht in DataFrame: {missing_regressors}\n"
                f"Verfügbare Spalten: {df.columns.tolist()}"
            )

        # Prüfe time_col und target_col
        if spec["time_col"] not in df.columns:
            raise ValueError(f"time_col '{spec['time_col']}' nicht in DataFrame")

        if spec["target_col"] not in df.columns:
            raise ValueError(f"target_col '{spec['target_col']}' nicht in DataFrame")

        # Prüfe group_cols
        missing_groups = set(spec["group_cols"]) - set(df.columns)
        if missing_groups:
            raise ValueError(f"group_cols nicht in DataFrame: {missing_groups}")

        print("✓ Spec-Validierung erfolgreich")


def main() -> None:
    """Erstellt Prophet-Spezifikation aus Train/Val/Test-Daten."""
    processed_dir = BASE_DIR / "data" / "processed" / _dataset_name

    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"
    test_path = processed_dir / "test.parquet"
    spec_path = processed_dir / "prophet_spec.json"

    # Prüfe ob Dateien existieren
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input-Datei fehlt: {path}\n"
                "Bitte vorher 'model_dataset' ausführen."
            )

    # Lade Daten
    print(f"[dataset_prophet] Lade Daten aus {processed_dir}...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    print(f"[dataset_prophet] Train: {len(train_df):,} Zeilen")
    print(f"[dataset_prophet] Val: {len(val_df):,} Zeilen")
    print(f"[dataset_prophet] Test: {len(test_df):,} Zeilen")

    # Erstelle Spec
    spec_creator = ProphetDatasetSpec(_dataset_config)
    spec = spec_creator.create_spec(train_df, val_df, test_df)

    # Validiere
    spec_creator.validate_spec(spec, train_df)

    # Speichere
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("PROPHET DATASET-SPEZIFIKATION")
    print(f"{'=' * 60}")
    print(f"Dataset: {_dataset_name}")
    print(f"Gruppen: {spec['n_groups']}")
    print(f"Regressoren ({len(spec['regressors'])}):")
    for reg in spec['regressors']:
        print(f"  - {reg}")
    print(f"\nCountry Holidays: {spec['country_holidays']}")
    print(f"Growth: {spec['growth']}")
    print(f"Seasonality Mode: {spec['seasonality_mode']}")
    print(f"Prediction Length: {spec['prediction_length']}")
    print(f"\n✓ Spec gespeichert: {spec_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

# Aufruf einzeln:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.modeling.dataset_prophet
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.modeling.dataset_prophet