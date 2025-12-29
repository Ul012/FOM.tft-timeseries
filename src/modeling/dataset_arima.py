"""
ARIMA Dataset-Spezifikation

Konvertiert vorbereitete Daten zu ARIMA-Format und erstellt eine
Spezifikation für das Training.

ARIMA (AutoRegressive Integrated Moving Average):
- Fokus auf Autokorrelation (AR-Teil)
- Differencing für Stationarität (I-Teil)
- Moving Average für Residuen (MA-Teil)
- Optional: Externe Regressoren (ARIMAX/SARIMAX)

Input: train/val/test.parquet
Output: arima_spec.json

Aufruf:
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
    python -m src.modeling.dataset_arima
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


class ARIMADatasetSpec:
    """Erstellt ARIMA-Datensatz-Spezifikation."""

    def __init__(self, dataset_config: dict):
        self.config = dataset_config
        self.schema = get_schema(dataset_config)
        self.arima_config = dataset_config.get("arima", {})

    def identify_exog_vars(self, df: pd.DataFrame) -> List[str]:
        """
        Identifiziert exogene Variablen (Regressoren) für ARIMAX.
        """
        # Start mit expliziten exog_vars aus Config
        exog_vars = self.arima_config.get("exog_vars", [])

        # Verfügbare Spalten
        available_cols = set(df.columns)
        numeric_cols = set(df.select_dtypes(include=["number"]).columns)

        # Ausschlussliste
        exclude = {
            TIME_COL,
            TARGET_COL,
            "time_idx",
        }
        exclude.update(ID_COLS)

        # Zyklische Encodings ausschließen
        exclude.update([col for col in available_cols if col.startswith("cyc_")])

        # Lag-Features ausschließen
        exclude.update([col for col in available_cols if col.startswith("lag_") or col.endswith("_missing")])

        # Kalender-Features ausschließen (ARIMA nutzt seasonality statt Features)
        calendar_cols = {"year", "month", "day", "dayofweek", "weekofyear", "is_weekend"}
        exclude.update(calendar_cols)

        # Automatische exog-Erkennung
        if not exog_vars:
            # Feiertags-Features
            holiday_cols = [col for col in available_cols if col.startswith("is_holiday")]

            # Custom Flags
            flag_cols = [
                col for col in available_cols
                if col.startswith("is_") and col not in exclude and col not in calendar_cols
            ]

            # Externe Features (z.B. temperature, fuel_price, markdown)
            external_cols = [
                col for col in numeric_cols
                if col not in exclude
                   and col not in calendar_cols
                   and col not in holiday_cols
                   and col not in flag_cols
                   and not col.startswith("cyc_")
                   and not col.startswith("lag_")
            ]

            # Kombiniere (Priorität: Feiertage > Flags > Externe)
            auto_exog = holiday_cols + flag_cols + external_cols

            exog_vars = [col for col in auto_exog if col in available_cols]

        # Nochmal Ausschlussliste anwenden
        exog_vars = [col for col in exog_vars if col not in exclude]

        return exog_vars

    def detect_frequency(self, df: pd.DataFrame) -> str:
        """
        Erkennt die Frequenz der Zeitreihe.
        """
        dates = df[TIME_COL].drop_duplicates().sort_values()
        if len(dates) < 2:
            return "D"  # Default: täglich

        median_diff = dates.diff().dropna().dt.days.median()

        if median_diff <= 1:
            return "D"
        elif 6 <= median_diff <= 8:
            return "W"  # Wöchentlich
        elif 28 <= median_diff <= 31:
            return "M"  # Monatlich
        else:
            return "D"  # Fallback

    def create_spec(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Erstellt ARIMA-Spezifikation.

        Returns:
            Dict mit:
            - time_col: Original-Zeitstempel-Spalte
            - target_col: Original-Target-Spalte
            - group_cols: ID-Spalten für Gruppen-Iteration
            - exog_vars: Liste der exogenen Variablen (Regressoren)
            - frequency: Zeitreihen-Frequenz (D/W/M)
            - seasonal_period: m-Wert für SARIMA (7 für täglich, 52 für wöchentlich, 12 für monatlich)
            - n_groups: Anzahl der Gruppen
            - prediction_length: Vorhersagehorizont
            - auto_arima: Ob auto_arima verwendet werden soll
            - order: ARIMA(p,d,q) falls nicht auto
            - seasonal_order: SARIMA(P,D,Q,m) falls nicht auto
        """
        exog_vars = self.identify_exog_vars(train_df)
        frequency = self.detect_frequency(train_df)

        seasonal_periods = {
            "D": 7,
            "W": 52,
            "M": 12,
        }
        seasonal_period = seasonal_periods.get(frequency, 7)

        # Gruppen-Anzahl
        if ID_COLS:
            n_groups = train_df.groupby(ID_COLS).ngroups
        else:
            n_groups = 1

        spec = {
            "time_col": TIME_COL,
            "target_col": TARGET_COL,
            "group_cols": ID_COLS,
            "exog_vars": exog_vars,
            "frequency": frequency,
            "seasonal_period": seasonal_period,
            "n_groups": n_groups,
            "prediction_length": self.arima_config.get("prediction_length", 7),
            "auto_arima": self.arima_config.get("auto_arima", True),
            "order": self.arima_config.get("order", [1, 1, 1]),  # (p,d,q)
            "seasonal_order": self.arima_config.get("seasonal_order", [1, 1, 1, seasonal_period]),  # (P,D,Q,m)
        }

        return spec

    def validate_spec(self, spec: Dict[str, Any], df: pd.DataFrame) -> None:
        """Validiert die Spezifikation gegen DataFrame."""
        missing_exog = set(spec["exog_vars"]) - set(df.columns)
        if missing_exog:
            raise ValueError(
                f"Exogene Variablen in Spec aber nicht in DataFrame: {missing_exog}\n"
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
    """Erstellt ARIMA-Spezifikation aus Train/Val/Test-Daten."""
    processed_dir = BASE_DIR / "data" / "processed" / _dataset_name

    # Pfade
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"
    test_path = processed_dir / "test.parquet"
    spec_path = processed_dir / "arima_spec.json"

    # Prüfe ob Dateien existieren
    for path in [train_path, val_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Input-Datei fehlt: {path}\n"
                "Bitte vorher 'model_dataset' ausführen."
            )

    # Lade Daten
    print(f"[dataset_arima] Lade Daten aus {processed_dir}...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    print(f"[dataset_arima] Train: {len(train_df):,} Zeilen")
    print(f"[dataset_arima] Val: {len(val_df):,} Zeilen")
    print(f"[dataset_arima] Test: {len(test_df):,} Zeilen")

    # Erstelle Spec
    spec_creator = ARIMADatasetSpec(_dataset_config)
    spec = spec_creator.create_spec(train_df, val_df, test_df)

    # Validiere
    spec_creator.validate_spec(spec, train_df)

    # Speichere
    with open(spec_path, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("ARIMA DATASET-SPEZIFIKATION")
    print(f"{'=' * 60}")
    print(f"Dataset: {_dataset_name}")
    print(f"Frequenz: {spec['frequency']} (seasonal_period={spec['seasonal_period']})")
    print(f"Gruppen: {spec['n_groups']}")
    print(f"Exogene Variablen ({len(spec['exog_vars'])}):")
    for var in spec['exog_vars']:
        print(f"  - {var}")
    print(f"\nAuto-ARIMA: {spec['auto_arima']}")
    if not spec['auto_arima']:
        print(f"Order (p,d,q): {spec['order']}")
        print(f"Seasonal Order (P,D,Q,m): {spec['seasonal_order']}")
    print(f"Prediction Length: {spec['prediction_length']}")
    print(f"\n✓ Spec gespeichert: {spec_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

# Aufruf einzeln:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.modeling.dataset_arima
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.modeling.dataset_arima