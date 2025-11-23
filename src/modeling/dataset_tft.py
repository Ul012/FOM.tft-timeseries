# src/modeling/dataset_tft.py
"""
Erzeugt eine Datensatz-Spezifikation für den Temporal Fusion Transformer (TFT).

- Liest train/val/test aus PROCESSED_DIR (config.py)
- Leitet Feature-Listen heuristisch ab (static_categoricals, known/unknown reals)
- Schreibt dataset_spec.json für den nachgelagerten Trainer

Kein Training, keine PyTorch-Abhängigkeit – reine Datenspezifikation.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Set
import json
import pandas as pd

from src.config import PROCESSED_DIR, BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema, get_tft_config

# Lade Config einmalig
_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]
_schema = get_schema(_dataset_config)

# Extrahiere Werte
TIME_COL = _schema["time_col"]
ID_COLS = _schema["id_cols"]
TARGET_COL = _schema["target_col"]
TFT_DATASET = get_tft_config(_dataset_config)

# ------------------------- Heuristiken -------------------------

CALENDAR_COLS = {"year", "month", "day", "dayofweek", "weekofyear", "is_weekend"}
HOLIDAY_PREFIXES = ("is_holiday",)  # z. B. is_holiday_de, is_holiday_*
# Hinweis: weitere Präfixe/Flags werden ausschließlich über TFT_DATASET gesteuert.


# ------------------------- Builder -------------------------

@dataclass
class TFTDatasetSpecBuilder:
    datasets_dir: Path
    time_col: str
    id_cols: List[str]
    target_col: str
    tft_cfg: Dict[str, Any]

    def run(self) -> Dict[str, Any]:
        # 1) Pfade für train/val/test
        paths = {
            "train": self.datasets_dir / "train.parquet",
            "val": self.datasets_dir / "val.parquet",
            "test": self.datasets_dir / "test.parquet",
        }
        for name, p in paths.items():
            if not p.exists():
                raise FileNotFoundError(f"{name}.parquet nicht gefunden: {p}")

        # 2) Trainingssatz einlesen und prüfen
        train = pd.read_parquet(paths["train"])

        # Imputing aus Config
        impute_cfg = self.tft_cfg.get("impute_cols", {})
        if impute_cfg:
            print(f"[dataset_tft] Impute {len(impute_cfg)} Spalten aus Config")
            for col, fill_val in impute_cfg.items():
                if col in train.columns:
                    train[col] = train[col].fillna(fill_val)
                    # Auch val/test imputen
                    for split in ["val", "test"]:
                        df_split = pd.read_parquet(paths[split])
                        if col in df_split.columns:
                            df_split[col] = df_split[col].fillna(fill_val)
                            df_split.to_parquet(paths[split], index=False)
            train.to_parquet(paths["train"], index=False)

        # Exclude-Spalten aus Config
        exclude_cols = self.tft_cfg.get("exclude_cols", [])
        if exclude_cols:
            print(f"[dataset_tft] Entferne {len(exclude_cols)} Spalten aus Config: {exclude_cols}")
            train = train.drop(columns=[c for c in exclude_cols if c in train.columns], errors='ignore')
            for split in ["val", "test"]:
                df_split = pd.read_parquet(paths[split])
                df_split = df_split.drop(columns=[c for c in exclude_cols if c in df_split.columns],
                                             errors='ignore')
                df_split.to_parquet(paths[split], index=False)
            train.to_parquet(paths["train"], index=False)

        self._basic_checks(train)

        all_cols = list(train.columns)
        # numerisch + bool zulassen (0/1-Flags können als bool gespeichert sein)
        numeric_cols = train.select_dtypes(include=["number", "bool"]).columns.tolist()

        # 3) Static categoricals: ID-Spalten
        static_categoricals = [c for c in self.id_cols if c in all_cols]

        # 4) Known reals: zyklische Features / Kalender / Feiertage / Flags / time_idx
        known_real_prefixes: List[str] = list(self.tft_cfg["known_real_prefixes"])
        lag_prefixes: List[str] = list(self.tft_cfg["lag_prefixes"])
        treat_calendar: bool = bool(self.tft_cfg["treat_calendar_as_known"])
        flag_cols_cfg: List[str] = list(self.tft_cfg["flag_cols"])

        known_reals: List[str] = []

        # zyklische Präfixe (z. B. cyc_dow_sin/cos)
        for c in all_cols:
            if any(c.startswith(pref) for pref in known_real_prefixes) and c in numeric_cols:
                known_reals.append(c)

        # Kalender und Feiertage
        if treat_calendar:
            # einfache Kalenderfeatures
            for c in CALENDAR_COLS:
                if c in numeric_cols:
                    known_reals.append(c)
            # Feiertage per Präfix
            for c in all_cols:
                if c.startswith(HOLIDAY_PREFIXES) and c in numeric_cols:
                    known_reals.append(c)
            # explizite Flags (nicht an numeric_cols binden -> auch bool erlauben)
            for c in flag_cols_cfg:
                if c in all_cols:
                    known_reals.append(c)

        # time_idx (falls vorhanden)
        if "time_idx" in numeric_cols:
            known_reals.append("time_idx")

        # Duplikate entfernen, Reihenfolge beibehalten
        seen = set()
        known_reals = [x for x in known_reals if not (x in seen or seen.add(x))]

        # 5) Unknown reals: target + Lags + sonstige numerische, die nicht known/IDs sind
        lag_cols = [
            c for c in all_cols
            if any(c.startswith(pref) for pref in lag_prefixes) and c in numeric_cols
        ]

        unknown_reals: List[str] = []
        if self.target_col in numeric_cols:
            unknown_reals.append(self.target_col)

        for c in numeric_cols:
            if c == self.target_col:
                continue
            if c in known_reals:
                continue
            if c in self.id_cols:
                continue
            if c in lag_cols:
                continue
            unknown_reals.append(c)

        # Lags ans Ende (nur für Übersicht)
        unknown_reals.extend(lag_cols)

        # 6) Sequenzlängen (müssen in TFT_DATASET konfiguriert sein)
        max_encoder_length = int(self.tft_cfg["max_encoder_length"])
        max_prediction_length = int(self.tft_cfg["max_prediction_length"])

        # 7) Spezifikation schreiben
        spec_path = self.datasets_dir / "dataset_spec.json"

        spec: Dict[str, Any] = {
            "time_col": self.time_col,
            "id_cols": self.id_cols,
            "target_col": self.target_col,
            "paths": {k: str(v) for k, v in paths.items()},
            "feature_lists": {
                "static_categoricals": static_categoricals,
                "time_varying_known_reals": known_reals,
                "time_varying_unknown_reals": unknown_reals,
                "time_varying_known_categoricals": [],  # bewusst leer gehalten
            },
            "lengths": {
                "max_encoder_length": max_encoder_length,
                "max_prediction_length": max_prediction_length,
            },
            "notes": {
                "calendar_as_known": treat_calendar,
                "heuristic_prefixes": {
                    "known_real_prefixes": known_real_prefixes,
                    "lag_prefixes": lag_prefixes,
                    "holiday_prefixes": list(HOLIDAY_PREFIXES),
                    "flag_cols": flag_cols_cfg,
                },
            },
        }

        with spec_path.open("w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)

        print("[dataset_tft] Spezifikation erstellt.")
        print(f"- static_categoricals: {static_categoricals}")
        print(f"- known_reals       : {len(known_reals)} Spalten")
        print(f"- unknown_reals     : {len(unknown_reals)} Spalten (inkl. Lags)")
        print(f"- Längen enc/pred   : {max_encoder_length}/{max_prediction_length}")
        print(f"- Ausgabe           : {spec_path}")

        return spec

    # ------------------------- intern -------------------------

    def _basic_checks(self, df: pd.DataFrame) -> None:
        for c in self.id_cols + [self.time_col, self.target_col]:
            if c not in df.columns:
                raise KeyError(f"Erwartete Spalte fehlt: {c}")


# ------------------------- CLI -------------------------

def main() -> None:
    builder = TFTDatasetSpecBuilder(
        datasets_dir=BASE_DIR / "data" / "processed" / _dataset_name,
        time_col=TIME_COL,
        id_cols=list(ID_COLS),
        target_col=TARGET_COL,
        tft_cfg=TFT_DATASET,
    )
    builder.run()


if __name__ == "__main__":
    main()

# Aufruf einzeln:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.modeling.dataset_tft
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.modeling.dataset_tft
#
# Via Pipeline:
#   python -m src.pipeline --dataset configs/datasets/walmart.yaml --steps model_dataset
#   python -m src.pipeline --dataset configs/datasets/booksales.yaml --steps model_dataset

