# src/modeling/dataset_tft.py
"""
Erzeugt eine Datensatz-Spezifikation für Temporal Fusion Transformer (TFT):
- Liest train/val/test aus DATASETS_DIR (config.py)
- Leitet Feature-Listen heuristisch ab (static_categoricals, known/unknown reals)
- Schreibt dataset_spec.json für nachgelagerte Trainer

Keine Abhängigkeiten zu PyTorch/FastAI/pytorch-forecasting – reine Datenspezifikation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import json
import pandas as pd


# ------------------------- Konfiguration laden -------------------------

def _load_config_safe() -> Dict[str, Any]:
    """
    Erwartete Keys (aus src/config.py):
      - DATASETS_DIR: str|Path
      - TIME_COL: str
      - ID_COLS: list[str]
      - TARGET_COL: str
      - TFT_DATASET: dict (optional) mit u.a.:
          * max_encoder_length: int
          * max_prediction_length: int
          * known_real_prefixes: list[str] (z. B. ["cyc_"])
          * lag_prefixes: list[str] (z. B. ["lag_"])
          * treat_calendar_as_known: bool
          * flag_cols: list[str] (z. B. ["is_lockdown_period"])
    """
    cfg: Dict[str, Any] = {}
    try:
        from src.config import (  # type: ignore
            DATASETS_DIR,
            TIME_COL,
            ID_COLS,
            TARGET_COL,
            TFT_DATASET,  # optional
        )
        cfg["DATASETS_DIR"] = Path(DATASETS_DIR)
        cfg["TIME_COL"] = TIME_COL
        cfg["ID_COLS"] = list(ID_COLS)
        cfg["TARGET_COL"] = TARGET_COL
        cfg["TFT_DATASET"] = dict(TFT_DATASET) if "TFT_DATASET" in locals() else {}
        return cfg
    except Exception:
        # Fallbacks (typisch)
        cfg["DATASETS_DIR"] = Path("data/processed/model_dataset")
        cfg["TIME_COL"] = "date"
        cfg["ID_COLS"] = ["country", "store", "product"]
        cfg["TARGET_COL"] = "num_sold"
        cfg["TFT_DATASET"] = {}
        return cfg


# ------------------------- Heuristiken/Defaults (nur module-scope) -------------------------

CALENDAR_COLS = {"year", "month", "day", "dayofweek", "weekofyear", "is_weekend"}
HOLIDAY_PREFIXES = ("is_holiday",)  # z. B. is_holiday_de, is_holiday_*
CYCLICAL_DEFAULT_PREFIXES = ["cyc_"]
LAG_DEFAULT_PREFIXES = ["lag_"]
FLAG_COLS_DEFAULT = {"is_lockdown_period"}  # Default-Flags; über config erweiterbar

DEFAULT_TFT_CFG = {
    "max_encoder_length": 28,
    "max_prediction_length": 7,
    "known_real_prefixes": CYCLICAL_DEFAULT_PREFIXES,
    "lag_prefixes": LAG_DEFAULT_PREFIXES,
    "treat_calendar_as_known": True,
}


# ------------------------- Builder -------------------------

@dataclass
class TFTDatasetSpecBuilder:
    datasets_dir: Path
    time_col: str
    id_cols: List[str]
    target_col: str
    tft_cfg: Dict[str, Any]

    def run(self) -> Dict[str, Any]:
        paths = {
            "train": self.datasets_dir / "train.parquet",
            "val": self.datasets_dir / "val.parquet",
            "test": self.datasets_dir / "test.parquet",
        }
        for name, p in paths.items():
            if not p.exists():
                raise FileNotFoundError(f"{name}.parquet nicht gefunden: {p}")

        # Trainingssatz einlesen
        train = pd.read_parquet(paths["train"])
        self._basic_checks(train)

        all_cols = list(train.columns)
        # numerisch + bool zulassen (0/1-Flags können als bool gespeichert sein)
        numeric_cols = train.select_dtypes(include=["number", "bool"]).columns.tolist()

        # 1) Static categoricals: ID-Spalten
        static_categoricals = [c for c in self.id_cols if c in all_cols]

        # 2) Known reals: zyklisch / Kalender / Feiertage / Flags / time_idx
        known_real_prefixes = self.tft_cfg.get(
            "known_real_prefixes", DEFAULT_TFT_CFG["known_real_prefixes"]
        )
        treat_calendar = self.tft_cfg.get(
            "treat_calendar_as_known", DEFAULT_TFT_CFG["treat_calendar_as_known"]
        )
        # Flag-Spalten: aus config oder Default
        flag_cols = set(self.tft_cfg.get("flag_cols", list(FLAG_COLS_DEFAULT)))

        known_reals: List[str] = []

        # zyklische Präfixe (z. B. cyc_dow_sin/cos)
        for c in all_cols:
            if any(c.startswith(pref) for pref in known_real_prefixes) and c in numeric_cols:
                known_reals.append(c)

        # Kalender und Feiertage
        if treat_calendar:
            for c in CALENDAR_COLS:
                if c in numeric_cols:
                    known_reals.append(c)
            # Feiertage per Präfix
            for c in all_cols:
                if c.startswith(HOLIDAY_PREFIXES) and c in numeric_cols:
                    known_reals.append(c)
            # explizite Flags (nicht an numeric_cols binden -> auch bool erlauben)
            for c in flag_cols:
                if c in all_cols:
                    known_reals.append(c)

        # time_idx (falls vorhanden)
        if "time_idx" in numeric_cols:
            known_reals.append("time_idx")

        # Reihenfolge stabilisieren, Duplikate entfernen (order-preserving)
        seen = set()
        known_reals = [x for x in known_reals if not (x in seen or seen.add(x))]

        # 3) Unknown reals: target + lags + sonstige numerische, die nicht known/IDs sind
        lag_prefixes = self.tft_cfg.get("lag_prefixes", DEFAULT_TFT_CFG["lag_prefixes"])
        lag_cols = [c for c in all_cols if any(c.startswith(pref) for pref in lag_prefixes) and c in numeric_cols]

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

        # 4) Sequenzlängen
        max_encoder_length = int(self.tft_cfg.get("max_encoder_length", DEFAULT_TFT_CFG["max_encoder_length"]))
        max_prediction_length = int(self.tft_cfg.get("max_prediction_length", DEFAULT_TFT_CFG["max_prediction_length"]))

        # 5) Speichern
        out_dir = self.datasets_dir / "tft"
        out_dir.mkdir(parents=True, exist_ok=True)
        spec_path = out_dir / "dataset_spec.json"

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
                    "flag_cols": list(flag_cols),
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
    cfg = _load_config_safe()
    builder = TFTDatasetSpecBuilder(
        datasets_dir=Path(cfg["DATASETS_DIR"]),
        time_col=cfg["TIME_COL"],
        id_cols=cfg["ID_COLS"],
        target_col=cfg["TARGET_COL"],
        tft_cfg=cfg.get("TFT_DATASET", {}),
    )
    builder.run()


if __name__ == "__main__":
    main()

# python -m src.modeling.dataset_tft