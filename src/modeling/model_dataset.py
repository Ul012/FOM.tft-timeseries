# src/modeling/model_dataset.py
"""
Erstellt train/val/test-Datensätze für das TFT-Training aus einem bereits
vorverarbeiteten DataFrame (z. B. Ergebnis aus preprocess.py + features.py).

Philosophie:
- Einfacher, deterministischer Zeit-Split.
- Sanity-Checks gegen Leckage.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any

import json
import pandas as pd


# ------------------------- Konfiguration laden (leichtgewichtig) -------------------------

def _load_config_safe() -> Dict[str, Any]:
    """
    Versucht, src/config.py zu importieren und relevante Felder zu lesen.
    Fällt auf sinnvolle Defaults zurück, falls nicht vorhanden.
    Erwartete Keys (falls vorhanden):
      - DATA_PROCESSED_PATH: str | Path  -> Pfad zur verarbeiteten Datei (csv/parquet)
      - DATASETS_DIR: str | Path         -> Ausgabeordner für train/val/test
      - TIME_COL: str                    -> Zeitspaltenname (z. B. 'date')
      - ID_COLS: list[str]               -> Gruppenidentifikatoren (z. B. ['store_id','item_id'])
      - TARGET_COL: str                  -> Zielvariable (z. B. 'sales')
      - VAL_START: str (YYYY-MM-DD)      -> Optional, explizite Startdate der Validation
      - TEST_START: str (YYYY-MM-DD)     -> Optional, explizite Startdate des Tests
      - SPLIT_RATIOS: tuple[float,float,float] -> Optional, z. B. (0.7,0.15,0.15), wenn keine festen Daten
      - SCALE_COLS: list[str]            -> Optional, Spalten für einfache Z-Standardisierung (group-wise)
    """
    cfg: Dict[str, Any] = {}
    try:
        # kein harter Import oben – schlank halten
        from src.config import (  # type: ignore
            DATA_PROCESSED_PATH,
            DATASETS_DIR,
            TIME_COL,
            ID_COLS,
            TARGET_COL,
            VAL_START,
            TEST_START,
            SPLIT_RATIOS,
            SCALE_COLS,
        )
        cfg["DATA_PROCESSED_PATH"] = Path(DATA_PROCESSED_PATH)
        cfg["DATASETS_DIR"] = Path(DATASETS_DIR)
        cfg["TIME_COL"] = TIME_COL
        cfg["ID_COLS"] = list(ID_COLS)
        cfg["TARGET_COL"] = TARGET_COL
        cfg["VAL_START"] = VAL_START
        cfg["TEST_START"] = TEST_START
        cfg["SPLIT_RATIOS"] = tuple(SPLIT_RATIOS) if SPLIT_RATIOS else None
        cfg["SCALE_COLS"] = list(SCALE_COLS) if SCALE_COLS else []
        return cfg
    except Exception:
        # Schlanke Defaults, falls keine config.py vorliegt
        cfg["DATA_PROCESSED_PATH"] = Path("data/processed/booksales.parquet")
        cfg["DATASETS_DIR"] = Path("data/sets")
        cfg["TIME_COL"] = "date"
        cfg["ID_COLS"] = ["group_id"]       # bei Bedarf in config.py überschreiben
        cfg["TARGET_COL"] = "sales"
        cfg["VAL_START"] = None
        cfg["TEST_START"] = None
        cfg["SPLIT_RATIOS"] = (0.7, 0.15, 0.15)  # nur, wenn keine festen Datumsgrenzen
        cfg["SCALE_COLS"] = []  # z. B. ["sales"] oder Feature-Spalten
        return cfg


# ------------------------- I/O-Helfer -------------------------

def _read_any_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    raise ValueError(f"Nicht unterstütztes Format: {path.suffix}")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ------------------------- Zeit-Split-Logik -------------------------

@dataclass(frozen=True)
class TimeSplitPlan:
    """Einfacher Plan: train < val < test (nach Zeit)."""
    val_start: Optional[pd.Timestamp] = None
    test_start: Optional[pd.Timestamp] = None
    ratios: Optional[Tuple[float, float, float]] = None  # Fallback, wenn keine Startdaten

    @classmethod
    def from_config(cls, val_start: Optional[str], test_start: Optional[str],
                    ratios: Optional[Tuple[float, float, float]]) -> "TimeSplitPlan":
        vs = pd.to_datetime(val_start) if val_start else None
        ts = pd.to_datetime(test_start) if test_start else None
        return cls(val_start=vs, test_start=ts, ratios=ratios)

    def compute_boundaries(self, df: pd.DataFrame, time_col: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Liefert (val_start, test_start). Wenn in config Datumswerte gegeben sind, nutzt diese.
        Ansonsten bestimmt es Grenzen über Ratios (global, nicht pro Gruppe).
        """
        if self.val_start is not None and self.test_start is not None:
            if not (self.val_start < self.test_start):
                raise ValueError("Erwartet: VAL_START < TEST_START.")
            return self.val_start, self.test_start

        if not self.ratios:
            raise ValueError("Weder feste Startdaten noch SPLIT_RATIOS vorhanden.")

        r_train, r_val, r_test = self.ratios
        if abs((r_train + r_val + r_test) - 1.0) > 1e-6:
            raise ValueError("SPLIT_RATIOS müssen zu 1.0 summieren, z. B. (0.7,0.15,0.15).")

        ts_sorted = df[time_col].sort_values().reset_index(drop=True)
        n = len(ts_sorted)
        if n < 10:
            raise ValueError("Zu wenige Zeilen für einen sinnvollen Split.")

        idx_val = max(1, int(n * r_train))
        idx_test = max(idx_val + 1, int(n * (r_train + r_val)))

        # Grenzwerte auf echte Zeitstempel mappen (Anfang der jeweiligen Segmente)
        val_start = pd.to_datetime(ts_sorted.iloc[idx_val])
        test_start = pd.to_datetime(ts_sorted.iloc[idx_test])
        if not (val_start < test_start):
            raise ValueError("Berechnete Grenzen verletzen val_start < test_start.")
        return val_start, test_start


def time_split(df: pd.DataFrame, time_col: str, val_start: pd.Timestamp, test_start: pd.Timestamp
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    train = df[df[time_col] < val_start]
    val = df[(df[time_col] >= val_start) & (df[time_col] < test_start)]
    test = df[df[time_col] >= test_start]
    return train, val, test


# ------------------------- Optionale Skalierung (einfach, gruppenweise) -------------------------

def zscore_groupwise(df: pd.DataFrame, group_cols: List[str], cols: Iterable[str]) -> pd.DataFrame:
    """
    Einfache Z-Standardisierung je Gruppe (Mittel=0, Std=1). Keine SciKit-Abhängigkeit.
    Achtung: Nur für Trainingsdaten fitten und auf val/test anwenden → hier:
    - Wir berechnen pro Gruppe mean/std aus TRAIN
    - Wenden dieselben Parameter auf val/test an
    """
    return df  # Platzhalter – echte Anwendung erfolgt in Builder (siehe unten)


# ------------------------- Builder -------------------------

@dataclass
class ModelDatasetBuilder:
    data_path: Path
    output_dir: Path
    time_col: str
    id_cols: List[str]
    target_col: str
    val_start: Optional[str] = None
    test_start: Optional[str] = None
    split_ratios: Optional[Tuple[float, float, float]] = None
    scale_cols: Optional[List[str]] = None  # leere Liste => keine Skalierung

    def run(self) -> Dict[str, Any]:
        # 1) Laden
        df = _read_any_table(self.data_path)
        if self.time_col not in df.columns:
            raise KeyError(f"TIME_COL '{self.time_col}' nicht in DataFrame.")
        if self.target_col not in df.columns:
            raise KeyError(f"TARGET_COL '{self.target_col}' nicht in DataFrame.")
        for c in self.id_cols:
            if c not in df.columns:
                raise KeyError(f"ID_COL '{c}' nicht in DataFrame.")

        # 2) Sortierung und Typen
        df = df.copy()
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df.sort_values([self.time_col] + self.id_cols, inplace=True)

        # 3) Split-Plan bestimmen
        plan = TimeSplitPlan.from_config(self.val_start, self.test_start, self.split_ratios)
        val_start_ts, test_start_ts = plan.compute_boundaries(df, self.time_col)

        # 4) Splitten
        train, val, test = time_split(df, self.time_col, val_start_ts, test_start_ts)

        # 5) Sanity-Checks
        self._sanity_checks(train, val, test)

        # 6) Optionale gruppenweise Z-Standardisierung auf ausgewählte Spalten
        if self.scale_cols:
            # Fit nur auf TRAIN (group-wise)
            stats = (
                train.groupby(self.id_cols)[self.scale_cols]
                .agg(["mean", "std"])
                .reset_index()
            )
            # MultiIndex-Spalten → flachziehen
            stats.columns = ["__".join([c for c in col if c]) for col in stats.columns.values]

            def _apply_scale(part: pd.DataFrame) -> pd.DataFrame:
                part = part.copy()
                merged = part.merge(
                    stats,
                    on=self.id_cols,
                    how="left",
                    suffixes=("", ""),
                )
                for col in self.scale_cols:  # type: ignore
                    mcol = f"{col}__mean"
                    scol = f"{col}__std"
                    mean = merged[mcol]
                    std = merged[scol].replace(0, pd.NA)
                    part[col] = (part[col] - mean).divide(std)
                return part

            train = _apply_scale(train)
            val = _apply_scale(val)
            test = _apply_scale(test)

        # 7) Speichern
        _ensure_dir(self.output_dir)
        paths = {
            "train": self.output_dir / "train.parquet",
            "val": self.output_dir / "val.parquet",
            "test": self.output_dir / "test.parquet",
            "manifest": self.output_dir / "manifest.json",
        }
        train.to_parquet(paths["train"], index=False)
        val.to_parquet(paths["val"], index=False)
        test.to_parquet(paths["test"], index=False)

        manifest = {
            "time_col": self.time_col,
            "id_cols": self.id_cols,
            "target_col": self.target_col,
            "val_start": str(val_start_ts.date()),
            "test_start": str(test_start_ts.date()),
            "rows": {
                "train": len(train),
                "val": len(val),
                "test": len(test),
            },
            "output_dir": str(self.output_dir),
            "source": str(self.data_path),
            "scaled_cols": self.scale_cols or [],
        }
        with paths["manifest"].open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # 8) Kurze Ausgabe
        print("[model_dataset] Fertig.")
        print(f"- Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
        print(f"- Grenzen: VAL_START={val_start_ts.date()}  TEST_START={test_start_ts.date()}")
        print(f"- Ausgabepfad: {self.output_dir}")

        return manifest

    # ------------------------- intern -------------------------

    def _sanity_checks(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
        """Leckage-Prüfungen und Basiskontrollen."""
        if train.empty or val.empty or test.empty:
            raise ValueError("Mindestens eine Split-Teilmenge ist leer – prüfe Grenzen/Datenbasis.")

        # Zeitliche Ordnung
        t_min, t_max = train[self.time_col].min(), train[self.time_col].max()
        v_min, v_max = val[self.time_col].min(), val[self.time_col].max()
        s_min, s_max = test[self.time_col].min(), test[self.time_col].max()

        if not (t_max < v_min <= v_max < s_min or t_max < s_min):
            # Weniger streng: Hauptsache train endet vor val/test beginnt.
            if not (t_max < v_min and t_max < s_min):
                raise ValueError("Zeitliche Trennung verletzt (Train überlappt).")

        # ID-Konsistenz (optional, pragmatisch)
        # Warnen, wenn IDs in val/test vorkommen, die nie in train waren – kann gewollt sein,
        # ist aber für TFT oft unerwünscht.
        train_ids = set(map(tuple, train[self.id_cols].drop_duplicates().values))
        val_ids = set(map(tuple, val[self.id_cols].drop_duplicates().values))
        test_ids = set(map(tuple, test[self.id_cols].drop_duplicates().values))
        unseen_val = val_ids - train_ids
        unseen_test = test_ids - train_ids
        if unseen_val:
            print(f"[Warnung] {len(unseen_val)} Gruppen nur in VAL (nicht in TRAIN).")
        if unseen_test:
            print(f"[Warnung] {len(unseen_test)} Gruppen nur in TEST (nicht in TRAIN).")


# ------------------------- CLI -------------------------

def main() -> None:
    cfg = _load_config_safe()
    builder = ModelDatasetBuilder(
        data_path=Path(cfg["DATA_PROCESSED_PATH"]),
        output_dir=Path(cfg["DATASETS_DIR"]),
        time_col=cfg["TIME_COL"],
        id_cols=cfg["ID_COLS"],
        target_col=cfg["TARGET_COL"],
        val_start=cfg.get("VAL_START"),
        test_start=cfg.get("TEST_START"),
        split_ratios=cfg.get("SPLIT_RATIOS"),
        scale_cols=cfg.get("SCALE_COLS", []),
    )
    builder.run()


if __name__ == "__main__":
    main()

# python -m src.modeling.model_dataset
