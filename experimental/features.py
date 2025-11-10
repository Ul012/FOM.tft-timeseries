# src/data/features.py
# -----------------------------------------------------------------------------
# Aufgabe:
# - Aus Rohdaten (train/test ODER eine dataset.csv) Feature-Tabellen erzeugen.
# - Schritte: FeatureEngineer -> (optional) Lag/Rolling -> (optional) zyklische Kodierung.
# - Ausgaben: features_train.parquet, features_test.parquet
# -----------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import numpy as np

# Projektbausteine
from src.data.feature_engineering import FeatureEngineer
from src.data.cyclical_encoder import CyclicalEncoder, CyclicalEncoderConfig

# Zentrale Konfiguration
try:
    from src.config import (
        RAW, FEATURES,
        FEATURES_FLAGS, CYCLICAL_CONF, LAG_CONF,
        DATETIME_COLUMN, TARGET_COLUMN, GROUP_COLS,
    )
except Exception as e:
    raise ImportError("Konnte Projekt-Konfiguration nicht laden. Prüfe src/config.py und PYTHONPATH.") from e


# -----------------------------------------------------------------------------
# Hilfsfunktionen: I/O
# -----------------------------------------------------------------------------
def _load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Lädt Rohdaten gemäß config.RAW.
    - Bevorzugt getrennte Dateien: RAW['train_csv'], RAW['test_csv']
    - Fallback: RAW['dataset_csv'] -> interner Split (zeitbasiert falls DATETIME_COLUMN vorhanden)
    """
    train_csv = Path(RAW.get("train_csv", ""))
    test_csv = Path(RAW.get("test_csv", ""))
    dataset_csv = Path(RAW.get("dataset_csv", ""))  # optional

    if train_csv.is_file() and test_csv.is_file():
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)
        return df_train, df_test, "two_files"

    if dataset_csv.is_file():
        df_all = pd.read_csv(dataset_csv)
        if DATETIME_COLUMN in df_all.columns:
            df_all = df_all.sort_values(DATETIME_COLUMN)
            split_idx = int(len(df_all) * 0.8)
            df_train, df_test = df_all.iloc[:split_idx], df_all.iloc[split_idx:]
        else:
            from sklearn.model_selection import train_test_split
            df_train, df_test = train_test_split(df_all, test_size=0.2, random_state=42)
        print(f"[features.py] Automatischer Split (dataset.csv) -> Train/Test: {len(df_train)}/{len(df_test)}")
        return df_train, df_test, "single_file"

    raise FileNotFoundError(
        "Keine passenden Eingabedateien gefunden. "
        "Erwarte train/test.csv ODER dataset.csv (siehe src/config.py → RAW)."
    )


def _validate_minimal_schema(df: pd.DataFrame, name: str) -> None:
    if DATETIME_COLUMN not in df.columns:
        raise ValueError(f"[{name}] fehlende Zeitspalte '{DATETIME_COLUMN}'.")
    for g in GROUP_COLS:
        if g not in df.columns:
            raise ValueError(f"[{name}] fehlende Gruppenspalte '{g}'.")


def _save_outputs(df_train_feat: pd.DataFrame, df_test_feat: pd.DataFrame) -> None:
    out_train = Path(FEATURES["train_path"])
    out_test = Path(FEATURES["test_path"])
    out_train.parent.mkdir(parents=True, exist_ok=True)
    df_train_feat.to_parquet(out_train, index=False)
    df_test_feat.to_parquet(out_test, index=False)
    print(f"[features.py] geschrieben: {out_train} (Zeilen: {len(df_train_feat):,})")
    print(f"[features.py] geschrieben: {out_test}  (Zeilen: {len(df_test_feat):,})")


# -----------------------------------------------------------------------------
# Schlanker, interner Lag-/Rolling-Helper (ohne externe Abhängigkeit)
# -----------------------------------------------------------------------------
def _apply_lag_features(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
    group_cols: List[str],
    lags: tuple = (1, 7, 14, 28),
    roll_windows: tuple = (7, 28),
    roll_stats: tuple = ("mean", "std"),
    min_periods: int = 1,
    prefix: str = "lag",
) -> pd.DataFrame:
    """
    Fügt Lags und Rolling-Statistiken für die Zielvariable hinzu.
    - Sortiert vorab nach Gruppen + Zeit.
    - Wenn target_col nicht vorhanden (z. B. Test), wird nichts hinzugefügt.
    """
    if target_col not in df.columns:
        return df

    work = df.copy()

    # Sortierung für stabile Lags
    sort_cols = list(group_cols) + [date_col] if group_cols else [date_col]
    work = work.sort_values(sort_cols).reset_index(drop=True)

    # Lags
    for L in lags:
        col_name = f"{prefix}_{target_col}_{L}"
        if group_cols:
            work[col_name] = work.groupby(group_cols)[target_col].shift(L)
        else:
            work[col_name] = work[target_col].shift(L)

    # Rolling-Stats
    if roll_windows:
        for W in roll_windows:
            if group_cols:
                g = work.groupby(group_cols, group_keys=False)[target_col]
                roll = g.rolling(window=W, min_periods=min_periods)
            else:
                roll = work[target_col].rolling(window=W, min_periods=min_periods)

            for stat in roll_stats:
                if stat == "mean":
                    vals = roll.mean()
                elif stat == "std":
                    vals = roll.std(ddof=0)
                elif stat == "min":
                    vals = roll.min()
                elif stat == "max":
                    vals = roll.max()
                elif stat == "median":
                    vals = roll.median()
                elif stat == "sum":
                    vals = roll.sum()
                else:
                    continue

                col_name = f"{prefix}_{target_col}_roll{W}_{stat}"
                # GroupBy.Rolling gibt SeriesGroupBy zurück → Werte extrahieren
                work[col_name] = vals.values if hasattr(vals, "values") else vals

    return work


# -----------------------------------------------------------------------------
# Feature-Pipeline
# -----------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    use_lags: bool = True,
    use_cyclical: bool = True,
) -> pd.DataFrame:
    """
    Führt Feature Engineering und optionale Schritte in sinnvoller Reihenfolge aus.
    Erwartet mindestens die Zeitspalte (DATETIME_COLUMN) und – falls definiert – Gruppenspalten (GROUP_COLS).
    """
    # Rohdaten-Kopie + **einheitlicher Zeit-Datentyp** (wichtig fürs spätere Merge)
    base = df.copy()
    base[DATETIME_COLUMN] = pd.to_datetime(base[DATETIME_COLUMN], utc=False, errors="coerce")

    # 1) Basis-Features (Kalender, Zeitindizes, Feiertage etc.)
    fe = FeatureEngineer(date_col=DATETIME_COLUMN, include_holiday_name=False)
    data = fe.transform(base)

    # 2) (Optional) Lag-/Rolling-Features (intern)
    if use_lags and LAG_CONF:
        data = _apply_lag_features(
            df=data,
            target_col=LAG_CONF.get("target_col", TARGET_COLUMN),
            date_col=LAG_CONF.get("date_col", DATETIME_COLUMN),
            group_cols=LAG_CONF.get("group_cols", GROUP_COLS),
            lags=tuple(LAG_CONF.get("lags", (1, 7, 14))),
            roll_windows=tuple(LAG_CONF.get("roll_windows", (7, 28))),
            roll_stats=tuple(LAG_CONF.get("roll_stats", ("mean", "std"))),
            min_periods=int(LAG_CONF.get("min_periods", 1)),
            prefix=str(LAG_CONF.get("prefix", "lag")),
        )

    # 3) (Optional) zyklische Kodierung
    if use_cyclical:
        enc = CyclicalEncoder(CyclicalEncoderConfig(**CYCLICAL_CONF)) if CYCLICAL_CONF else CyclicalEncoder()
        data = enc.fit_transform(data)

    # 4) Schlüssel + Target sicherstellen (robuster Merge – **dtype-harmonisiert**)
    keys = [DATETIME_COLUMN] + list(GROUP_COLS)
    keep = [c for c in keys + [TARGET_COLUMN] if c in base.columns]

    if keep:
        # Sicherstellen, dass auch in data die Zeitspalte im gleichen dtype ist
        data[DATETIME_COLUMN] = pd.to_datetime(data[DATETIME_COLUMN], utc=False, errors="coerce")
        data = data.merge(
            base[keep].drop_duplicates(),
            on=[c for c in keys if c in base.columns],
            how="left",
        )

    return data


# -----------------------------------------------------------------------------
# Main-Orchestrierung
# -----------------------------------------------------------------------------
def main() -> None:
    # 1) Rohdaten laden
    df_train_raw, df_test_raw, mode = _load_inputs()

    # 2) Feature-Flags aus config
    use_lags = bool(FEATURES_FLAGS.get("use_lags", True))
    use_cyc = bool(FEATURES_FLAGS.get("use_cyclical", True))

    # 3) Features erzeugen
    df_train_feat = build_features(df_train_raw, use_lags=use_lags, use_cyclical=use_cyc)
    df_test_feat = build_features(df_test_raw, use_lags=use_lags, use_cyclical=use_cyc)

    # 4) Minimal-Validierung
    _validate_minimal_schema(df_train_feat, "train_features")
    _validate_minimal_schema(df_test_feat, "test_features")

    # 5) Speichern
    _save_outputs(df_train_feat, df_test_feat)

    print(f"[features.py] Features erfolgreich erstellt. Modus: {mode}")


if __name__ == "__main__":
    main()
