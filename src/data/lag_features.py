# src/data/lag_features.py
"""
Erzeugt Lag- und Rolling-Features für Zeitreihen.

Features:
- Lag-Features (z.B. lag_1, lag_7, lag_365)
- Rolling-Features (z.B. lag_7_mean)
- NaN-Handling mit Median-Imputation + Missing-Indicator
- min_group_length Filtering (entfernt zu kurze Gruppen)

Aufruf:
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.lag_features
"""

import pandas as pd
import numpy as np

from src.config import PROCESSED_DIR, BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema, get_preprocessing_params

# Lade Config einmalig
_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]
_schema = get_schema(_dataset_config)
_lag_params = get_preprocessing_params(_dataset_config, "lag_features")

# Extrahiere Werte
GROUP_COLS = _schema["id_cols"]
TIME_COL = _schema["time_col"]
TARGET_COL = _schema["target_col"]

# Lag-Config
LAG_CONF = {
    "target_col": TARGET_COL,
    "lags": _lag_params.get("lags", [1, 7, 14]),
    "roll_windows": _lag_params.get("roll_windows", []),
    "roll_stats": _lag_params.get("roll_stats", []),
    "prefix": _lag_params.get("prefix", "lag_"),
    "min_group_length": _lag_params.get("min_group_length", None),
}


def filter_short_groups(df: pd.DataFrame, min_length: int) -> pd.DataFrame:
    """
    Entfernt Gruppen die kürzer als min_length sind.

    Args:
        df: DataFrame mit GROUP_COLS
        min_length: Minimale Anzahl Zeitschritte pro Gruppe

    Returns:
        DataFrame ohne zu kurze Gruppen
    """
    if not GROUP_COLS or min_length is None:
        return df

    # Berechne Gruppenlängen
    group_lengths = df.groupby(GROUP_COLS).size()

    # Finde Gruppen die lang genug sind
    valid_groups = group_lengths[group_lengths >= min_length].index
    n_removed = len(group_lengths) - len(valid_groups)

    if n_removed > 0:
        print(f"[lag_features] Entferne {n_removed} Gruppen mit < {min_length} Zeitschritten")
        print(f"[lag_features] Verbleibende Gruppen: {len(valid_groups)}")

        # Filtere DataFrame
        if len(GROUP_COLS) == 1:
            df = df[df[GROUP_COLS[0]].isin(valid_groups)]
        else:
            # Multi-Index für mehrere GROUP_COLS
            df = df.set_index(GROUP_COLS)
            df = df[df.index.isin(valid_groups)]
            df = df.reset_index()

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Erzeugt Lag- und Rolling-Features basierend auf LAG_CONF.

    Includes:
    - Lag-Features für alle konfigurierten Lags
    - Rolling-Features (optional)
    - NaN-Handling: Median-Imputation + Missing-Indicator
    """
    target = LAG_CONF["target_col"]
    lags = LAG_CONF["lags"]
    roll_windows = LAG_CONF.get("roll_windows", [])
    roll_stats = LAG_CONF.get("roll_stats", [])
    prefix = LAG_CONF.get("prefix", "lag_")

    # Nach Gruppe und Zeit sortieren
    df = df.sort_values(GROUP_COLS + [TIME_COL]).copy()

    # === Lag-Features ===
    for lag in lags:
        col_name = f"{prefix}{lag}"
        df[col_name] = df.groupby(GROUP_COLS)[target].shift(lag)
        print(f"[lag_features] Erstellt: {col_name}")

    # === Rolling-Features ===
    for window in roll_windows:
        for stat in roll_stats:
            col_name = f"{prefix}{window}_{stat}"
            rolled = df.groupby(GROUP_COLS)[target].transform(
                lambda x: getattr(x.shift(1).rolling(window=window, min_periods=1), stat)()
            )
            df[col_name] = rolled
            print(f"[lag_features] Erstellt: {col_name}")

    # === NaN-Handling mit Indikator-Features ===
    # Best Practice laut PyTorch Forecasting FAQ:
    # "Fill with median value and add missing indicator categorical variable"
    lag_cols = [col for col in df.columns if col.startswith(prefix)]

    for col in lag_cols:
        n_nans = df[col].isna().sum()
        if n_nans > 0:
            pct = (n_nans / len(df)) * 100
            print(f"[lag_features] NaN-Handling: {col} hat {n_nans} ({pct:.1f}%) NaN-Werte")

            # 1. Indikator-Feature: 1 = war NaN, 0 = hatte Wert
            df[f"{col}_missing"] = df[col].isna().astype(int)

            # 2. Fülle mit Gruppen-Median (Fallback: globaler Median)
            group_medians = df.groupby(GROUP_COLS)[col].transform('median')
            global_median = df[col].median()

            # Erst Gruppen-Median, dann globaler Median als Fallback
            df[col] = df[col].fillna(group_medians)
            df[col] = df[col].fillna(global_median)

            # Prüfe ob noch NaN übrig
            remaining_nans = df[col].isna().sum()
            if remaining_nans > 0:
                print(f"[WARNUNG] {col} hat noch {remaining_nans} NaN nach Imputation - fülle mit 0")
                df[col] = df[col].fillna(0)

    return df


def main() -> None:
    """Liest train_features_cyc.parquet, erzeugt Lag/Rolling-Features und speichert."""
    in_path = BASE_DIR / "data" / "processed" / _dataset_name / "train_features_cyc.parquet"
    out_path = BASE_DIR / "data" / "processed" / _dataset_name / "train_features_cyc_lag.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(
            f"Input fehlt: {in_path}\nBitte vorher feature_engineering.py und cyclical_encoder.py ausführen."
        )

    print(f"[lag_features] Lade {in_path} ...")
    df = pd.read_parquet(in_path)
    print(f"[lag_features] Geladen: {len(df):,} Zeilen")

    # === 1. Filter zu kurze Gruppen ===
    min_group_length = LAG_CONF.get("min_group_length")
    if min_group_length:
        df = filter_short_groups(df, min_group_length)

    # === 2. Lag/Rolling-Features erstellen ===
    df_out = add_lag_features(df)

    # === 3. Speichern ===
    df_out.to_parquet(out_path, index=False)
    print(f"[lag_features] ✓ Gespeichert: {out_path} (Zeilen: {len(df_out):,})")

    # Summary
    lag_cols = [col for col in df_out.columns if col.startswith(LAG_CONF["prefix"])]
    missing_cols = [col for col in df_out.columns if col.endswith("_missing")]
    print(f"[lag_features] Lag-Features: {len(lag_cols) - len(missing_cols)}")
    print(f"[lag_features] Missing-Indikatoren: {len(missing_cols)}")


if __name__ == "__main__":
    main()

# Aufruf einzeln:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.lag_features
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.lag_features
#
# Via Pipeline:
#   python -m src.pipeline --dataset configs/datasets/walmart.yaml --steps preprocessing
#   python -m src.pipeline --dataset configs/datasets/booksales.yaml --steps preprocessing