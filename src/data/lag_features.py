# src/data/lag_features.py

import pandas as pd
from src.config import PROCESSED_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema, get_preprocessing_params

# Lade Config einmalig
_dataset_config = load_dataset_config()
_schema = get_schema(_dataset_config)
_lag_params = get_preprocessing_params(_dataset_config, "lag_features")

# Extrahiere Werte (wie vorher aus config.py)
GROUP_COLS = _schema["id_cols"]
TIME_COL = _schema["time_col"]
TARGET_COL = _schema["target_col"]

# Lag-Config
LAG_CONF = {
    "target_col": TARGET_COL,
    "lags": _lag_params["lags"],
    "roll_windows": _lag_params.get("roll_windows", []),
    "roll_stats": _lag_params.get("roll_stats", []),
    "prefix": _lag_params.get("prefix", "lag_"),
}


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Erzeugt Lag- und optionale Rolling-Features basierend auf LAG_CONF."""
    target = LAG_CONF["target_col"]
    lags = LAG_CONF["lags"]
    roll_windows = LAG_CONF.get("roll_windows", [])
    roll_stats = LAG_CONF.get("roll_stats", [])
    prefix = LAG_CONF.get("prefix", "lag_")

    # Nach Gruppe und Zeit sortieren
    df = df.sort_values(GROUP_COLS + [TIME_COL]).copy()

    # Lag-Features aus der Config
    for lag in lags:
        df[f"{prefix}{lag}"] = (
            df.groupby(GROUP_COLS)[target]
            .shift(lag)
        )

    # Jahres-Lag für starke Saisonalität
    df[f"{prefix}365"] = (
        df.groupby(GROUP_COLS)[target]
        .shift(365)
    )

    # Rolling-Features (optional)
    for window in roll_windows:
        for stat in roll_stats:
            colname = f"{prefix}{window}_{stat}"
            rolled = df.groupby(GROUP_COLS)[target].transform(
                lambda x: getattr(x.shift(1).rolling(window=window, min_periods=1), stat)()
            )
            df[colname] = rolled

    return df


def main() -> None:
    """Liest train_features_cyc.parquet, erzeugt Lag/Rolling-Features und speichert train_features_cyc_lag.parquet."""
    in_path = PROCESSED_DIR / "train_features_cyc.parquet"
    out_path = PROCESSED_DIR / "train_features_cyc_lag.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(
            f"Input fehlt: {in_path}\nBitte vorher feature_engineering.py und cyclical_encoder.py ausführen."
        )

    print(f"[lag_features] Lade {in_path} ...")
    df = pd.read_parquet(in_path)

    df_out = add_lag_features(df)
    df_out.to_parquet(out_path, index=False)
    print(f"[lag_features] ✓ Gespeichert: {out_path} (Zeilen: {len(df_out):,})")


if __name__ == "__main__":
    main()

# Aufruf einzeln:
#   python -m src.data.lag_features
# Via Pipeline:
#   python -m src.pipeline --dataset configs/datasets/booksales.yaml --steps preprocessing
