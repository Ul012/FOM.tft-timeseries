# src/data/lag_features.py

import pandas as pd
from src.config import PROCESSED_DIR, LAG_CONF, GROUP_COLS, TIME_COL


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Erzeugt Lag- und optionale Rolling-Features basierend auf LAG_CONF."""
    target = LAG_CONF["target_col"]
    lags = LAG_CONF["lags"]
    roll_windows = LAG_CONF.get("roll_windows", [])
    roll_stats = LAG_CONF.get("roll_stats", [])
    prefix = LAG_CONF.get("prefix", "lag_")

    # Nach Gruppe und Zeit sortieren
    df = df.sort_values(GROUP_COLS + [TIME_COL]).copy()

    # Lag-Features
    for lag in lags:
        df[f"{prefix}{lag}"] = df.groupby(GROUP_COLS)[target].shift(lag)

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
    # python -m src.data.lag_features
    main()
