# src/data/lag_features.py

import pandas as pd
from pathlib import Path
from src.config import PROCESSED_DIR, LAG_CONF, GROUP_COLS, TIME_COL

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Erzeugt Lag-Features basierend auf den Einstellungen in LAG_CONF."""
    target = LAG_CONF["target_col"]
    lags = LAG_CONF["lags"]
    roll_windows = LAG_CONF.get("roll_windows", [])
    roll_stats = LAG_CONF.get("roll_stats", [])
    prefix = LAG_CONF.get("prefix", "lag_")

    # Nach Zeit und Gruppe sortieren
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
    import pandas as pd
    from pathlib import Path
    from src.config import LAG_CONF, GROUP_COLS, TIME_COL  # nur inhaltliche Settings aus config

    # robust relativ zur Datei, nicht zum Working Directory
    base_dir = Path(__file__).resolve().parents[2]
    in_path = base_dir / "data" / "processed" / "train_features_cyc.parquet"
    out_path = base_dir / "data" / "processed" / "train_features_cyc_lag.parquet"

    print(f"[lag_features] Lade {in_path} ...")
    df = pd.read_parquet(in_path)

    target = LAG_CONF.get("target_col", "num_sold")
    lags = tuple(LAG_CONF.get("lags", (1, 7, 14)))
    roll_windows = tuple(LAG_CONF.get("roll_windows", ()))  # optional
    roll_stats = tuple(LAG_CONF.get("roll_stats", ()))      # optional
    prefix = str(LAG_CONF.get("prefix", "lag_"))

    # sortiert für stabile Shifts
    sort_cols = (list(GROUP_COLS) if GROUP_COLS else []) + [TIME_COL]
    df = df.sort_values(sort_cols).copy()

    # Lags
    if GROUP_COLS:
        g = df.groupby(GROUP_COLS)[target]
        for L in lags:
            df[f"{prefix}{L}"] = g.shift(L)
    else:
        for L in lags:
            df[f"{prefix}{L}"] = df[target].shift(L)

    # Optional: Rolling-Mean (sparsam)
    if roll_windows:
        if GROUP_COLS:
            gshift = df.groupby(GROUP_COLS)[target].shift(1)
            for W in roll_windows:
                if (not roll_stats) or ("mean" in roll_stats):
                    r = gshift.groupby(df[GROUP_COLS].apply(tuple, axis=1)).rolling(W, min_periods=1).mean()
                    df[f"{prefix}{W}_mean"] = r.reset_index(level=0, drop=True)
        else:
            s = df[target].shift(1)
            for W in roll_windows:
                if (not roll_stats) or ("mean" in roll_stats):
                    df[f"{prefix}{W}_mean"] = s.rolling(W, min_periods=1).mean()

    df.to_parquet(out_path, index=False)
    print(f"[lag_features] Fertig. Gespeichert: {out_path}")

if __name__ == "__main__":
    main()
