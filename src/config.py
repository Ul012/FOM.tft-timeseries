# src/config.py
# Zentrale Projektkonfiguration – bewusst schlank und YAML-freundlich.
# Enthält nur projektweite Konstanten (Pfade, Schema, Split).
# KEINE Trainings-/Modell-Hyperparameter mehr (die stehen in configs/*.yaml).

from pathlib import Path

# -----------------------------------------------------------------------------
# Verzeichnisse
# -----------------------------------------------------------------------------
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# -----------------------------------------------------------------------------
# Spalten / Schema
# -----------------------------------------------------------------------------
DATETIME_COLUMN: str = "date"
GROUP_COLS: list[str] = ["country", "store", "product"]

# Einheitliche Zielspalte im gesamten Projekt.
# Hinweis: Falls durch einen Merge temporär num_sold_y entsteht,
# bitte direkt wieder auf num_sold zurückbenennen (df.rename(columns={"num_sold_y": "num_sold"})).
TARGET_COL: str = "num_sold"

# Aliase für Konsistenz im Code
TIME_COL: str = DATETIME_COLUMN
ID_COLS: list[str] = GROUP_COLS

# -----------------------------------------------------------------------------
# Pfade für Feature-/Dataset-Erzeugung
# -----------------------------------------------------------------------------

# Upstream-Features aus Schritt 3A (optional genutzt)
FEATURES_TRAIN_PATH: Path = PROCESSED_DIR / "train_features.parquet"

# Input für model_dataset.py (Ergebnis aus 3C: lag_features)
MODEL_INPUT_PATH: Path = PROCESSED_DIR / "train_features_cyc_lag.parquet"

# -----------------------------------------------------------------------------
# Split-Parameter (zeitbasiert)
# Entweder feste Grenzen (VAL_START/TEST_START) ODER SPLIT_RATIOS verwenden.
# -----------------------------------------------------------------------------
VAL_START: str | None = None         # z. B. "2020-04-01"
TEST_START: str | None = None        # z. B. "2020-08-01"
SPLIT_RATIOS: tuple[float, float, float] = (0.80, 0.10, 0.10)

# Optional: gruppenweise Skalierung (falls in der Pipeline genutzt)
SCALE_COLS: list[str] = []


# -----------------------------------------------------------------------------
# Lag-Features (für Zeitbezug des TFT und anderer Modelle)
# -----------------------------------------------------------------------------
LAG_CONF: dict = {
    "target_col": TARGET_COL,        # übernimmt die globale Zielspalte
    "lags": [1, 7, 14],              # zeitliche Rückblicke
    "roll_windows": [7],             # optionale Rolling-Fenster
    "roll_stats": ["mean"],          # z. B. Mittelwert über 7 Tage
    "prefix": "lag_",                # muss zu dataset_tft.py passen
}

# -----------------------------------------------------------------------------
# TFT-Dataset-Metadaten (für Feature-Pipeline / Dataset-Bau)
# -----------------------------------------------------------------------------
TFT_DATASET: dict = {
    "max_encoder_length": 28,
    "max_prediction_length": 7,
    # bekannte reelle Features (typisch: Kalenderzyklen), werden als "known" behandelt
    "known_real_prefixes": ["cyc_"],        # z. B. cyc_dow_sin/cos, cyc_month_sin/cos
    # Lag-Features: Spalten-Prefix für von dir erzeugte Lags
    "lag_prefixes": ["lag_"],               # z. B. lag_num_sold_7, lag_num_sold_28
    # Kalenderfelder (year, month, day, is_weekend, is_holiday_*)
    "treat_calendar_as_known": True,
    # explizite Flags (0/1)
    "flag_cols": ["is_lockdown_period"],
}