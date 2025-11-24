# src/data/data_cleaning.py
"""
Datenbereinigung für Zeitreihen-Forecasting.

Features:
- Einzelne Outlier-Dates behandeln
- Lockdown-Perioden markieren und interpolieren
- Target-Werte clippen (negative → 0)
- Target-NaN entfernen
- Target auf float32 casten

Aufruf:
    $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.data_cleaning
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.data_cleaning
"""

from pathlib import Path
import pandas as pd
import numpy as np

from src.config import INTERIM_DIR, BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema, get_preprocessing_params

# Lade Config
_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]
_schema = get_schema(_dataset_config)

TARGET_COL = _schema["target_col"]
GROUP_COLS = _schema["id_cols"]
TIME_COL = _schema["time_col"]


class DataCleaner:
    """Bereinigt Ausreißer, fehlende Werte und Datenqualitätsprobleme."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.target_col: str = TARGET_COL

        if not pd.api.types.is_datetime64_any_dtype(self.df[TIME_COL]):
            self.df[TIME_COL] = pd.to_datetime(self.df[TIME_COL], errors="coerce")

        # Sortieren nach Gruppen + Datum
        self.df = self.df.sort_values(GROUP_COLS + [TIME_COL])
        self.df = self.df.set_index(TIME_COL)

        if "is_lockdown_period" not in self.df.columns:
            self.df["is_lockdown_period"] = 0

        self.group_cols = GROUP_COLS

    def handle_single_day_outlier(self, date_str: str) -> None:
        """Setzt den Wert an einem bestimmten Datum auf NaN (Einzelausreißer)."""
        target_date = pd.Timestamp(date_str)
        if target_date in self.df.index:
            self.df.loc[target_date, self.target_col] = np.nan

    def handle_lockdown_period(self, start_date: str, end_date: str) -> None:
        """Setzt einen Zeitraum auf NaN und markiert Lockdown."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        mask = (self.df.index >= start) & (self.df.index <= end)
        self.df.loc[mask, "is_lockdown_period"] = 1
        self.df.loc[mask, self.target_col] = np.nan

    def _fill_with_shifted_mean(self, periods: int, repeats: int = 3) -> None:
        """Ersetzt NaN-Werte durch Mittelwerte über verschobene Zeitfenster."""
        shifted_series = []
        for x in range(repeats):
            shifted = (
                self.df
                .groupby(self.group_cols)[self.target_col]
                .shift(periods=periods * x)
            )
            shifted_series.append(shifted)

        df_shifted = pd.concat(shifted_series, axis=1)
        self.df[self.target_col] = self.df[self.target_col].fillna(df_shifted.mean(axis=1))

    def clip_target(self, clip_min: float = None, clip_max: float = None) -> None:
        """
        Clippt Target-Werte auf einen Bereich.

        Args:
            clip_min: Minimaler Wert (z.B. 0 für keine negativen Verkäufe)
            clip_max: Maximaler Wert (z.B. für Outlier-Capping)
        """
        n_before = len(self.df)

        if clip_min is not None:
            n_below = (self.df[self.target_col] < clip_min).sum()
            if n_below > 0:
                print(f"[DataCleaner] Clippe {n_below} Werte < {clip_min} auf {clip_min}")
                self.df[self.target_col] = self.df[self.target_col].clip(lower=clip_min)

        if clip_max is not None:
            n_above = (self.df[self.target_col] > clip_max).sum()
            if n_above > 0:
                print(f"[DataCleaner] Clippe {n_above} Werte > {clip_max} auf {clip_max}")
                self.df[self.target_col] = self.df[self.target_col].clip(upper=clip_max)

    def remove_target_nan(self) -> None:
        """Entfernt Zeilen mit NaN im Target (TFT erlaubt keine NaN im Target)."""
        n_before = len(self.df)
        n_nan = self.df[self.target_col].isna().sum()

        if n_nan > 0:
            self.df = self.df[self.df[self.target_col].notna()]
            print(f"[DataCleaner] Target-NaN entfernt: {n_before} → {len(self.df)} (-{n_nan})")

    def convert_target_dtype(self) -> None:
        """Konvertiert Target zu float32 (TFT-Anforderung)."""
        self.df[self.target_col] = pd.to_numeric(
            self.df[self.target_col], errors="coerce"
        ).astype("float32")

    def clean(
            self,
            outlier_dates: list = None,
            lockdown_start: str = None,
            lockdown_end: str = None,
            clip_target_min: float = None,
            clip_target_max: float = None,
            remove_nan: bool = True,
    ) -> pd.DataFrame:
        """
        Führt alle Bereinigungen durch.

        Args:
            outlier_dates: Liste von Outlier-Dates (z.B. ["2020-01-01"])
            lockdown_start: Lockdown-Startdatum
            lockdown_end: Lockdown-Enddatum
            clip_target_min: Minimaler Target-Wert (z.B. 0)
            clip_target_max: Maximaler Target-Wert
            remove_nan: Ob Target-NaN entfernt werden sollen (default: True)
        """
        # 1) Outlier-Dates behandeln
        if outlier_dates:
            for date_str in outlier_dates:
                self.handle_single_day_outlier(date_str)
            self._fill_with_shifted_mean(periods=365, repeats=3)

        # 2) Lockdown-Periode
        if lockdown_start and lockdown_end:
            print(f"[DataCleaner] Lockdown-Periode: {lockdown_start} bis {lockdown_end}")
            self.handle_lockdown_period(lockdown_start, lockdown_end)
            self._fill_with_shifted_mean(periods=365, repeats=3)
        else:
            print("[DataCleaner] Keine Lockdown-Periode konfiguriert")

        # 3) Target clippen (negative Werte, Ausreißer)
        if clip_target_min is not None or clip_target_max is not None:
            self.clip_target(clip_min=clip_target_min, clip_max=clip_target_max)

        # 4) Target auf float32 konvertieren
        self.convert_target_dtype()

        # 5) Target-NaN entfernen (nach allen anderen Bereinigungen!)
        if remove_nan:
            self.remove_target_nan()

        return self.df.reset_index()


def main() -> None:
    interim_dir = BASE_DIR / "data" / "interim" / _dataset_name

    # Input-Datei finden (aligned > raw)
    if (interim_dir / "train_aligned.parquet").exists():
        parquet_path = interim_dir / "train_aligned.parquet"
    elif (interim_dir / "train_raw.parquet").exists():
        parquet_path = interim_dir / "train_raw.parquet"
    else:
        # Fallback: Direkt aus Raw-Verzeichnis laden
        raw_path = BASE_DIR / "data" / "raw" / _dataset_name / "train.csv"
        if raw_path.exists():
            print(f"[DataCleaner] Lade direkt aus Raw: {raw_path}")
            df = pd.read_csv(raw_path)
            # Speichere als train_raw.parquet für konsistente Pipeline
            interim_dir.mkdir(parents=True, exist_ok=True)
            raw_parquet = interim_dir / "train_raw.parquet"
            df.to_parquet(raw_parquet, index=False)
            parquet_path = raw_parquet
        else:
            raise FileNotFoundError(f"Keine Input-Datei gefunden in {interim_dir} oder {raw_path}")

    cleaned_path = interim_dir / "train_cleaned.parquet"

    # Parameter aus YAML laden
    cleaning_step = next(
        (step for step in _dataset_config.get("preprocessing", [])
         if step["step"] == "cleaning"),
        None
    )

    if not cleaning_step:
        # Kein Cleaning-Step definiert → nur Basis-Bereinigung
        print("[DataCleaner] Kein 'cleaning' Step in YAML - nur Basis-Bereinigung (dtype, NaN)")
        params = {}
    else:
        params = cleaning_step.get("params", {})

    # Parameter extrahieren
    outlier_dates = params.get("outlier_dates", [])
    lockdown_start = params.get("lockdown_start")
    lockdown_end = params.get("lockdown_end")
    clip_target_min = params.get("clip_target_min")
    clip_target_max = params.get("clip_target_max")
    remove_nan = params.get("remove_nan", True)

    print(f"[DataCleaner] Lade: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"[DataCleaner] Input: {len(df):,} Zeilen")

    cleaner = DataCleaner(df)
    df_cleaned = cleaner.clean(
        outlier_dates=outlier_dates,
        lockdown_start=lockdown_start,
        lockdown_end=lockdown_end,
        clip_target_min=clip_target_min,
        clip_target_max=clip_target_max,
        remove_nan=remove_nan,
    )

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_parquet(cleaned_path, index=False)
    print(f"[DataCleaner] ✓ Gespeichert: {cleaned_path} (Zeilen: {len(df_cleaned):,})")


if __name__ == "__main__":
    main()

# Aufruf einzeln:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.data_cleaning
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.data_cleaning
#
# Via Pipeline:
#   python -m src.pipeline --dataset configs/datasets/walmart.yaml --steps preprocessing
#   python -m src.pipeline --dataset configs/datasets/booksales.yaml --steps preprocessing