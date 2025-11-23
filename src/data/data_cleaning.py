# src/data/data_cleaning.py
# Zweck: Behandlung von Ausreißern und fehlenden Werten in den Verkaufsdaten

from pathlib import Path
import pandas as pd
import numpy as np

from src.config import INTERIM_DIR, BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema

# Lade Config
_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]
_schema = get_schema(_dataset_config)

TARGET_COL = _schema["target_col"]
GROUP_COLS = _schema["id_cols"]
TIME_COL = _schema["time_col"]

class DataCleaner:
    """Bereinigt offensichtliche Ausreißer und ersetzt Werte durch
    gleitende Mittelwerte ähnlicher Zeitpunkte (Booksales-spezifisch)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # Zielspalte aus globaler Config.
        # Hinweis: Aktuell ist TARGET_COL = "num_sold". Wenn der Target-Name
        # geändert wird (z. B. nach einem Merge), greift das hier automatisch.
        self.target_col: str = TARGET_COL

        if not pd.api.types.is_datetime64_any_dtype(self.df[TIME_COL]):
            self.df[TIME_COL] = pd.to_datetime(self.df[TIME_COL], errors="coerce")

            # wichtig: sortieren nach Gruppen + Datum
        self.df = self.df.sort_values(GROUP_COLS + [TIME_COL])
        self.df = self.df.set_index(TIME_COL)

        if "is_lockdown_period" not in self.df.columns:
            self.df["is_lockdown_period"] = 0

        # merke dir die Gruppen
        self.group_cols = GROUP_COLS

    def handle_single_day_outlier(self, date_str: str) -> None:
        """Setzt den Wert an einem bestimmten Datum auf NaN (Einzelausreißer)."""
        target_date = pd.Timestamp(date_str)
        if target_date in self.df.index:
            self.df.loc[target_date, self.target_col] = np.nan

    def handle_lockdown_period(self, start_date: str, end_date: str) -> None:
        """
        Setzt einen Zeitraum (z.B. März–Mai 2020) auf NaN und markiert Lockdown.

        Args:
            start_date: Startdatum als String (z.B. "2020-03-15")
            end_date: Enddatum als String (z.B. "2020-05-31")
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        mask = (self.df.index >= start) & (self.df.index <= end)

        # Lockdown-Flag setzen (bleibt als Feature erhalten)
        self.df.loc[mask, "is_lockdown_period"] = 1
        # Zielwerte auf NaN setzen (werden später geglättet)
        self.df.loc[mask, self.target_col] = np.nan

    def _fill_with_shifted_mean(self, periods: int, repeats: int = 3) -> None:
        """
        Ersetzt NaN-Werte durch Mittelwerte über verschobene Zeitfenster,
        berechnet gruppenweise pro (country, store, product).
        """
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

    def clean(self, outlier_dates: list = None, lockdown_start: str = None, lockdown_end: str = None) -> pd.DataFrame:
        """
        Führt Bereinigungen durch.

        Args:
            outlier_dates: Liste von Outlier-Dates (z.B. ["2020-01-01"])
            lockdown_start: Optional - Lockdown-Startdatum
            lockdown_end: Optional - Lockdown-Enddatum
        """
        # 1) Outlier-Dates behandeln
        if outlier_dates:
            for date_str in outlier_dates:
                self.handle_single_day_outlier(date_str)
            self._fill_with_shifted_mean(periods=365, repeats=3)

        # 2) Lockdown-Periode (nur wenn beide Parameter gesetzt sind)
        if lockdown_start and lockdown_end:
            print(f"[DataCleaner] Lockdown-Periode: {lockdown_start} bis {lockdown_end}")
            self.handle_lockdown_period(lockdown_start, lockdown_end)
            self._fill_with_shifted_mean(periods=365, repeats=3)
        else:
            print("[DataCleaner] Keine Lockdown-Periode konfiguriert")

        return self.df.reset_index()


def main() -> None:
    interim_dir = BASE_DIR / "data" / "interim" / _dataset_name

    if (interim_dir / "train_aligned.parquet").exists():
        parquet_path = interim_dir / "train_aligned.parquet"
    elif (interim_dir / "train_raw.parquet").exists():
        parquet_path = interim_dir / "train_raw.parquet"
    else:
        raise FileNotFoundError(f"Keine Input-Datei in {interim_dir}")

    cleaned_path = interim_dir / "train_cleaned.parquet"

    # NEU: Parameter aus YAML laden
    cleaning_step = next(
        (step for step in _dataset_config.get("preprocessing", [])
         if step["step"] == "cleaning"),
        None
    )

    if not cleaning_step:
        raise ValueError(
            "Cleaning-Step nicht in Dataset-Config gefunden.\n"
            "Bitte 'cleaning' in configs/datasets/*.yaml unter 'preprocessing' definieren."
        )

    params = cleaning_step.get("params", {})
    outlier_dates = params.get("outlier_dates", [])
    lockdown_start = params.get("lockdown_start")
    lockdown_end = params.get("lockdown_end")

    df = pd.read_parquet(parquet_path)
    cleaner = DataCleaner(df)
    df_cleaned = cleaner.clean(
        outlier_dates=outlier_dates,
        lockdown_start=lockdown_start,
        lockdown_end=lockdown_end
    )

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_parquet(cleaned_path, index=False)
    print(f"✓ Bereinigte Datei gespeichert: {cleaned_path}  (Zeilen: {len(df_cleaned):,})")


if __name__ == "__main__":
    main()

# Aufruf einzeln:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.data_cleaning
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.data_cleaning
# Via Pipeline:
#   python -m src.pipeline --dataset configs/datasets/walmart.yaml --steps preprocessing
#   python -m src.pipeline --dataset configs/datasets/booksales.yaml --steps preprocessing