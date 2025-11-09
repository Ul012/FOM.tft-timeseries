# src/data/data_cleaning.py
# Zweck: Behandlung von Ausreißern und fehlenden Werten in den Verkaufsdaten

from pathlib import Path
import pandas as pd
import numpy as np


class DataCleaner:
    """Bereinigt offensichtliche Ausreißer und ersetzt Werte durch
    gleitende Mittelwerte ähnlicher Zeitpunkte."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.df["date"]):
            self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df = self.df.set_index("date")

        # neues Lockdown-Flag initialisieren
        if "is_lockdown_period" not in self.df.columns:
            self.df["is_lockdown_period"] = 0

    def handle_single_day_outlier(self, date_str: str) -> None:
        """Setzt Ausreißer für einen bestimmten Tag auf NaN."""
        target_date = pd.Timestamp(date_str)
        if target_date in self.df.index:
            self.df.loc[target_date, "num_sold"] = np.nan

    def handle_lockdown_period(self, year: int, months=(3, 4, 5)) -> None:
        """Setzt Werte in einem gegebenen Jahr und Zeitraum auf NaN."""
        mask = (self.df.index.year == year) & (self.df.index.month.isin(months))
        # Lockdown-Flag setzen (bleibt bestehen, auch nach Imputation)
        self.df.loc[mask, "is_lockdown_period"] = 1
        # Zielwerte auf NaN setzen (werden danach geglättet aufgefüllt)
        self.df.loc[mask, "num_sold"] = np.nan

    def _fill_with_shifted_mean(self, periods: int, repeats: int = 3) -> None:
        """Ersetzt NaN-Werte durch Mittelwert über verschobene Zeitfenster."""
        df_shifted = pd.concat(
            [self.df[["num_sold"]].shift(periods=periods * x) for x in range(repeats)],
            axis=1,
        )
        self.df["num_sold"] = self.df["num_sold"].fillna(
            df_shifted.mean(axis=1)
        )

    def clean(self) -> pd.DataFrame:
        """Führt alle Bereinigungsschritte aus."""
        # 1. Einzelner Ausreißer 01.01.2020
        self.handle_single_day_outlier("2020-01-01")
        self._fill_with_shifted_mean(periods=365 * 48)

        # 2. Lockdown-Monate März–Mai 2020
        self.handle_lockdown_period(year=2020, months=(3, 4, 5))
        self._fill_with_shifted_mean(periods=52 * 7 * 48)

        return self.df.reset_index()


def main() -> None:
    """Lädt Parquet-Datei, bereinigt sie und speichert Ergebnis."""
    base_dir = Path(__file__).resolve().parents[2]
    parquet_path = base_dir / "data" / "interim" / "train_aligned.parquet"
    cleaned_path = base_dir / "data" / "interim" / "train_cleaned.parquet"

    df = pd.read_parquet(parquet_path)
    cleaner = DataCleaner(df)
    df_cleaned = cleaner.clean()

    df_cleaned.to_parquet(cleaned_path, index=False)
    print(f"✓ Bereinigte Datei gespeichert: {cleaned_path}  (Zeilen: {len(df_cleaned):,})")


if __name__ == "__main__":
    main()

# python -m src.data.data_cleaning