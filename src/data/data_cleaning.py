# src/data/data_cleaning.py
# Zweck: Behandlung von Ausreißern und fehlenden Werten in den Verkaufsdaten

from pathlib import Path
import pandas as pd
import numpy as np

from src.config import INTERIM_DIR, TARGET_COL

class DataCleaner:
    """Bereinigt offensichtliche Ausreißer und ersetzt Werte durch
    gleitende Mittelwerte ähnlicher Zeitpunkte (Booksales-spezifisch)."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

        # Zielspalte aus globaler Config.
        # Hinweis: Aktuell ist TARGET_COL = "num_sold". Wenn der Target-Name
        # geändert wird (z. B. nach einem Merge), greift das hier automatisch.
        self.target_col: str = TARGET_COL

        if not pd.api.types.is_datetime64_any_dtype(self.df["date"]):
            self.df["date"] = pd.to_datetime(self.df["date"], errors="coerce")

            # wichtig: sortieren nach Gruppen + Datum
        self.df = self.df.sort_values(["country", "store", "product", "date"])
        self.df = self.df.set_index("date")

        if "is_lockdown_period" not in self.df.columns:
            self.df["is_lockdown_period"] = 0

        # merke dir die Gruppen
        self.group_cols = ["country", "store", "product"]

    def handle_single_day_outlier(self, date_str: str) -> None:
        """Setzt den Wert an einem bestimmten Datum auf NaN (Einzelausreißer)."""
        target_date = pd.Timestamp(date_str)
        if target_date in self.df.index:
            self.df.loc[target_date, self.target_col] = np.nan

    def handle_lockdown_period(self, year: int, months: tuple[int, ...]) -> None:
        """Setzt einen Zeitraum (z. B. März–Mai 2020) auf NaN und markiert Lockdown."""
        mask = (self.df.index.year == year) & (self.df.index.month.isin(months))
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

    def clean(self) -> pd.DataFrame:
        # 1) Outlier 01.01.2020 -> Jahreswerte
        self.handle_single_day_outlier("2020-01-01")
        self._fill_with_shifted_mean(periods=365, repeats=3)

        # 2) Lockdown März–Mai 2020 -> ebenfalls Jahreswerte
        self.handle_lockdown_period(year=2020, months=(3, 4, 5))
        self._fill_with_shifted_mean(periods=365, repeats=3)

        return self.df.reset_index()


def main() -> None:
    """Lädt die ausgerichteten Daten, bereinigt sie und speichert das Ergebnis."""
    parquet_path = INTERIM_DIR / "train_aligned.parquet"
    cleaned_path = INTERIM_DIR / "train_cleaned.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet-Datei nicht gefunden: {parquet_path}\n"
            "Bitte zuerst data_alignment.py ausführen (oder passenden Input bereitstellen)."
        )

    df = pd.read_parquet(parquet_path)
    cleaner = DataCleaner(df)
    df_cleaned = cleaner.clean()

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_parquet(cleaned_path, index=False)
    print(f"✓ Bereinigte Datei gespeichert: {cleaned_path}  (Zeilen: {len(df_cleaned):,})")


if __name__ == "__main__":
    # python -m src.data.data_cleaning
    main()