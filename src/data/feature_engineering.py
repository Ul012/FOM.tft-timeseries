# src/data/feature_engineering.py
# Zweck: Feature Engineering für TFT – Kalender- & Feiertagsfeatures, Zeitindex

from pathlib import Path
import pandas as pd
import numpy as np

try:
    import holidays
except ImportError as e:
    raise ImportError(
        "Das Paket 'holidays' fehlt. Bitte installieren mit: pip install holidays"
    ) from e


class FeatureEngineer:
    """Erzeugt zeitliche Features für TFT:
    - Kalendermerkmale (Jahr, Monat, Wochentag, KW, Wochenende)
    - Zeitindex (time_idx)
    - gesamtdeutsches Feiertagsflag (is_holiday_de) + optional holiday_name
    """

    def __init__(self, date_col: str = "date", include_holiday_name: bool = False):
        self.date_col = date_col
        self.include_holiday_name = include_holiday_name

    def _ensure_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(out[self.date_col]):
            out[self.date_col] = pd.to_datetime(out[self.date_col], errors="coerce")
        return out

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._ensure_datetime(df)
        dt = out[self.date_col].dt
        out["year"] = dt.year
        out["month"] = dt.month
        out["day"] = dt.day
        out["dayofweek"] = dt.dayofweek  # Montag=0 … Sonntag=6
        # Kalenderwoche: ISO-Woche (1–53)
        out["weekofyear"] = out[self.date_col].dt.isocalendar().week.astype("int64")
        out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype("int8")
        return out

    def add_time_index(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._ensure_datetime(df).sort_values(self.date_col)
        # Zeitindex als fortlaufende Integer-Skala (tägliche Frequenz → ein Index pro Datum)
        # Falls mehrere Reihen pro Datum (z. B. Länder), gilt der gleiche time_idx
        first_date = out[self.date_col].min()
        out["time_idx"] = (out[self.date_col] - first_date).dt.days.astype("int64")
        return out

    def add_holiday_features_de(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gesamtdeutsche Feiertage (bundesweit). Keine Länderspezifika.
        Hinweis: Einmalige bundesweite Ausnahmen (z. B. Reformationstag 2017) werden korrekt markiert.
        """
        out = self._ensure_datetime(df)

        years = out[self.date_col].dt.year.unique().tolist()
        # subdivisions=None → nur bundesweite Feiertage
        de_holidays = holidays.Germany(years=years, subdiv=None)  # observed: Default

        # Schnelles Lookup: Bool-Flag pro Datum
        is_holiday = out[self.date_col].dt.date.map(lambda d: d in de_holidays)
        out["is_holiday_de"] = is_holiday.astype("int8")

        if self.include_holiday_name:
            # Namen für Debug/Erklärung (NaN, wenn kein Feiertag)
            out["holiday_name"] = out[self.date_col].dt.date.map(de_holidays.get)

        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = self.add_calendar_features(out)
        out = self.add_time_index(out)
        out = self.add_holiday_features_de(out)
        return out


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    inp = base_dir / "data" / "interim" / "train_cleaned.parquet"
    outp = base_dir / "data" / "processed" / "train_features.parquet"
    outp.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise FileNotFoundError(
            f"Input fehlt: {inp}\nBitte vorher Alignment und Cleaning ausführen."
        )

    df = pd.read_parquet(inp)
    fe = FeatureEngineer(date_col="date", include_holiday_name=False)  # Namen optional
    df_feats = fe.transform(df)

    df_feats.to_parquet(outp, index=False)
    print(f"✓ Features gespeichert: {outp}  (Zeilen: {len(df_feats):,})")


if __name__ == "__main__":
    main()

# python -m src.data.feature_engineering