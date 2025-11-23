# src/data/feature_engineering.py
# Zweck: Feature Engineering für TFT – Kalender- & Feiertagsfeatures, Zeitindex

from pathlib import Path
import pandas as pd
import holidays

from src.config import INTERIM_DIR, PROCESSED_DIR, BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema, get_preprocessing_params

_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]
_schema = get_schema(_dataset_config)
_fe_params = get_preprocessing_params(_dataset_config, "feature_engineering")

TIME_COL = _schema["time_col"]


class FeatureEngineer:
    """Erzeugt zeitliche Features für TFT:
    - Kalendermerkmale (Jahr, Monat, Wochentag, KW, Wochenende)
    - Zeitindex (time_idx)
    - gesamtdeutsches Feiertagsflag (is_holiday_de) + optional holiday_name
    """

    def __init__(self, date_col: str, country: str = "DE", include_holiday_name: bool = False):
        self.country = country
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

    def add_holiday_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feiertage basierend auf self.country."""
        out = self._ensure_datetime(df)

        years = out[self.date_col].dt.year.unique().tolist()

        # Country-spezifische Holidays
        if self.country == "DE":
            country_holidays = holidays.Germany(years=years, subdiv=None)
        elif self.country == "US":
            country_holidays = holidays.UnitedStates(years=years)
        elif self.country == "EU":
            # Gemeinsame EU-Feiertage (vereinfacht)
            country_holidays = holidays.Germany(years=years, subdiv=None)
        else:
            raise ValueError(f"Unbekanntes Country: {self.country}")

        is_holiday = out[self.date_col].dt.date.map(lambda d: d in country_holidays)
        out["is_holiday"] = is_holiday.astype("int8")

        if self.include_holiday_name:
            out["holiday_name"] = out[self.date_col].dt.date.map(country_holidays.get)

        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out = self.add_calendar_features(out)
        out = self.add_time_index(out)
        out = self.add_holiday_features(out)
        return out


def main() -> None:
    inp = BASE_DIR / "data" / "interim" / _dataset_name / "train_cleaned.parquet"
    outp = BASE_DIR / "data" / "processed" / _dataset_name / "train_features.parquet"
    outp.parent.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise FileNotFoundError(
            f"Input fehlt: {inp}\nBitte vorher Alignment und Cleaning ausführen."
        )

    df = pd.read_parquet(inp)

    country = _fe_params.get("country", "DE")
    include_holiday_name = _fe_params.get("include_holiday_name", False)

    fe = FeatureEngineer(date_col=TIME_COL, country=country, include_holiday_name=include_holiday_name)
    df_feats = fe.transform(df)

    df_feats.to_parquet(outp, index=False)
    print(f"✓ Features gespeichert: {outp}  (Zeilen: {len(df_feats):,})")


if __name__ == "__main__":
    main()

# Aufruf einzeln:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.feature_engineering
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.feature_engineering
#
# Via Pipeline:
#   python -m src.pipeline --dataset configs/datasets/walmart.yaml --steps preprocessing
#   python -m src.pipeline --dataset configs/datasets/booksales.yaml --steps preprocessing