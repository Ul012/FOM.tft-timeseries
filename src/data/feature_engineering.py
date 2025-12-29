# src/data/feature_engineering.py
# Zweck: Feature Engineering für TFT – Kalender- & Feiertagsfeatures, Zeitindex

from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional
import holidays

from src.config import INTERIM_DIR, PROCESSED_DIR, BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema, get_preprocessing_params

_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]
_schema = get_schema(_dataset_config)
_fe_params = get_preprocessing_params(_dataset_config, "feature_engineering")

TIME_COL = _schema["time_col"]
ID_COLS = _schema["id_cols"]


class FeatureEngineer:
    """Erzeugt zeitliche Features für TFT:
    - Kalendermerkmale (Jahr, Monat, Wochentag, KW, Wochenende)
    - Zeitindex (time_idx)
    - gesamtdeutsches Feiertagsflag (is_holiday_de) + optional holiday_name
    """

    def __init__(
            self,
            date_col: str,
            country: str = "DE",
            include_holiday_name: bool = False,
            date_flags: Dict[str, List[Dict[str, int]]] = None,
            id_cols: List[str] = None,
            for_models: Optional[List[str]] = None,  # ["tft", "prophet"]
    ):
        self.country = country
        self.date_col = date_col
        self.include_holiday_name = include_holiday_name
        self.date_flags = date_flags or {}
        self.id_cols = id_cols or []
        self.for_models = for_models or ["tft"]  # Default: nur TFT

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
        if "tft" not in self.for_models:
            print("[FeatureEngineer] time_idx übersprungen")
            return df

        out = self._ensure_datetime(df).sort_values(self.date_col)
        # Fortlaufender Index basierend auf unique Dates (funktioniert für täglich UND wöchentlich)
        unique_dates = out[self.date_col].drop_duplicates().sort_values().reset_index(drop=True)
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}
        out["time_idx"] = out[self.date_col].map(date_to_idx).astype("int64")
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

    def add_date_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fügt Custom Date Flags aus YAML hinzu.

        YAML-Format:
            date_flags:
              is_newyear:
                - {month: 12, day_start: 27, day_end: 31}
                - {month: 1, day_start: 1, day_end: 2}
        """
        if not self.date_flags:
            return df

        out = self._ensure_datetime(df)
        month = out[self.date_col].dt.month
        day = out[self.date_col].dt.day

        for flag_name, periods in self.date_flags.items():
            mask = pd.Series(False, index=out.index)

            for period in periods:
                m = period.get("month")
                d_start = period.get("day_start", 1)
                d_end = period.get("day_end", 31)

                period_mask = (month == m) & (day >= d_start) & (day <= d_end)
                mask = mask | period_mask

            out[flag_name] = mask.astype("int8")
            print(f"  - {flag_name}: {mask.sum():,} Zeilen markiert")

        return out

    def _convert_id_cols_to_string(self, df: pd.DataFrame) -> pd.DataFrame:
        """ID-Spalten zu String  (für TFT)"""
        if "tft" not in self.for_models:
            print("[FeatureEngineer] ID-String-Konvertierung übersprungen")
            return df

        out = df.copy()
        for col in self.id_cols:
            if col in out.columns and out[col].dtype in ["int64", "int32", "float64"]:
                out[col] = out[col].astype(str)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._ensure_datetime(df)

        # ID-Spalten zu String konvertieren (für PyTorch Forecasting)
        for col in self.id_cols:
            if col in out.columns and out[col].dtype in ["int64", "int32", "float64"]:
                out[col] = out[col].astype(str)
                print(f"  - {col}: zu String konvertiert")

        out = self._convert_id_cols_to_string(out)
        out = self.add_calendar_features(out)
        out = self.add_time_index(out)
        out = self.add_holiday_features(out)
        out = self.add_date_flags(out)
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
    date_flags = _fe_params.get("date_flags", {})
    for_models = _fe_params.get("for_models", ["tft"])

    if date_flags:
        print(f"[feature_engineering] Date Flags: {list(date_flags.keys())}")

    fe = FeatureEngineer(
        date_col=TIME_COL,
        country=country,
        include_holiday_name=include_holiday_name,
        date_flags=date_flags,
        id_cols=ID_COLS,
        for_models=for_models
    )
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