# src/data/cyclical_encoder.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

# Direkte Imports, kein try/except – schlank und pythonic
from src.config import PROCESSED_DIR


@dataclass(frozen=True)
class CyclicalEncoderConfig:
    datetime_col: str = "date"
    periodicities: Dict[str, Tuple[str, int]] = field(default_factory=lambda: {
        "dow": ("dow", 7),
        "month": ("month", 12),
        "doy": ("doy", 366),
        "week": ("week", 53),
        "hour": ("hour", 24),
    })
    prefix: str = "cyc"
    drop_source_cols: bool = True
    tz: Optional[str] = "Europe/Berlin"
    coerce_invalid: bool = True


class CyclicalEncoder:
    """Erzeugt Sin/Cos-Features für zyklische Zeitmerkmale."""

    def __init__(self, config: Optional[CyclicalEncoderConfig] = None) -> None:
        self.cfg = config or CyclicalEncoderConfig()

    @staticmethod
    def _ensure_datetime(series: pd.Series, tz: Optional[str], coerce_invalid: bool) -> pd.Series:
        s = pd.to_datetime(series, errors=("coerce" if coerce_invalid else "raise"))
        if tz is not None:
            if s.dt.tz is None:
                s = s.dt.tz_localize("UTC").dt.tz_convert(tz)
            else:
                s = s.dt.tz_convert(tz)
        return s

    @staticmethod
    def _extract(values: pd.Series, kind: str) -> pd.Series:
        mask = values.notna()
        out = pd.Series(np.nan, index=values.index, dtype="float64")

        if kind == "dow":
            out.loc[mask] = values.loc[mask].dt.dayofweek.astype("int64")
        elif kind == "month":
            out.loc[mask] = values.loc[mask].dt.month.astype("int64") - 1
        elif kind == "doy":
            out.loc[mask] = values.loc[mask].dt.dayofyear.astype("int64") - 1
        elif kind == "week":
            iso_week = values.loc[mask].dt.isocalendar().week.astype("int64") - 1
            out.loc[mask] = iso_week.to_numpy()
        elif kind == "hour":
            out.loc[mask] = values.loc[mask].dt.hour.astype("int64")
        else:
            raise ValueError(f"Unbekannter extractor: {kind}")

        return out

    @staticmethod
    def _to_sin_cos(x: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
        angle = 2.0 * np.pi * (x / float(period))
        return np.sin(angle), np.cos(angle)

    def fit(self, df: pd.DataFrame) -> "CyclicalEncoder":
        if self.cfg.datetime_col not in df.columns:
            raise KeyError(f"datetime_col '{self.cfg.datetime_col}' fehlt.")
        _ = self._ensure_datetime(df[self.cfg.datetime_col], self.cfg.tz, self.cfg.coerce_invalid)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        out = df.copy()

        dt = self._ensure_datetime(out[cfg.datetime_col], cfg.tz, cfg.coerce_invalid)

        source_cols: List[str] = []
        for name, (kind, period) in cfg.periodicities.items():
            idx = self._extract(dt, kind)
            src_col = f"{cfg.prefix}_{name}_idx"
            out[src_col] = idx

            sin_col = f"{cfg.prefix}_{name}_sin"
            cos_col = f"{cfg.prefix}_{name}_cos"
            sin_vals, cos_vals = self._to_sin_cos(idx, period)
            out[sin_col], out[cos_col] = sin_vals, cos_vals

            source_cols.append(src_col)

        if cfg.drop_source_cols:
            out.drop(columns=source_cols, inplace=True)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def main() -> None:
    in_path = PROCESSED_DIR / "train_features.parquet"
    out_path = PROCESSED_DIR / "train_features_cyc.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cyclical_encoder] Lade {in_path} ...")
    df = pd.read_parquet(in_path)

    enc = CyclicalEncoder()
    out = enc.fit_transform(df)

    out.to_parquet(out_path, index=False)
    print(f"[cyclical_encoder] ✓ geschrieben: {out_path} (Zeilen: {len(out):,})")


if __name__ == "__main__":
    # python -m src.data.cyclical_encoder
    main()
