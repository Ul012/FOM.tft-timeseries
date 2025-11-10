# src/data/cyclical_encoder.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd

# Optional: zentrale Konfiguration aus src/config.py laden.
# Fallback: Wenn config nicht existiert, läuft der Encoder mit Defaults.
try:
    from src.config import CYCLICAL_CONF as _CYCLICAL_CONF
    _HAS_CONFIG = True
except Exception:
    _CYCLICAL_CONF = None
    _HAS_CONFIG = False


@dataclass(frozen=True)
class CyclicalEncoderConfig:
    """
    Konfiguration für die zyklische Kodierung.

    datetime_col: Name der Zeitspalte.
    periodicities: Mapping feature_name -> (extractor, period)
        extractor ∈ {"dow","month","doy","week","hour","minute","quarter"}
        period: natürliche Periodenlänge (z. B. 7 für dow, 12 für month, 366 für doy).
    prefix: Präfix für erzeugte Spalten.
    drop_source_cols: Integer-Indexspalten nach Erstellung entfernen?
    tz: Zielzeitzone (z. B. "Europe/Berlin") oder None.
    coerce_invalid: Ungültige Datumswerte zu NaT statt Fehler.
    """
    datetime_col: str = "date"
    periodicities: Dict[str, Tuple[str, int]] = field(default_factory=lambda: {
        "dow": ("dow", 7),
        "month": ("month", 12),
        "doy": ("doy", 366),
        "week": ("week", 53),
        "hour": ("hour", 24),
        # "minute": ("minute", 60),
        # "quarter": ("quarter", 4),
    })
    prefix: str = "cyc"
    drop_source_cols: bool = True
    tz: Optional[str] = "Europe/Berlin"
    coerce_invalid: bool = True


class CyclicalEncoder:
    """
    Zyklische Kalendermerkmale → Sin/Cos-Features (zustandslos).
    """

    def __init__(self, config: Optional[CyclicalEncoderConfig] = None) -> None:
        # Wenn Config aus src.config vorhanden ist, nutze sie; sonst Defaults.
        if config is not None:
            self.cfg = config
        elif _CYCLICAL_CONF is not None:
            self.cfg = CyclicalEncoderConfig(**_CYCLICAL_CONF)
        else:
            self.cfg = CyclicalEncoderConfig()

    @staticmethod
    def _ensure_datetime(series: pd.Series, tz: Optional[str], coerce_invalid: bool) -> pd.Series:
        s = pd.to_datetime(series, errors=("coerce" if coerce_invalid else "raise"))
        if tz is not None:
            # Naive Zeitstempel als UTC interpretieren, dann in Ziel-TZ konvertieren.
            if s.dt.tz is None:
                s = s.dt.tz_localize("UTC").dt.tz_convert(tz)
            else:
                s = s.dt.tz_convert(tz)
        return s

    @staticmethod
    def _extract(values: pd.Series, kind: str) -> pd.Series:
        """
        Liefert einen floatigen Index (0-basig) mit NaN an Positionen, an denen values NaT ist.
        Damit vermeiden wir IntCastingNaNError bei Pandas.
        """
        mask = values.notna()
        out = pd.Series(np.nan, index=values.index, dtype="float64")

        if kind == "dow":
            out.loc[mask] = values.loc[mask].dt.dayofweek.astype("int64")  # 0..6
            return out
        if kind == "month":
            out.loc[mask] = (values.loc[mask].dt.month.astype("int64") - 1)  # 0..11
            return out
        if kind == "doy":
            out.loc[mask] = (values.loc[mask].dt.dayofyear.astype("int64") - 1)  # 0..365
            return out
        if kind == "week":
            # ISO-Woche: 1..53 -> 0..52
            iso_week = values.loc[mask].dt.isocalendar().week.astype("int64") - 1
            out.loc[mask] = iso_week.to_numpy()
            return out
        if kind == "hour":
            out.loc[mask] = values.loc[mask].dt.hour.astype("int64")  # 0..23
            return out
        if kind == "minute":
            out.loc[mask] = values.loc[mask].dt.minute.astype("int64")  # 0..59
            return out
        if kind == "quarter":
            out.loc[mask] = (values.loc[mask].dt.quarter.astype("int64") - 1)  # 0..3
            return out

        raise ValueError(f"Unbekannter extractor: {kind}")

    @staticmethod
    def _to_sin_cos(x: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
        angle = 2.0 * np.pi * (x / float(period))
        return np.sin(angle), np.cos(angle)

    def fit(self, df: pd.DataFrame) -> "CyclicalEncoder":
        """
        No-op zur Pipeline-Kompatibilität; führt eine frühe Eingabeprüfung durch.
        """
        cfg = self.cfg
        if cfg.datetime_col not in df.columns:
            raise KeyError(f"datetime_col '{cfg.datetime_col}' fehlt im DataFrame.")
        _ = self._ensure_datetime(df[cfg.datetime_col], cfg.tz, cfg.coerce_invalid)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        if cfg.datetime_col not in df.columns:
            raise KeyError(f"datetime_col '{cfg.datetime_col}' fehlt im DataFrame.")
        if any(p <= 1 for _, (_, p) in cfg.periodicities.items()):
            raise ValueError("Alle Perioden müssen > 1 sein.")

        out = df.copy()
        dt = self._ensure_datetime(out[cfg.datetime_col], cfg.tz, cfg.coerce_invalid)

        source_cols: List[str] = []
        for name, (kind, period) in cfg.periodicities.items():
            idx = self._extract(dt, kind=kind)  # float mit NaN erlaubt
            src_col = f"{cfg.prefix}_{name}_idx"
            out[src_col] = idx

            sin_col = f"{cfg.prefix}_{name}_sin"
            cos_col = f"{cfg.prefix}_{name}_cos"
            sin_vals, cos_vals = self._to_sin_cos(idx, period=period)
            out[sin_col], out[cos_col] = sin_vals, cos_vals

            source_cols.append(src_col)

        if cfg.drop_source_cols and source_cols:
            out.drop(columns=source_cols, inplace=True)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# --- Pipeline-IO: erzeugt train_features_cyc.parquet -------------------------------
def main() -> None:
    """Liest train_features.parquet, kodiert zyklisch und speichert train_features_cyc.parquet."""
    from src.config import PROCESSED_DIR
    import pandas as pd
    from pathlib import Path

    base_dir = Path(__file__).resolve().parents[2]
    in_path = base_dir / "data" / "processed" / "train_features.parquet"
    out_path = base_dir / "data" / "processed" / "train_features_cyc.parquet"

    print(f"[cyclical_encoder] Lade {in_path} ...")
    df = pd.read_parquet(in_path)

    enc = CyclicalEncoder()
    out = enc.fit_transform(df)

    out.to_parquet(out_path, index=False)
    print(f"[cyclical_encoder] geschrieben: {out_path}")


if __name__ == "__main__":
    main()




# python -m src.data.cyclical_encoder