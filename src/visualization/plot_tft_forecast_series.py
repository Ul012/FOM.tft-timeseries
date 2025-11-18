# src/visualization/plot_tft_forecast_series.py
# Plot für eine einzelne Zeitreihe:
# - durchgängige Linie für Ist-Werte
# - ab Forecast-Start andere Farbe
# - Vorhersage-Linie im Vorhersagehorizont
#
# Nutzung:
#   python -m src.visualization.plot_tft_forecast_series \
#       --run-id run_YYYYMMDD_HHMMSS_suffix \
#       --split test \
#       --history-length 120

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pytorch_forecasting.models import TemporalFusionTransformer

from src.config import (  # type: ignore
    BASE_DIR,
    PROCESSED_DIR,
    TIME_COL,
    ID_COLS,
    TARGET_COL,
    TFT_DATASET,
)


# -----------------------------------------------------------------------------
# Argumente
# -----------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plottet Ist vs. Prognose für eine einzelne Zeitreihe, "
            "mit Farbwechsel ab Forecast-Start."
        )
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run-ID wie in results/tft/run_20251117_232558_lr001_mel120/<run_id>/",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["val", "test"],
        help="Zu plottender Split (val oder test).",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=None,
        help=(
            "Anzahl Punkte vor Forecast-Start, die angezeigt werden sollen. "
            "Wenn nicht gesetzt, wird die gesamte Historie der Serie geplottet."
        ),
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------
def _find_checkpoint(run_id: str) -> Path:
    """
    Findet den zu verwendenden Checkpoint für einen Run.

    Präferenz:
    - Dateien mit 'best' im Namen, sonst die erste .ckpt-Datei.
    """
    ckpt_dir = BASE_DIR / "results" / "tft" / "runs" / run_id / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint-Verzeichnis nicht gefunden: {ckpt_dir}")

    ckpts: List[Path] = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"Keine .ckpt-Dateien in {ckpt_dir} gefunden.")

    best_ckpts = [p for p in ckpts if "best" in p.name.lower()]
    if best_ckpts:
        return best_ckpts[0]
    return ckpts[0]


def _load_split(split: str) -> pd.DataFrame:
    """Lädt den gewünschten Split (val oder test) aus data/processed."""
    if split == "val":
        path = PROCESSED_DIR / "val.parquet"
    elif split == "test":
        path = PROCESSED_DIR / "test.parquet"
    else:
        raise ValueError(f"Unbekannter Split: {split!r}")

    if not path.is_file():
        raise FileNotFoundError(f"Split-Datei nicht gefunden: {path}")
    return pd.read_parquet(path)


def _select_first_series(df_split: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Wählt die erste Zeitreihe aus (erste Kombination aus ID_COLS).
    Gibt den gefilterten DataFrame und die ID-Werte zurück.
    """
    if ID_COLS:
        first_ids = df_split[ID_COLS].drop_duplicates().iloc[0].to_dict()
        mask = np.ones(len(df_split), dtype=bool)
        for col, val in first_ids.items():
            mask &= df_split[col] == val
        df_series = df_split.loc[mask].copy()
    else:
        # Falls keine ID_COLS definiert sind, gesamte Tabelle als eine Serie
        df_series = df_split.copy()
        first_ids = {}

    df_series.sort_values(TIME_COL, inplace=True)
    df_series.reset_index(drop=True, inplace=True)
    return df_series, first_ids


def _apply_history_window(
    df_series: pd.DataFrame,
    prediction_length: int,
    history_length: int | None,
) -> pd.DataFrame:
    """
    Schneidet optional die Historie zu:
    - behält immer mindestens den Forecast-Horizont,
    - ergänzt davor history_length Punkte, falls gesetzt.
    """
    n = len(df_series)
    if history_length is None:
        return df_series

    total_needed = history_length + prediction_length
    if n <= total_needed:
        return df_series

    return df_series.iloc[n - total_needed :].reset_index(drop=True)


def _build_series_forecast_frame(
    model: TemporalFusionTransformer,
    df_series: pd.DataFrame,
) -> pd.DataFrame:
    """
    Erzeugt ein DataFrame für eine einzelne Serie mit:
    - TIME_COL
    - y_true (Ist-Wert)
    - y_pred (nur im Forecast-Horizont befüllt)
    - is_forecast (bool: True für Forecast-Bereich)
    """
    if TARGET_COL not in df_series.columns:
        raise KeyError(f"Zielspalte {TARGET_COL!r} fehlt im DataFrame.")

    prediction_length = int(TFT_DATASET["max_prediction_length"])

    # 1) Modellvorhersage für diese Serie
    preds = model.predict(df_series)
    if hasattr(preds, "numpy"):
        preds = preds.numpy()
    y_pred = np.asarray(preds, dtype=float).reshape(-1)

    if len(y_pred) != prediction_length:
        raise ValueError(
            f"Erwartet wurden {prediction_length} Vorhersagen, erhalten: {len(y_pred)}. "
            "Bitte prüfen, ob max_prediction_length mit dem Modell konsistent ist."
        )

    # 2) True-Werte
    df_series = df_series.copy()
    df_series["y_true"] = df_series[TARGET_COL].astype(float)

    # Forecast-Horizont = letzte prediction_length Zeilen
    df_series["is_forecast"] = False
    df_series.loc[df_series.index[-prediction_length:], "is_forecast"] = True

    # y_pred nur im Forecast-Bereich eintragen
    df_series["y_pred"] = np.nan
    df_series.loc[df_series.index[-prediction_length:], "y_pred"] = y_pred

    return df_series


def _plot_series_forecast(
    df_series: pd.DataFrame,
    run_id: str,
    split: str,
    series_ids: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Plottet:
    - Ist-Werte in der Historie (vor Forecast-Start) in Farbe 1
    - Ist-Werte im Forecast-Horizont in Farbe 2
    - Vorhersage im Forecast-Horizont in Farbe 3
    """
    time = df_series[TIME_COL]
    y_true = df_series["y_true"]
    y_pred = df_series["y_pred"]
    is_forecast = df_series["is_forecast"]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Historie (ohne Forecast-Bereich)
    hist_mask = ~is_forecast
    if hist_mask.any():
        ax.plot(
            time[hist_mask],
            y_true[hist_mask],
            label="Ist (Historie)",
        )

    # Ist im Forecast-Bereich
    fc_mask = is_forecast
    ax.plot(
        time[fc_mask],
        y_true[fc_mask],
        label="Ist (Forecast-Bereich)",
    )

    # Prognose im Forecast-Bereich
    ax.plot(
        time[fc_mask],
        y_pred[fc_mask],
        label="Prognose",
    )

    ax.set_xlabel("Zeit")
    ax.set_ylabel("Verkäufe")
    title_parts = [f"TFT – Vorhersage vs. Ist ({split})", f"Run: {run_id}"]
    if series_ids:
        id_str = ", ".join(f"{k}={v}" for k, v in series_ids.items())
        title_parts.append(f"Serie: {id_str}")
    ax.set_title(" | ".join(title_parts))

    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    run_id: str = args.run_id
    split: str = args.split
    history_length: int | None = args.history_length

    # Checkpoint laden
    ckpt_path = _find_checkpoint(run_id)
    print(f"[plot_tft_forecast_series] Verwende Checkpoint: {ckpt_path}")

    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)

    # Split laden und erste Serie auswählen
    df_split = _load_split(split)
    df_series, series_ids = _select_first_series(df_split)

    prediction_length = int(TFT_DATASET["max_prediction_length"])
    df_series = _apply_history_window(df_series, prediction_length, history_length)

    # Forecast-Frame bauen
    df_series_fc = _build_series_forecast_frame(model, df_series)

    # Plot
    plots_root = BASE_DIR / "results" / "tft" / "plots" / "eval"
    output_path = plots_root / f"{run_id}_{split}_forecast_series.png"

    _plot_series_forecast(df_series_fc, run_id, split, series_ids, output_path)

    print("[plot_tft_forecast_series] Plot erstellt.")
    print(f"- Run-ID : {run_id}")
    print(f"- Split  : {split}")
    print(f"- Datei  : {output_path}")


if __name__ == "__main__":
    # python -m src.visualization.plot_tft_forecast_series --run-id run_20251117_232558_lr001_mel120 --split test --history-length 120
    main()
