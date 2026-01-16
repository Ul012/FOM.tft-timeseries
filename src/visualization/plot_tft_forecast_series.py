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

from src.config import BASE_DIR, PROCESSED_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema, get_tft_config

# Lade Dataset-Config
_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]
_schema = get_schema(_dataset_config)

TIME_COL = _schema["time_col"]
ID_COLS = _schema["id_cols"]
TARGET_COL = _schema["target_col"]
TFT_DATASET = get_tft_config(_dataset_config)


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
        required=False, # geändert von True auf False durch Hinzunahme checkpoints
        help="Run-ID wie in results/tft/runs/<run_id>/",
    )
    parser.add_argument( # für Erkennung von Checkpoints hinzugefügt
        "--checkpoint",
        required=False,
        help="Direkter Pfad zum Checkpoint (für Optuna-Trials)",
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
        path = BASE_DIR / "data" / "processed" / _dataset_name / "val.parquet"
    elif split == "test":
        path = BASE_DIR / "data" / "processed" / _dataset_name / "test.parquet"
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

        # Füge ersten Forecast-Punkt hinzu für nahtlose Verbindung
        last_hist_idx = hist_mask[hist_mask].index[-1]
        first_fc_idx = is_forecast[is_forecast].index[0]
        extended_mask = hist_mask.copy()
        extended_mask.loc[first_fc_idx] = True

        ax.plot(
            time[extended_mask], # time[hist_mask],
            y_true[extended_mask], # y_true[hist_mask],
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
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot gespeichert: {output_path}")

    plt.show()
    plt.close(fig)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    if not args.run_id and not args.checkpoint:
        raise ValueError("Entweder --run-id oder --checkpoint muss angegeben werden.")

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt_path}")
        run_id = ckpt_path.parent.parent.name  # z.B. "trial_0020"
    else:
        run_id = args.run_id
        ckpt_path = _find_checkpoint(run_id)

    split = args.split
    history_length = args.history_length

    # Checkpoint laden

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

    # === NEU: Zahlenausgabe für Forecast-Bereich ===
    df_forecast = df_series_fc[df_series_fc["is_forecast"]].copy()
    df_forecast["error"] = df_forecast["y_pred"] - df_forecast["y_true"]
    df_forecast["abs_error"] = df_forecast["error"].abs()
    df_forecast["pct_error"] = (df_forecast["error"] / df_forecast["y_true"]) * 100

    print("\n" + "=" * 70)
    print("FORECAST-DETAILS (pro Tag)")
    print("=" * 70)
    print(f"{'Datum':<12} {'Ist':>10} {'Prognose':>10} {'Fehler':>10} {'Fehler %':>10}")
    print("-" * 70)

    for _, row in df_forecast.iterrows():
        date_str = str(row[TIME_COL])[:10] if hasattr(row[TIME_COL], 'strftime') else str(row[TIME_COL])[:10]
        print(f"{date_str:<12} {row['y_true']:>10.1f} {row['y_pred']:>10.1f} {row['error']:>+10.1f} {row['pct_error']:>+9.1f}%")

    # Zusammenfassung
    mae = df_forecast["abs_error"].mean()
    rmse = np.sqrt((df_forecast["error"] ** 2).mean())
    mape = df_forecast["pct_error"].abs().mean()

    print("-" * 70)
    print(f"{'MAE:':<12} {mae:>10.2f}")
    print(f"{'RMSE:':<12} {rmse:>10.2f}")
    print(f"{'MAPE:':<12} {mape:>9.2f}%")
    print("=" * 70)

    # CSV speichern
    csv_path = plots_root / f"{run_id}_{split}_forecast_details.csv"
    df_forecast[[TIME_COL, "y_true", "y_pred", "error", "abs_error", "pct_error"]].to_csv(csv_path, index=False)
    print(f"\n✓ Forecast-Details als CSV: {csv_path}")

    print("\n[plot_tft_forecast_series] Plot erstellt.")
    print(f"- Run-ID : {run_id}")
    print(f"- Split  : {split}")
    print(f"- Datei  : {output_path}")


if __name__ == "__main__":
    main()

# Aufruf:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.visualization.plot_tft_forecast_series --run-id run_20260111_113916_booksales_optuna_tft_newyear_best --split test --history-length 120
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.visualization.plot_tft_forecast_series --run-id run_20260111_113926_walmart_optuna_walmart_full_best --split test --history-length 120
