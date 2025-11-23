# src/visualization/plot_tft_eval_comparison.py
# Visualisiert eine ausgewählte Metrik über mehrere TFT-Runs hinweg.
#
# WICHTIG - Workflow vor dem Plotten:
#   1. Runs evaluieren:
#      python -m src.evaluation.evaluate_tft --run-id run_20251121_123456_baseline
#   2. Overview aggregieren (erstellt eval_overview.csv):
#      python -m src.evaluation.aggregate_tft_eval
#   3. Dann plotten:
#      python -m src.visualization.plot_tft_eval_comparison --metric smape --split test
#
# Nutzung:
#   python -m src.visualization.plot_tft_eval_comparison --metric smape --split test

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (  # type: ignore
    BASE_DIR,
)


_VALID_METRICS = {"mae", "rmse", "mape", "smape"}
_VALID_SPLITS = {"val", "test"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vergleich von TFT-Evaluationsmetriken über mehrere Runs."
    )
    parser.add_argument(
        "--metric",
        required=True,
        choices=sorted(_VALID_METRICS),
        help="Metrik, die verglichen werden soll (mae, rmse, mape, smape).",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=sorted(_VALID_SPLITS),
        help="Split, der verwendet werden soll (val oder test).",
    )
    return parser.parse_args()


def _load_overview() -> pd.DataFrame:
    eval_root = BASE_DIR / "results" / "tft" / "eval"
    csv_path = eval_root / "eval_overview.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"eval_overview.csv nicht gefunden: {csv_path}. "
            f"Bitte zuerst 'aggregate_tft_eval.py' ausführen."
        )
    df = pd.read_csv(csv_path)
    return df


def _plot_comparison(
    df: pd.DataFrame,
    metric: str,
    split: str,
    output_path: Path,
) -> None:
    col_name = f"{split}_{metric}"
    if col_name not in df.columns:
        raise KeyError(f"Spalte {col_name!r} nicht in eval_overview.csv gefunden.")

    # Nur Runs mit gültigen Werten verwenden
    df_plot = df.dropna(subset=[col_name]).copy()
    if df_plot.empty:
        raise ValueError(f"Keine gültigen Werte für {col_name!r} vorhanden.")

    run_ids: List[str] = df_plot["run_id"].astype(str).tolist()
    values = df_plot[col_name].tolist()

    x = range(len(run_ids))

    fig, ax = plt.subplots()
    ax.bar(x, values)

    ax.set_xticks(list(x))
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.set_ylabel(col_name)
    ax.set_title(f"TFT-Vergleich – {col_name}")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot gespeichert: {output_path}")

    plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    metric: str = args.metric
    split: str = args.split

    df = _load_overview()

    plots_root = BASE_DIR / "results" / "tft" / "plots" / "eval"
    output_path = plots_root / f"compare_{split}_{metric}.png"

    _plot_comparison(df, metric, split, output_path)

    print("[plot_tft_eval_comparison] Plot erstellt.")
    print(f"- Split   : {split}")
    print(f"- Metrik  : {metric}")
    print(f"- Datei   : {output_path}")


if __name__ == "__main__":
    main()

# Aufruf (nach mehreren Evaluationen):
#   python -m src.visualization.plot_tft_eval_comparison --metric mae --split test