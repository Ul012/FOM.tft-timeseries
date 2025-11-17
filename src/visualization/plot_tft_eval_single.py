# src/visualization/plot_tft_eval_single.py
# Visualisiert Val- und Testmetriken eines einzelnen TFT-Evaluationslaufs.
# Nutzung:
#   python -m src.visualization.plot_tft_eval_single --run-id run_YYYYMMDD_HHMMSS_suffix

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.config import (  # type: ignore
    BASE_DIR,
)


def _load_eval_summary(run_id: str) -> Dict[str, Any]:
    eval_path = BASE_DIR / "results" / "tft" / "eval" / run_id / "eval_summary.json"
    if not eval_path.is_file():
        raise FileNotFoundError(f"eval_summary.json nicht gefunden: {eval_path}")
    return json.loads(eval_path.read_text(encoding="utf-8"))


def _plot_single_run(
    run_id: str,
    eval_data: Dict[str, Any],
    output_path: Path,
) -> None:
    metrics = eval_data.get("metrics", {})
    val = metrics.get("val", {})
    test = metrics.get("test", {})

    metric_names: List[str] = ["mae", "rmse", "mape", "smape"]

    val_values = [val.get(m, np.nan) for m in metric_names]
    test_values = [test.get(m, np.nan) for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width / 2, val_values, width, label="Val")
    ax.bar(x + width / 2, test_values, width, label="Test")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylabel("Fehlermaß")
    ax.set_title(f"TFT-Evaluation – {run_id}")
    ax.legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualisiert Val- und Testmetriken eines TFT-Evaluationslaufs."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run-ID wie in results/tft/eval/<run_id>/eval_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id: str = args.run_id

    data = _load_eval_summary(run_id)

    plots_root = BASE_DIR / "results" / "tft" / "plots" / "eval"
    output_path = plots_root / f"{run_id}_metrics.png"

    _plot_single_run(run_id, data, output_path)

    print("[plot_tft_eval_single] Plot erstellt.")
    print(f"- Run-ID : {run_id}")
    print(f"- Datei  : {output_path}")


if __name__ == "__main__":
    # python -m src.visualization.plot_tft_eval_single --run-id run_20251116_183848_baseline02
    # python -m src.visualization.plot_tft_eval_single --run-id run_20251115_160147_bs32
    # python -m src.visualization.plot_tft_eval_single --run-id run_20251116_230357_lr001
    # python -m src.visualization.plot_tft_eval_single --run-id run_20251117_091520_lr001_hs64_hcs32
    main()
