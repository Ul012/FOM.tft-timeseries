# src/visualization/plot_tft_eval_single.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.config import BASE_DIR  # type: ignore


def _load_eval_summary(run_id: str) -> Any:
    path = BASE_DIR / "results" / "tft" / "eval" / run_id / "eval_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _to_metrics(eval_data: Any) -> Dict[str, Dict[str, float]]:
    if isinstance(eval_data, dict) and "metrics" in eval_data:
        return {
            "val": dict(eval_data.get("metrics", {}).get("val", {})),
            "test": dict(eval_data.get("metrics", {}).get("test", {})),
        }

    if isinstance(eval_data, list):
        metrics: Dict[str, Dict[str, float]] = {}
        for row in eval_data:
            if not isinstance(row, dict):
                continue
            split = row.get("split") or row.get("set") or row.get("dataset")
            if not split:
                continue
            metrics[split] = {
                k: float(v)
                for k, v in row.items()
                if k not in {"split", "set", "dataset"}
            }
        return metrics

    raise TypeError(f"Unerwartetes Format in eval_overview.json: {type(eval_data)}")


def _plot_single_run(run_id: str, eval_data: Any, output_path: Path) -> None:
    metrics = _to_metrics(eval_data)
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
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot gespeichert: {output_path}")

    plt.show()
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id: str = args.run_id

    data = _load_eval_summary(run_id)

    out_dir = BASE_DIR / "results" / "tft" / "plots" / "eval"
    output_path = out_dir / f"{run_id}_metrics.png"

    _plot_single_run(run_id, data, output_path)


if __name__ == "__main__":
    main()

# Aufruf (nach Evaluation):
#   python -m src.visualization.plot_tft_eval_single --run-id run_20251121_125758_baseline
#   python -m src.visualization.plot_tft_eval_single --run-id run_20251121_150832_bs_small
#   python -m src.visualization.plot_tft_eval_single --run-id run_20251121_174613_lr_high
#
# Output: results/tft/plots/eval/<run_id>_metrics.png


