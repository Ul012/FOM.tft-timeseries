# src/evaluation/aggregate_tft_eval.py
# Aggregiert alle eval_summary.json Dateien zu einer Übersichtstabelle.
# Nutzung:
#   python -m src.evaluation.aggregate_tft_eval

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import (
    BASE_DIR,
    EVALUATION_METRICS,
    EVALUATION_SPLITS,
)


def _load_single_eval(eval_path: Path) -> Dict[str, Any]:
    """Lädt eine einzelne eval_summary.json und gibt ein flaches Dict zurück."""
    data = json.loads(eval_path.read_text(encoding="utf-8"))

    run_id = data.get("run_id", eval_path.parent.name)
    metrics = data.get("metrics", {})
    val = metrics.get("val", {})
    test = metrics.get("test", {})

    row: Dict[str, Any] = {
        "run_id": run_id,
        "checkpoint_path": data.get("checkpoint_path", ""),
    }

    # Dynamisch alle Splits und Metriken
    for split in EVALUATION_SPLITS:
        split_metrics = metrics.get(split, {})
        for metric in EVALUATION_METRICS:
            row[f"{split}_{metric}"] = split_metrics.get(metric)
    return row


def aggregate_evaluations(eval_root: Path) -> pd.DataFrame:
    """
    Sucht rekursiv nach eval_summary.json unterhalb von eval_root und
    fasst alle Ergebnisse in einem DataFrame zusammen.
    """
    eval_files = sorted(eval_root.rglob("eval_summary.json"))
    rows: List[Dict[str, Any]] = []

    for path in eval_files:
        try:
            row = _load_single_eval(path)
            rows.append(row)
        except Exception as exc:  # bewusst simpel gehalten
            print(f"[aggregate_tft_eval] Warnung: Fehler beim Laden von {path}: {exc}")

    if not rows:
        raise FileNotFoundError(f"Keine eval_summary.json unterhalb von {eval_root} gefunden.")

    df = pd.DataFrame(rows)
    df.sort_values("run_id", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main() -> None:
    eval_root = BASE_DIR / "results" / "tft" / "eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    df = aggregate_evaluations(eval_root)

    csv_path = eval_root / "eval_overview.csv"
    json_path = eval_root / "eval_overview.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print("[aggregate_tft_eval] Aggregation abgeschlossen.")
    print(f"- Anzahl Runs : {len(df)}")
    print(f"- CSV         : {csv_path}")
    print(f"- JSON        : {json_path}")


if __name__ == "__main__":
    main()

# Aufruf (nachdem mindestens ein Run evaluiert wurde):
#   python -m src.evaluation.aggregate_tft_eval
#
# Erzeugt: results/tft/eval/eval_overview.{csv,json}