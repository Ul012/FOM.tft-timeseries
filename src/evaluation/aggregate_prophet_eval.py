# src/evaluation/aggregate_prophet_eval.py
# Aggregiert alle Prophet eval_val.json + eval_test.json zu einer Übersichtstabelle.
#
# Prophet-Struktur: eval_val.json und eval_test.json sind getrennt (nicht kombiniert wie TFT)
#
# Nutzung:
#   python -m src.evaluation.aggregate_prophet_eval

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import (
    BASE_DIR,
    EVALUATION_METRICS,
)


def _load_single_run(run_dir: Path) -> Dict[str, Any] | None:
    """
    Lädt eval_val.json + eval_test.json für einen Prophet-Run und kombiniert sie.

    Returns:
        Dict mit run_id und Metriken für val/test, oder None wenn Dateien fehlen
    """
    val_path = run_dir / "eval_val.json"
    test_path = run_dir / "eval_test.json"

    # Skip wenn Dateien fehlen
    if not val_path.exists() or not test_path.exists():
        return None

    try:
        val_data = json.loads(val_path.read_text(encoding="utf-8"))
        test_data = json.loads(test_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[aggregate_prophet_eval] Warnung: Fehler beim Laden von {run_dir.name}: {e}")
        return None

    run_id = val_data.get("run_id", run_dir.name)

    row: Dict[str, Any] = {
        "run_id": run_id,
        "dataset": val_data.get("dataset", ""),
        "n_groups": val_data.get("n_groups", 0),
    }

    # Metriken für beide Splits
    val_metrics = val_data.get("metrics", {}).get("overall", {})
    test_metrics = test_data.get("metrics", {}).get("overall", {})

    for metric in EVALUATION_METRICS:
        row[f"val_{metric}"] = val_metrics.get(metric)
        row[f"test_{metric}"] = test_metrics.get(metric)

    return row


def aggregate_evaluations(runs_root: Path) -> pd.DataFrame:
    """
    Sucht nach Prophet-Runs (Ordner mit eval_val.json + eval_test.json)
    und fasst alle Ergebnisse in einem DataFrame zusammen.
    """
    rows: List[Dict[str, Any]] = []

    # Iteriere durch alle run_* Ordner
    for run_dir in sorted(runs_root.glob("run_*")):
        if not run_dir.is_dir():
            continue

        row = _load_single_run(run_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"Keine Prophet-Evaluierungen gefunden in {runs_root}. "
            f"Bitte zuerst evaluate_prophet.py ausführen."
        )

    df = pd.DataFrame(rows)
    df.sort_values("run_id", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def main() -> None:
    runs_root = BASE_DIR / "results" / "prophet" / "runs"

    if not runs_root.exists():
        raise FileNotFoundError(f"Prophet runs Verzeichnis nicht gefunden: {runs_root}")

    df = aggregate_evaluations(runs_root)

    # Output ins prophet-Verzeichnis (analog zu TFT)
    output_dir = BASE_DIR / "results" / "prophet"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "eval_overview.csv"
    json_path = output_dir / "eval_overview.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print("[aggregate_prophet_eval] Aggregation abgeschlossen.")
    print(f"- Anzahl Runs : {len(df)}")
    print(f"- Datasets    : {df['dataset'].unique().tolist()}")
    print(f"- CSV         : {csv_path}")
    print(f"- JSON        : {json_path}")


if __name__ == "__main__":
    main()

# Aufruf (nachdem Prophet-Modelle evaluiert wurden):
#   python -m src.evaluation.aggregate_prophet_eval
#
# Erzeugt: results/prophet/eval_overview.{csv,json}