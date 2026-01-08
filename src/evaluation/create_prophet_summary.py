# src/evaluation/create_prophet_summary.py
# Kombiniert eval_val.json + eval_test.json zu eval_summary.json (pro Prophet-Run)

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from src.config import BASE_DIR


def create_summary(run_id: str) -> None:
    """Erstellt eval_summary.json aus eval_val.json + eval_test.json."""
    run_dir = BASE_DIR / "results" / "prophet" / "runs" / run_id

    val_path = run_dir / "eval_val.json"
    test_path = run_dir / "eval_test.json"

    if not val_path.exists():
        raise FileNotFoundError(f"eval_val.json nicht gefunden: {val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"eval_test.json nicht gefunden: {test_path}")

    val_data = json.loads(val_path.read_text(encoding="utf-8"))
    test_data = json.loads(test_path.read_text(encoding="utf-8"))

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "dataset": val_data["dataset"],
        "n_groups": val_data["n_groups"],
        "metrics": {
            "val": val_data["metrics"]["overall"],
            "test": test_data["metrics"]["overall"]
        }
    }

    summary_path = run_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[create_prophet_summary] Erstellt: {summary_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Erstellt eval_summary.json für Prophet-Run")
    parser.add_argument("--run-id", type=str, required=True, help="Run-ID (z.B. run_20260108_000719_prophet_baseline)")
    args = parser.parse_args()

    create_summary(args.run_id)


if __name__ == "__main__":
    main()

# Aufruf:
#   python -m src.evaluation.create_prophet_summary --run-id run_20260108_000719_prophet_baseline
#   python -m src.evaluation.create_prophet_summary --run-id run_20260108_000405_prophet_baseline
