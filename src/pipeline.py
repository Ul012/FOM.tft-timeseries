# src/pipeline.py
"""
Minimale Pipeline-Orchestrierung für reproduzierbare Experimente.

Aufruf-Beispiele:
    # Kompletter Run (Preprocessing + Training)
    python -m src.pipeline --dataset configs/datasets/booksales.yaml --model configs/models/tft/baseline.yaml

    # Nur Preprocessing
    python -m src.pipeline --dataset configs/datasets/booksales.yaml --steps preprocessing,model_dataset,dataset_tft

    # Nur Training (Preprocessing bereits erledigt)
    python -m src.pipeline --dataset configs/datasets/booksales.yaml --model configs/models/tft/baseline.yaml --steps training
    python -m src.pipeline --dataset configs/datasets/booksales.yaml --model configs/models/tft/lr_high.yaml --steps training

Philosophie:
- Nutzt bestehende Module via subprocess (kein Refactoring nötig)
- Jeder Schritt bleibt einzeln testbar
- Erzeugt Manifest für volle Reproduzierbarkeit
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import sys


def run_subprocess(cmd: List[str], step_name: str) -> None:
    """
    Führt einen Subprocess aus und gibt Output direkt aus.

    Args:
        cmd: Command als Liste (z.B. ["python", "-m", "src.data.feature_engineering"])
        step_name: Name des Schritts für Logging

    Raises:
        RuntimeError: Wenn Subprocess fehlschlägt
    """
    print(f"\n{'=' * 70}")
    print(f"[Pipeline] Führe aus: {step_name}")
    print(f"[Pipeline] Command: {' '.join(cmd)}")
    print(f"{'=' * 70}\n")

    # UTF-8 für Windows-Subprozesse erzwingen
    import os
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        cmd,
        env=env,
        encoding='utf-8',
        # Kein capture_output! → Live-Output direkt an Terminal
    )

    if result.returncode != 0:
        print(f"\n[ERROR] Schritt '{step_name}' fehlgeschlagen:", file=sys.stderr)
        raise RuntimeError(f"Pipeline-Schritt fehlgeschlagen: {step_name}")


def run_preprocessing_steps(dataset_cfg: Dict[str, Any]) -> List[str]:
    """
    Führt alle aktivierten Preprocessing-Schritte aus der Dataset-Config aus.

    Args:
        dataset_cfg: Geladene Dataset-YAML

    Returns:
        Liste der ausgeführten Schritte
    """
    module_map = {
        "alignment": "src.data.data_alignment",
        "cleaning": "src.data.data_cleaning",
        "feature_engineering": "src.data.feature_engineering",
        "cyclical_encoder": "src.data.cyclical_encoder",
        "lag_features": "src.data.lag_features",
    }

    executed = []

    for step_cfg in dataset_cfg.get("preprocessing", []):
        step_name = step_cfg["step"]
        enabled = step_cfg.get("enabled", True)

        if not enabled:
            print(f"[Pipeline] Überspringe (disabled): {step_name}")
            continue

        module = module_map.get(step_name)
        if not module:
            raise ValueError(f"Unbekannter Preprocessing-Step: {step_name}")

        cmd = [sys.executable, "-m", module]
        run_subprocess(cmd, step_name)
        executed.append(step_name)

    return executed


def run_model_dataset() -> None:
    """Führt model_dataset.py aus (Split in train/val/test)."""
    cmd = [sys.executable, "-m", "src.modeling.model_dataset"]
    run_subprocess(cmd, "model_dataset")


def run_dataset_tft() -> None:
    """Führt dataset_tft.py aus (TFT-spezifische Dataset-Spec)."""
    cmd = [sys.executable, "-m", "src.modeling.dataset_tft"]
    run_subprocess(cmd, "dataset_tft")


def run_training(model_cfg_path: Path) -> str:
    """
    Führt Training aus (trainer_tft.py mit Config).

    Args:
        model_cfg_path: Pfad zur Model-YAML

    Returns:
        Run-ID des Trainings (extrahiert aus logs/)
    """
    cmd = [sys.executable, "-m", "src.modeling.trainer_tft", "--config", str(model_cfg_path)]
    run_subprocess(cmd, "training")

    # Extrahiere Run-ID aus logs/ (letztes run_* Verzeichnis)
    logs_dir = Path("logs/tft")
    if logs_dir.exists():
        runs = sorted(logs_dir.glob("run_*"))
        if runs:
            return runs[-1].name

    return "unknown"


def create_manifest(
        run_id: str,
        dataset_cfg: Dict[str, Any],
        model_cfg: Dict[str, Any] | None,
        steps_requested: List[str],
        steps_executed: Dict[str, Any],
) -> Path:
    """
    Erstellt ein JSON-Manifest für volle Reproduzierbarkeit.

    Args:
        run_id: Pipeline-Run-ID
        dataset_cfg: Geladene Dataset-Config
        model_cfg: Geladene Model-Config (optional)
        steps_requested: Vom User angeforderte Schritte
        steps_executed: Dictionary mit ausgeführten Schritten pro Phase

    Returns:
        Pfad zum erstellten Manifest
    """
    manifest = {
        "pipeline_run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "name": dataset_cfg.get("name"),
            "config": dataset_cfg,
        },
        "model": {
            "type": model_cfg.get("type") if model_cfg else None,
            "name": model_cfg.get("name") if model_cfg else None,
            "config": model_cfg,
        } if model_cfg else None,
        "execution": {
            "steps_requested": steps_requested,
            "steps_executed": steps_executed,
        },
    }

    manifest_dir = Path("results/pipeline_runs")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / f"{run_id}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    return manifest_path


def run_pipeline(
        dataset_cfg_path: Path,
        model_cfg_path: Path | None,
        steps: List[str],
) -> None:
    """
    Hauptfunktion: Orchestriert die komplette Pipeline.

    Args:
        dataset_cfg_path: Pfad zur Dataset-YAML
        model_cfg_path: Pfad zur Model-YAML (optional)
        steps: Liste der auszuführenden Schritte
    """
    # Run-ID generieren
    run_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'#' * 70}")
    print(f"# Pipeline-Run: {run_id}")
    print(f"{'#' * 70}\n")

    # Configs laden
    if not dataset_cfg_path.exists():
        raise FileNotFoundError(f"Dataset-Config nicht gefunden: {dataset_cfg_path}")

    dataset_cfg = yaml.safe_load(dataset_cfg_path.read_text(encoding="utf-8"))
    print(f"[Pipeline] Dataset: {dataset_cfg.get('name')}")

    model_cfg = None
    if model_cfg_path:
        if not model_cfg_path.exists():
            raise FileNotFoundError(f"Model-Config nicht gefunden: {model_cfg_path}")
        model_cfg = yaml.safe_load(model_cfg_path.read_text(encoding="utf-8"))
        print(f"[Pipeline] Model: {model_cfg.get('type')} / {model_cfg.get('name')}")

    print(f"[Pipeline] Schritte: {', '.join(steps)}\n")

    # Tracking für Manifest
    steps_executed: Dict[str, Any] = {}

    # Phase 1: Preprocessing
    if "preprocessing" in steps:
        executed = run_preprocessing_steps(dataset_cfg)
        steps_executed["preprocessing"] = executed

    # Phase 2: Model Dataset
    if "model_dataset" in steps:
        run_model_dataset()
        steps_executed["model_dataset"] = True

    # Phase 3: Dataset TFT
    if "dataset_tft" in steps:
        run_dataset_tft()
        steps_executed["dataset_tft"] = True

    # Phase 4: Training
    training_run_id = None
    if "training" in steps:
        if not model_cfg_path:
            raise ValueError("Training requested but no --model config provided")
        training_run_id = run_training(model_cfg_path)
        steps_executed["training"] = {"run_id": training_run_id}

    # Manifest erstellen
    manifest_path = create_manifest(
        run_id=run_id,
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        steps_requested=steps,
        steps_executed=steps_executed,
    )

    # Erfolgs-Summary
    print(f"\n{'#' * 70}")
    print(f"# Pipeline abgeschlossen: {run_id}")
    print(f"{'#' * 70}\n")
    print(f"[Pipeline] ✓ Alle Schritte erfolgreich ausgeführt")
    print(f"[Pipeline] Manifest: {manifest_path}")

    if training_run_id:
        print(f"[Pipeline] Training-Run-ID: {training_run_id}")
        print(f"[Pipeline] Logs: logs/tft/{training_run_id}/")
        print(f"[Pipeline] Checkpoints: results/tft/runs/{training_run_id}/checkpoints/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline-Orchestrierung für TFT-TimeSeries-Projekt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:

  # Kompletter Run (Preprocessing + Training)
  python -m src.pipeline \\
      --dataset configs/datasets/booksales.yaml \\
      --model configs/models/tft/baseline.yaml

  # Nur Preprocessing
  python -m src.pipeline \\
      --dataset configs/datasets/booksales.yaml \\
      --steps preprocessing,model_dataset,dataset_tft

  # Nur Training (wenn Preprocessing bereits erledigt)
  python -m src.pipeline \\
      --dataset configs/datasets/booksales.yaml \\
      --model configs/models/tft/baseline.yaml \\
      --steps training
        """
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Pfad zur Dataset-Config (z.B. configs/datasets/booksales.yaml)",
    )

    parser.add_argument(
        "--model",
        type=Path,
        help="Pfad zur Model-Config (z.B. configs/models/tft/baseline.yaml)",
    )

    parser.add_argument(
        "--steps",
        type=str,
        default="preprocessing,model_dataset,dataset_tft,training",
        help="Komma-separierte Liste der auszuführenden Schritte (default: alle)",
    )

    args = parser.parse_args()

    # Steps parsen
    steps = [s.strip() for s in args.steps.split(",")]

    # Validierung
    valid_steps = {"preprocessing", "model_dataset", "dataset_tft", "training"}
    invalid = set(steps) - valid_steps
    if invalid:
        parser.error(f"Ungültige Steps: {invalid}. Erlaubt: {valid_steps}")

    # Pipeline ausführen
    try:
        run_pipeline(
            dataset_cfg_path=args.dataset,
            model_cfg_path=args.model,
            steps=steps,
        )
    except Exception as e:
        print(f"\n[FEHLER] Pipeline abgebrochen: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()