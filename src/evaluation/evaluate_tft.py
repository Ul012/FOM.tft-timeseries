# src/evaluation/evaluate_tft.py
# Evaluation eines trainierten TFT-Modells für einen gegebenen Run.
# Nutzung (Beispiel):
#   python -m src.evaluation.evaluate_tft --run-id run_YYYYMMDD_HHMMSS_baseline

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from pytorch_forecasting.models import TemporalFusionTransformer

from src.config import (  # type: ignore
    BASE_DIR,
    PROCESSED_DIR,
    TIME_COL,
    ID_COLS,
    TARGET_COL,
    TFT_DATASET
)


# ---------------------------------------------------------------------------
# Metriken
# ---------------------------------------------------------------------------


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.abs(y_true - y_pred)
    return float(np.mean(diff))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sq = (y_true - y_pred) ** 2
    return float(np.sqrt(np.mean(sq)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    num = np.abs(y_pred - y_true)
    denom = np.clip(np.abs(y_true) + np.abs(y_pred), eps, None)
    return float(np.mean(2.0 * num / denom) * 100.0)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": _mae(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
        "smape": _smape(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Datenstrukturen
# ---------------------------------------------------------------------------


@dataclass
class SplitMetrics:
    mae: float
    rmse: float
    mape: float
    smape: float


@dataclass
class EvaluationResult:
    run_id: str
    checkpoint_path: str
    metrics_val: SplitMetrics
    metrics_test: SplitMetrics
    eval_dir: Path


# ---------------------------------------------------------------------------
# Einfacher Logger (später durch MLflow ersetzbar)
# ---------------------------------------------------------------------------


class EvalLogger:
    """
    Minimaler Logger für Evaluationsläufe.
    Kümmert sich nur um das Schreiben einfacher Dateien (JSON/CSV).
    Kann später durch MLflow ersetzt werden, ohne die Evaluationslogik zu ändern.
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_json(self, data: Dict[str, Any], filename: str = "eval_summary.json") -> Path:
        path = self.output_dir / filename
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Kernlogik: Evaluierung eines TFT-Runs
# ---------------------------------------------------------------------------


def _find_best_checkpoint(model_cfg: Dict[str, Any]) -> Path:
    """
    Modellkonfiguration erwartet mindestens:
      - 'checkpoint_root': Basisordner, z. B. results/tft/runs
      - 'run_id'        : Run-ID
      - optional 'checkpoint_pattern': z. B. '*.ckpt'
    """
    checkpoint_root = Path(model_cfg["checkpoint_root"])
    run_id = model_cfg["run_id"]
    pattern = model_cfg.get("checkpoint_pattern", "*.ckpt")

    ckpt_dir = checkpoint_root / run_id / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint-Verzeichnis nicht gefunden: {ckpt_dir}")

    candidates = sorted(ckpt_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"Keine .ckpt-Dateien im Verzeichnis: {ckpt_dir}")

    for path in candidates:
        if "best" in path.name.lower():
            return path

    return candidates[0]


def _load_splits(data_cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Datenkonfiguration erwartet:
      - 'val_path': Pfad zur Validation-Datei
      - 'test_path': Pfad zur Test-Datei
    """
    val_path = Path(data_cfg["val_path"])
    test_path = Path(data_cfg["test_path"])

    if not val_path.is_file():
        raise FileNotFoundError(f"Validation-Datei nicht gefunden: {val_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test-Datei nicht gefunden: {test_path}")

    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)
    return df_val, df_test


def _evaluate_split(
    model: TemporalFusionTransformer,
    df: pd.DataFrame,
) -> SplitMetrics:
    """
    Führt Vorhersage für einen Split (Val oder Test) durch und berechnet Metriken.

    Annahmen:
    - df enthält die Spalte TARGET_COL.
    - Es existieren ID_COLS und TIME_COL, wie in src.config definiert.
    - Es wird nur der Vorhersagehorizont (max_prediction_length) jeder Zeitreihe bewertet.
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Zielspalte {TARGET_COL!r} fehlt im DataFrame.")

    # 1) Modellvorhersage für den gesamten Split
    preds = model.predict(df)
    if hasattr(preds, "numpy"):
        preds = preds.numpy()
    y_pred = np.asarray(preds, dtype=float).reshape(-1)

    # 2) Ground Truth nur für den Vorhersagehorizont je Zeitreihe
    prediction_length = int(TFT_DATASET["max_prediction_length"])

    # Falls keine ID-Spalten definiert sind, wird der gesamte DataFrame als eine Serie behandelt.
    if ID_COLS:
        group_cols = list(ID_COLS)
    else:
        # Fallback: eine künstliche Gruppe, nutzt den gesamten df
        group_cols = []

    y_true_list: list[float] = []

    if group_cols:
        # Pro Zeitreihe nach Zeit sortieren und die letzten prediction_length Werte nehmen
        grouped = df.groupby(group_cols, sort=False)
        for _, g in grouped:
            g_sorted = g.sort_values(TIME_COL)
            tail = g_sorted.tail(prediction_length)
            y_true_list.extend(tail[TARGET_COL].to_numpy(dtype=float))
    else:
        # Eine "globale" Serie: nach Zeit sortieren und die letzten prediction_length Werte nehmen
        df_sorted = df.sort_values(TIME_COL)
        tail = df_sorted.tail(prediction_length)
        y_true_list.extend(tail[TARGET_COL].to_numpy(dtype=float))

    y_true = np.asarray(y_true_list, dtype=float).reshape(-1)

    # 3) Plausibilitätscheck
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shapes passen nicht zusammen (nach Horizont-Auswahl): "
            f"y_true={y_true.shape}, y_pred={y_pred.shape}. "
            f"Prüfen, ob max_prediction_length und die Gruppierung zu den Modellvorhersagen passen."
        )

    metrics_raw = _compute_metrics(y_true, y_pred)

    return SplitMetrics(
        mae=metrics_raw["mae"],
        rmse=metrics_raw["rmse"],
        mape=metrics_raw["mape"],
        smape=metrics_raw["smape"],
    )


def evaluate_tft_run(
    data_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Zentrale Evaluationsfunktion für einen TFT-Run.

    Erwartete Keys:
      data_cfg:
        - 'val_path'
        - 'test_path'
      model_cfg:
        - 'checkpoint_root'
        - 'run_id'
        - optional 'checkpoint_pattern'
      eval_cfg:
        - 'eval_root' (Basisordner für Evaluation, z. B. results/tft/eval)
        - optional weitere Einstellungen (z. B. batch_size, seeds, Tags)

    Rückgabe:
      Dictionary mit Metriken und Pfaden zu erzeugten Artefakten.
    """
    run_id = model_cfg["run_id"]

    # 1) Checkpoint + Modell laden
    ckpt_path = _find_best_checkpoint(model_cfg)
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))

    # 2) Daten laden
    df_val, df_test = _load_splits(data_cfg)

    # 3) Metriken berechnen
    metrics_val = _evaluate_split(model, df_val)
    metrics_test = _evaluate_split(model, df_test)

    # 4) Eval-Ordner + Logger
    eval_root = Path(eval_cfg["eval_root"])
    eval_dir = eval_root / run_id
    logger = EvalLogger(eval_dir)

    # 5) Ergebnis-Payload bauen
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "checkpoint_path": str(ckpt_path),
        "metrics": {
            "val": asdict(metrics_val),
            "test": asdict(metrics_test),
        },
        "meta": {
            "time_col": TIME_COL,
            "id_cols": list(ID_COLS),
            "target_col": TARGET_COL,
        },
    }

    summary_path = logger.log_json(payload, filename="eval_summary.json")

    # 6) Rückgabe mit Artefaktpfaden
    result: Dict[str, Any] = {
        "run_id": run_id,
        "metrics": payload["metrics"],
        "checkpoint_path": str(ckpt_path),
        "artifacts": {
            "eval_summary_path": str(summary_path),
            "eval_dir": str(eval_dir),
        },
    }
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluierung eines TFT-Runs auf Basis von Validation- und Testdaten."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run-ID wie in logs/tft/<run_id>/ und results/tft/runs/<run_id>/",
    )
    # Optional: später --eval-config hinzufügen, um YAML einzulesen.
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = args.run_id

    # Konfiguration zentral in Dicts sammeln.
    # Diese Blöcke können später 1:1 aus einer YAML geladen werden.
    data_cfg: Dict[str, Any] = {
        "val_path": str(PROCESSED_DIR / "val.parquet"),
        "test_path": str(PROCESSED_DIR / "test.parquet"),
    }

    model_cfg: Dict[str, Any] = {
        "checkpoint_root": str(BASE_DIR / "results" / "tft" / "runs"),
        "run_id": run_id,
        "checkpoint_pattern": "*.ckpt",
    }

    eval_cfg: Dict[str, Any] = {
        "eval_root": str(BASE_DIR / "results" / "tft" / "eval"),
        # Platzhalter für zukünftige Erweiterungen:
        # "batch_size": 128,
        # "num_workers": 0,
    }

    result = evaluate_tft_run(data_cfg=data_cfg, model_cfg=model_cfg, eval_cfg=eval_cfg)

    print("[evaluate_tft] Evaluierung abgeschlossen.")
    print(f"- Run-ID           : {result['run_id']}")
    print(f"- Checkpoint       : {result['checkpoint_path']}")
    print(
        f"- Val-Metriken     : "
        f"MAE={result['metrics']['val']['mae']:.4f}, "
        f"RMSE={result['metrics']['val']['rmse']:.4f}, "
        f"MAPE={result['metrics']['val']['mape']:.2f}%, "
        f"SMAPE={result['metrics']['val']['smape']:.2f}%"
    )
    print(
        f"- Test-Metriken    : "
        f"MAE={result['metrics']['test']['mae']:.4f}, "
        f"RMSE={result['metrics']['test']['rmse']:.4f}, "
        f"MAPE={result['metrics']['test']['mape']:.2f}%, "
        f"SMAPE={result['metrics']['test']['smape']:.2f}%"
    )
    print(f"- Eval-Summary     : {result['artifacts']['eval_summary_path']}")


if __name__ == "__main__":
    # python -m src.evaluation.evaluate_tft --run-id run_20251116_183848_baseline02
    # python -m src.evaluation.evaluate_tft --run-id run_20251115_160147_bs32
    # python -m src.evaluation.evaluate_tft --run-id run_20251116_230357_lr001
    # python -m src.evaluation.evaluate_tft --run-id run_20251117_091520_lr001_hs64_hcs32
    main()
