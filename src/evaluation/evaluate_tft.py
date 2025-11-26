# src/evaluation/evaluate_tft.py
# Evaluation eines trainierten TFT-Modells für einen gegebenen Run.

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.models import TemporalFusionTransformer

from src.config import BASE_DIR, PROCESSED_DIR, EVALUATION_METRICS
from src.utils.load_dataset_config import load_dataset_config, get_schema, get_tft_config

# Lade Config einmalig
_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]
_schema = get_schema(_dataset_config)

# Extrahiere Werte
TIME_COL = _schema["time_col"]
ID_COLS = _schema["id_cols"]
TARGET_COL = _schema["target_col"]
TFT_DATASET = get_tft_config(_dataset_config)


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


def _r2(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Berechnet R² (Coefficient of Determination).
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot < eps:
        return 0.0

    return float(1.0 - (ss_res / ss_tot))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metric_funcs = {
        "mae": _mae,
        "rmse": _rmse,
        "mape": _mape,
        "smape": _smape,
        "r2": _r2,
    }

    return {
        metric: metric_funcs[metric](y_true, y_pred)
        for metric in EVALUATION_METRICS
    }


@dataclass
class SplitMetrics:
    mae: float
    rmse: float
    mape: float
    smape: float
    r2: float


@dataclass
class EvaluationResult:
    run_id: str
    checkpoint_path: str
    metrics_val: SplitMetrics
    metrics_test: SplitMetrics
    eval_dir: Path


class EvalLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log_json(self, data: Dict[str, Any], filename: str = "eval_summary.json") -> Path:
        path = self.output_dir / filename
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def log_csv(self, row: Dict[str, Any], filename: str = "eval_summary.csv") -> Path:
        path = self.output_dir / filename
        df_new = pd.DataFrame([row])

        if path.exists():
            df_new.to_csv(path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(path, index=False)

        return path


def _find_best_checkpoint(model_cfg: Dict[str, Any]) -> Path:
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
        eval_cfg: Dict[str, Any],
) -> SplitMetrics:
    """
    Evaluiert einen Split mit den Feature-Listen aus dem Modell-Checkpoint.

    WICHTIG: Verwendet model.hparams.dataset_parameters um exakt die Features zu bekommen,
    mit denen das Modell trainiert wurde - nicht die aktuelle dataset_spec.json!
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Zielspalte {TARGET_COL!r} fehlt im DataFrame.")

    # Dataset-Parameter aus Modell-Checkpoint extrahieren
    dp = model.hparams.dataset_parameters

    # Sortierung nach Gruppen und Zeit
    df = df.sort_values(by=ID_COLS + [TIME_COL]).reset_index(drop=True)

    # Target als float32
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("float32")

    # Static categoricals zu String (TFT-Anforderung)
    for cat_col in dp['static_categoricals']:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)

    time_idx_col = dp.get('time_idx', 'time_idx')

    # TimeSeriesDataSet mit Feature-Listen aus Modell-Checkpoint
    # WICHTIG: add_encoder_length muss identisch zum Training sein!
    dataset = TimeSeriesDataSet(
        df,
        time_idx=time_idx_col,
        target=TARGET_COL,
        group_ids=dp['group_ids'],
        max_encoder_length=dp['max_encoder_length'],
        max_prediction_length=dp['max_prediction_length'],
        static_categoricals=dp['static_categoricals'],
        time_varying_known_reals=dp['time_varying_known_reals'],
        time_varying_unknown_reals=dp['time_varying_unknown_reals'],
        time_varying_known_categoricals=dp.get('time_varying_known_categoricals') or [],
        target_normalizer=GroupNormalizer(groups=ID_COLS, transformation="softplus"),
        allow_missing_timesteps=True,
        add_encoder_length=dp.get('add_encoder_length', True),
        add_relative_time_idx=dp.get('add_relative_time_idx', False),
        add_target_scales=dp.get('add_target_scales', False),
    )

    dataloader = dataset.to_dataloader(
        train=False,
        batch_size=eval_cfg["batch_size"],
        num_workers=eval_cfg["num_workers"],
    )

    # Prediction
    model.eval()
    with torch.no_grad():
        y_pred_raw = model.predict(dataloader, mode="prediction")

    y_pred = y_pred_raw.cpu().numpy()

    # Actuals extrahieren
    actuals = []
    for batch_x, batch_y in dataloader:
        target = batch_y[0]
        actuals.append(target)

    y_true = torch.cat(actuals, dim=0).cpu().numpy()

    # Quantile-Predictions: Median nehmen
    if y_pred.ndim == 3:
        y_pred = y_pred[:, :, y_pred.shape[2] // 2]

    # Flatten für Metrik-Berechnung
    y_pred = y_pred.reshape(-1)
    y_true = y_true.reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape-Mismatch: y_true={y_true.shape}, y_pred={y_pred.shape}")

    metrics_raw = _compute_metrics(y_true, y_pred)

    return SplitMetrics(
        mae=metrics_raw["mae"],
        rmse=metrics_raw["rmse"],
        mape=metrics_raw["mape"],
        smape=metrics_raw["smape"],
        r2=metrics_raw["r2"],
    )


def evaluate_tft_run(
        data_cfg: Dict[str, Any],
        model_cfg: Dict[str, Any],
        eval_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    run_id = model_cfg["run_id"]

    ckpt_path = _find_best_checkpoint(model_cfg)
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))

    df_val, df_test = _load_splits(data_cfg)

    # Feature-Listen kommen jetzt aus dem Modell-Checkpoint
    metrics_val = _evaluate_split(model, df_val, eval_cfg)
    metrics_test = _evaluate_split(model, df_test, eval_cfg)

    eval_root = Path(eval_cfg["eval_root"])
    eval_dir = eval_root / run_id
    eval_logger = EvalLogger(eval_dir)

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

    summary_path = eval_logger.log_json(payload, filename="eval_summary.json")

    # CSV pro Run
    csv_row: Dict[str, Any] = {
        "run_id": run_id,
        "checkpoint_path": str(ckpt_path),
    }

    val_dict = asdict(metrics_val)
    test_dict = asdict(metrics_test)

    for metric in EVALUATION_METRICS:
        csv_row[f"val_{metric}"] = val_dict[metric]
        csv_row[f"test_{metric}"] = test_dict[metric]

    csv_path = eval_logger.log_csv(csv_row, filename="eval_summary.csv")

    result: Dict[str, Any] = {
        "run_id": run_id,
        "metrics": payload["metrics"],
        "checkpoint_path": str(ckpt_path),
        "artifacts": {
            "eval_summary_path": str(summary_path),
            "eval_summary_csv_path": str(csv_path),
            "eval_dir": str(eval_dir),
        },
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluierung eines TFT-Runs auf Basis von Validation- und Testdaten."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Run-ID wie in logs/tft/<run_id>/ und results/tft/runs/<run_id>/",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = args.run_id

    data_cfg: Dict[str, Any] = {
        "val_path": str(BASE_DIR / "data" / "processed" / _dataset_name / "val.parquet"),
        "test_path": str(BASE_DIR / "data" / "processed" / _dataset_name / "test.parquet"),
    }

    model_cfg: Dict[str, Any] = {
        "checkpoint_root": str(BASE_DIR / "results" / "tft" / "runs"),
        "run_id": run_id,
        "checkpoint_pattern": "*.ckpt",
    }

    eval_cfg: Dict[str, Any] = {
        "eval_root": str(BASE_DIR / "results" / "tft" / "eval"),
        "batch_size": 128,
        "num_workers": 0,
    }

    result = evaluate_tft_run(data_cfg=data_cfg, model_cfg=model_cfg, eval_cfg=eval_cfg)

    print("[evaluate_tft] Evaluierung abgeschlossen.")
    print(f"- Run-ID           : {result['run_id']}")
    print(f"- Checkpoint       : {result['checkpoint_path']}")

    val_metrics = result['metrics']['val']
    test_metrics = result['metrics']['test']

    val_str = ", ".join([f"{m.upper()}={val_metrics[m]:.4f}" for m in EVALUATION_METRICS])
    test_str = ", ".join([f"{m.upper()}={test_metrics[m]:.4f}" for m in EVALUATION_METRICS])

    print(f"- Val-Metriken     : {val_str}")
    print(f"- Test-Metriken    : {test_str}")
    print(f"- Eval-Summary     : {result['artifacts']['eval_summary_path']}")


if __name__ == "__main__":
    main()

# Aufruf:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.evaluation.evaluate_tft --run-id run_20251125_215131_booksales_optuna_tft_day_trial_15