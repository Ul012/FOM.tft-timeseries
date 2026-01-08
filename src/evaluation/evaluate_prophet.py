# src/evaluation/evaluate_prophet.py
"""
Prophet Evaluation Script

Evaluiert trainierte Prophet-Modelle auf Val/Test-Daten.

Workflow:
1. Lade gespeicherte Prophet-Modelle
2. Erstelle Forecasts für Val/Test
3. Berechne Metriken (MAE, RMSE, MAPE, SMAPE)
4. Speichere Evaluations-Summary

Input:
    - results/prophet/runs/<run_id>/models/*.pkl
    - data/processed/<dataset>/val.parquet oder test.parquet

Output:
    - results/prophet/runs/<run_id>/eval_summary.json

Aufruf:
    python -m src.evaluation.evaluate_prophet --run-id <run_id> --split val
    python -m src.evaluation.evaluate_prophet --run-id <run_id> --split test
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema

_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]
_schema = get_schema(_dataset_config)


class ProphetEvaluator:
    """Evaluiert Prophet-Modelle auf Val/Test-Daten."""

    def __init__(self, run_id: str, split: str):
        self.run_id = run_id
        self.split = split  # "val" oder "test"
        self.results_dir = BASE_DIR / "results" / "prophet" / "runs" / run_id
        self.models_dir = self.results_dir / "models"

        # Lade prophet_spec
        processed_dir = BASE_DIR / "data" / "processed" / _dataset_name
        spec_path = processed_dir / "prophet_spec.json"

        if not spec_path.exists():
            raise FileNotFoundError(f"prophet_spec.json nicht gefunden: {spec_path}")

        with open(spec_path, "r", encoding="utf-8") as f:
            self.prophet_spec = json.load(f)

        # Lade Summary für prediction_length
        summary_path = self.results_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                self.summary = json.load(f)
        else:
            self.summary = None

    def _load_model(self, group_id: str):
        """Lädt gespeichertes Prophet-Modell."""
        model_path = self.models_dir / f"prophet_{group_id}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")

        with open(model_path, "rb") as f:
            return pickle.load(f)

    def _prepare_prophet_dataframe(self, df: pd.DataFrame, regressors: List[str]) -> pd.DataFrame:
        """
        Konvertiert zu Prophet-Format.

        WICHTIG: Alle Werte als float64!
        """
        time_col = self.prophet_spec["time_col"]
        target_col = self.prophet_spec["target_col"]

        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(df[time_col]),
            "y": df[target_col].astype("float64")
        })

        # Regressoren - explizit float64
        for reg in regressors:
            if reg in df.columns:
                prophet_df[reg] = pd.to_numeric(df[reg], errors='coerce').astype("float64").fillna(0.0)
            else:
                print(f"  ⚠ Regressor '{reg}' nicht in {self.split}-Daten, fülle mit 0")
                prophet_df[reg] = 0.0

        return prophet_df

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Berechnet Standard-Metriken."""
        # Entferne NaN/Inf
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {"mae": None, "rmse": None, "mape": None, "smape": None}

        # MAE
        mae = float(np.mean(np.abs(y_true - y_pred)))

        # RMSE
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        # MAPE
        mape = None
        if not (y_true == 0).any():
            mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

        # SMAPE
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        smape = None
        if not (denominator == 0).any():
            smape = float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)

        return {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "smape": smape
        }

    def evaluate_single_group(self, group_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluiert ein Prophet-Modell auf einer Gruppe.

        Args:
            group_id: Gruppe-ID
            df: Val/Test-DataFrame für diese Gruppe

        Returns:
            Dict mit Metriken
        """
        print(f"\n[Evaluation] Gruppe: {group_id}")
        print(f"  {self.split.upper()}-Länge: {len(df)}")

        # Modell laden (skip wenn nicht vorhanden)
        try:
            model = self._load_model(group_id)
        except FileNotFoundError:
            print(f"  ⚠ Modell nicht gefunden, überspringe Gruppe")
            return None

        # Prophet-Format
        regressors = self.prophet_spec["regressors"]
        prophet_df = self._prepare_prophet_dataframe(df, regressors)

        # Forecast erstellen
        try:
            forecast = model.predict(prophet_df)
        except Exception as e:
            print(f"  ✗ Forecast fehlgeschlagen: {e}")
            return {
                "mae": None,
                "rmse": None,
                "mape": None,
                "smape": None,
                "error": str(e)
            }

        # Metriken berechnen
        y_true = prophet_df["y"].values
        y_pred = forecast["yhat"].values

        metrics = self._calculate_metrics(y_true, y_pred)

        print(f"  MAE: {metrics['mae']:.2f}" if metrics['mae'] else "  MAE: N/A")
        print(f"  RMSE: {metrics['rmse']:.2f}" if metrics['rmse'] else "  RMSE: N/A")
        if metrics['mape']:
            print(f"  MAPE: {metrics['mape']:.2f}%")
        if metrics['smape']:
            print(f"  SMAPE: {metrics['smape']:.2f}%")

        return metrics

    def evaluate_all_groups(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluiert alle Gruppen.

        Args:
            df: Kompletter Val/Test-DataFrame

        Returns:
            Evaluation-Summary
        """
        group_cols = self.prophet_spec["group_cols"]
        n_groups = self.prophet_spec["n_groups"]

        print(f"\n{'=' * 60}")
        print(f"PROPHET EVALUATION - {self.split.upper()}")
        print(f"{'=' * 60}")
        print(f"Run ID: {self.run_id}")
        print(f"Dataset: {_dataset_name}")
        print(f"Anzahl Gruppen: {n_groups}")
        print(f"{'=' * 60}")

        all_metrics = {}

        if not group_cols:
            # Keine Gruppen
            metrics = self.evaluate_single_group("all", df)
            all_metrics["all"] = metrics

        else:
            # Iteriere über Gruppen
            for group_values, group_df in df.groupby(group_cols):
                if isinstance(group_values, tuple):
                    group_id = "_".join(str(v) for v in group_values)
                else:
                    group_id = str(group_values)

                metrics = self.evaluate_single_group(group_id, group_df)
                if metrics is None:
                    continue
                all_metrics[group_id] = metrics

        # Aggregierte Metriken
        valid_metrics = [m for m in all_metrics.values() if m.get("mae") is not None]

        if valid_metrics:
            avg_metrics = {
                "mae": np.mean([m["mae"] for m in valid_metrics]),
                "rmse": np.mean([m["rmse"] for m in valid_metrics]),
                "mape": np.mean([m["mape"] for m in valid_metrics if m.get("mape") is not None]),
                "smape": np.mean([m["smape"] for m in valid_metrics if m.get("smape") is not None])
            }
        else:
            avg_metrics = {"mae": None, "rmse": None, "mape": None, "smape": None}

        # Summary erstellen
        eval_summary = {
            "run_id": self.run_id,
            "dataset": _dataset_name,
            "split": self.split,
            "n_groups": n_groups,
            "metrics": {
                "by_group": all_metrics,
                "overall": avg_metrics
            }
        }

        # Speichern
        eval_path = self.results_dir / f"eval_{self.split}.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        print("EVALUATION ABGESCHLOSSEN")
        print(f"{'=' * 60}")
        if avg_metrics["mae"] is not None:
            print(f"Durchschnittliche Metriken ({self.split.upper()}):")
            print(f"  MAE:  {avg_metrics['mae']:.2f}")
            print(f"  RMSE: {avg_metrics['rmse']:.2f}")
            if avg_metrics['mape'] and not np.isnan(avg_metrics['mape']):
                print(f"  MAPE: {avg_metrics['mape']:.2f}%")
            if avg_metrics['smape'] and not np.isnan(avg_metrics['smape']):
                print(f"  SMAPE: {avg_metrics['smape']:.2f}%")
        else:
            print("⚠ Keine validen Metriken")
        print(f"\n✓ Evaluation gespeichert: {eval_path}")
        print(f"{'=' * 60}")

        return eval_summary


def main() -> None:
    """Hauptfunktion für Prophet-Evaluation."""
    parser = argparse.ArgumentParser(description="Prophet Evaluation")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run-ID des trainierten Modells"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test"],
        default="val",
        help="Split für Evaluation (val oder test)"
    )
    args = parser.parse_args()

    # Lade Split-Daten
    processed_dir = BASE_DIR / "data" / "processed" / _dataset_name
    split_path = processed_dir / f"{args.split}.parquet"

    if not split_path.exists():
        raise FileNotFoundError(f"{args.split}.parquet nicht gefunden: {split_path}")

    df_split = pd.read_parquet(split_path)

    # Evaluation starten
    evaluator = ProphetEvaluator(args.run_id, args.split)
    eval_summary = evaluator.evaluate_all_groups(df_split)


if __name__ == "__main__":
    main()

# Aufruf:
#   Walmart:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"
#   # Val-Evaluation
#   python -m src.evaluation.evaluate_prophet --run-id run_20260108_000719_prophet_baseline --split val
#
#   # Test-Evaluation
#   python -m src.evaluation.evaluate_prophet --run-id run_20260108_000719_prophet_baseline --split test
#
#   Booksales:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
#
#   # Val-Evaluation
#   python -m src.evaluation.evaluate_prophet --run-id run_20260108_000405_prophet_baseline --split val
#
#   # Test-Evaluation
#   python -m src.evaluation.evaluate_prophet --run-id run_20260108_000405_prophet_baseline --split test
