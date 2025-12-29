# src/evaluation/evaluate_arima.py
"""
ARIMA Evaluation Script

Evaluiert trainierte ARIMA-Modelle auf Val/Test-Daten.

Aufruf:
    python -m src.evaluation.evaluate_arima --run-id <run_id> --split val
    python -m src.evaluation.evaluate_arima --run-id <run_id> --split test
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


class ARIMAEvaluator:
    """Evaluiert ARIMA-Modelle auf Val/Test-Daten."""

    def __init__(self, run_id: str, split: str):
        self.run_id = run_id
        self.split = split
        self.results_dir = BASE_DIR / "results" / "arima" / "runs" / run_id
        self.models_dir = self.results_dir / "models"

        # Lade arima_spec
        processed_dir = BASE_DIR / "data" / "processed" / _dataset_name
        spec_path = processed_dir / "arima_spec.json"

        if not spec_path.exists():
            raise FileNotFoundError(f"arima_spec.json nicht gefunden: {spec_path}")

        with open(spec_path, "r", encoding="utf-8") as f:
            self.arima_spec = json.load(f)

    def _load_model(self, group_id: str):
        """Lädt gespeichertes ARIMA-Modell."""
        model_path = self.models_dir / f"arima_{group_id}.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Modell nicht gefunden: {model_path}")

        with open(model_path, "rb") as f:
            return pickle.load(f)

    def _prepare_exog(self, df: pd.DataFrame, exog_vars: List[str]) -> pd.DataFrame:
        """Bereitet exogene Variablen vor."""
        if not exog_vars:
            return None

        exog = df[exog_vars].copy()
        for col in exog.columns:
            exog[col] = pd.to_numeric(exog[col], errors='coerce').fillna(0).astype("float64")

        return exog

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Berechnet Standard-Metriken."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {"mae": None, "rmse": None, "mape": None, "smape": None}

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        mape = None
        if not (y_true == 0).any():
            mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

        smape = None
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        if not (denominator == 0).any():
            smape = float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)

        return {"mae": mae, "rmse": rmse, "mape": mape, "smape": smape}

    def evaluate_single_group(self, group_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluiert ein ARIMA-Modell auf einer Gruppe.

        Args:
            group_id: Gruppe-ID
            df: Val/Test-DataFrame für diese Gruppe

        Returns:
            Dict mit Metriken
        """
        print(f"\n[Evaluation] Gruppe: {group_id}")
        print(f"  {self.split.upper()}-Länge: {len(df)}")

        # Modell laden
        model = self._load_model(group_id)

        # Exog vorbereiten
        exog_vars = self.arima_spec["exog_vars"]
        exog = self._prepare_exog(df, exog_vars)

        # Forecast erstellen
        try:
            n_periods = len(df)

            # Forecast mit exog
            if exog is not None:
                forecast = model.predict(n_periods=n_periods, exogenous=exog)
            else:
                forecast = model.predict(n_periods=n_periods)

            y_pred = forecast.values if hasattr(forecast, 'values') else forecast

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
        target_col = self.arima_spec["target_col"]
        y_true = df[target_col].values

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
        group_cols = self.arima_spec["group_cols"]
        n_groups = self.arima_spec["n_groups"]

        print(f"\n{'=' * 60}")
        print(f"ARIMA EVALUATION - {self.split.upper()}")
        print(f"{'=' * 60}")
        print(f"Run ID: {self.run_id}")
        print(f"Dataset: {_dataset_name}")
        print(f"Anzahl Gruppen: {n_groups}")
        print(f"{'=' * 60}")

        all_metrics = {}

        if not group_cols:
            metrics = self.evaluate_single_group("all", df)
            all_metrics["all"] = metrics
        else:
            for group_values, group_df in df.groupby(group_cols):
                if isinstance(group_values, tuple):
                    group_id = "_".join(str(v) for v in group_values)
                else:
                    group_id = str(group_values)

                try:
                    metrics = self.evaluate_single_group(group_id, group_df)
                    all_metrics[group_id] = metrics
                except Exception as e:
                    print(f"  ✗ Gruppe {group_id} übersprungen: {e}")
                    all_metrics[group_id] = {
                        "mae": None,
                        "rmse": None,
                        "mape": None,
                        "smape": None,
                        "error": str(e)
                    }

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
            "n_successful": len(valid_metrics),
            "n_failed": n_groups - len(valid_metrics),
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
        print(f"Erfolgreich: {len(valid_metrics)} / {n_groups}")
        if avg_metrics["mae"] is not None:
            print(f"Durchschnittliche Metriken ({self.split.upper()}):")
            print(f"  MAE:  {avg_metrics['mae']:.2f}")
            print(f"  RMSE: {avg_metrics['rmse']:.2f}")
            if avg_metrics['mape'] and not np.isnan(avg_metrics['mape']):
                print(f"  MAPE: {avg_metrics['mape']:.2f}%")
            if avg_metrics['smape'] and not np.isnan(avg_metrics['smape']):
                print(f"  SMAPE: {avg_metrics['smape']:.2f}%")
        print(f"\n✓ Evaluation gespeichert: {eval_path}")
        print(f"{'=' * 60}")

        return eval_summary


def main() -> None:
    """Hauptfunktion für ARIMA-Evaluation."""
    parser = argparse.ArgumentParser(description="ARIMA Evaluation")
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
    evaluator = ARIMAEvaluator(args.run_id, args.split)
    eval_summary = evaluator.evaluate_all_groups(df_split)


if __name__ == "__main__":
    main()

# Aufruf:
#   python -m src.evaluation.evaluate_arima --run-id run_20251228_150000_arima_baseline --split val
#   python -m src.evaluation.evaluate_arima --run-id run_20251228_150000_arima_baseline --split test