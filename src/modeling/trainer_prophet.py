"""
Prophet Training Script

Trainiert Prophet-Modelle für jede Gruppe (z.B. pro Country, Store, etc.)
und speichert die Modelle sowie Forecasts.

Workflow:
1. Lade prophet_spec.json und Train-Daten
2. Iteriere über Gruppen (falls vorhanden)
3. Pro Gruppe:
   - Konvertiere zu Prophet-Format (ds/y)
   - Füge Regressoren hinzu
   - Trainiere Prophet-Modell
   - Erstelle Forecast
   - Speichere Modell
4. Aggregiere Metriken

Input:
    - data/processed/<dataset>/train.parquet
    - data/processed/<dataset>/prophet_spec.json
    - configs/models/prophet/<config>.yaml

Output:
    - results/prophet/runs/run_<id>/models/prophet_{group}.pkl
    - results/prophet/runs/run_<id>/forecasts/train_{group}.parquet
    - results/prophet/runs/run_<id>/summary.json

Aufruf:
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
    python -m src.modeling.trainer_prophet --config configs/models/prophet/baseline.yaml
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from prophet import Prophet

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema

_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]
_schema = get_schema(_dataset_config)


class ProphetTrainer:
    """Trainiert Prophet-Modelle für Time Series Forecasting."""

    def __init__(self, model_config: dict, prophet_spec: dict):
        self.model_config = model_config
        self.prophet_spec = prophet_spec
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_dataset_name}_prophet_{model_config['name']}"
        self.results_dir = BASE_DIR / "results" / "prophet" / "runs" / self.run_id
        self.models_dir = self.results_dir / "models"
        self.forecasts_dir = self.results_dir / "forecasts"

        # Erstelle Verzeichnisse
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.forecasts_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_prophet_dataframe(
            self,
            df: pd.DataFrame,
            regressors: List[str]
    ) -> pd.DataFrame:
        """
        Konvertiert DataFrame zu Prophet-Format.

        Alle Regressoren werden zu float64 konvertiert!
        Prophet hat Probleme mit anderen dtypes (int8, int32, object).

        Args:
            df: Input-DataFrame mit time_col und target_col
            regressors: Liste der Regressor-Spalten

        Returns:
            DataFrame mit 'ds', 'y' und Regressoren
        """
        time_col = self.prophet_spec["time_col"]
        target_col = self.prophet_spec["target_col"]

        # Prophet-Format: ds, y
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(df[time_col]),
            "y": df[target_col].astype("float64")
        })

        # Füge Regressoren hinzu
        for reg in regressors:
            if reg in df.columns:
                prophet_df[reg] = pd.to_numeric(df[reg], errors='coerce').astype("float64")

                if prophet_df[reg].isna().any():
                    n_nan = prophet_df[reg].isna().sum()
                    print(f"  ⚠ {reg}: {n_nan} NaN-Werte, fülle mit 0")
                    prophet_df[reg] = prophet_df[reg].fillna(0.0)
            else:
                raise ValueError(f"Regressor '{reg}' nicht in DataFrame")

        for col in prophet_df.columns:
            if col != "ds" and prophet_df[col].dtype != "float64":
                print(f"  ⚠ WARNUNG: {col} ist {prophet_df[col].dtype}, konvertiere zu float64")
                prophet_df[col] = prophet_df[col].astype("float64")

        return prophet_df

    def _create_prophet_model(self) -> Prophet:
        """
        Erstellt Prophet-Modell basierend auf Config.

        Returns:
            Konfiguriertes Prophet-Modell
        """
        model_params = self.model_config.get("model", {})

        # Prophet initialisieren
        model = Prophet(
            growth=model_params.get("growth", "linear"),
            seasonality_mode=model_params.get("seasonality_mode", "multiplicative"),
            yearly_seasonality=model_params.get("yearly_seasonality", True),
            weekly_seasonality=model_params.get("weekly_seasonality", True),
            daily_seasonality=model_params.get("daily_seasonality", False),
            changepoint_prior_scale=model_params.get("changepoint_prior_scale", 0.05),
            seasonality_prior_scale=model_params.get("seasonality_prior_scale", 10.0),
            holidays_prior_scale=model_params.get("holidays_prior_scale", 10.0),
            interval_width=model_params.get("interval_width", 0.95),
            mcmc_samples=model_params.get("mcmc_samples", 0),
        )

        # Country Holidays
        country = self.prophet_spec.get("country_holidays")
        if country:
            try:
                model.add_country_holidays(country_name=country)
                print(f"  ✓ Country Holidays: {country}")
            except Exception as e:
                print(f"  ⚠ Warnung: Country Holidays für '{country}' nicht verfügbar: {e}")

        # Regressoren hinzufügen
        for reg in self.prophet_spec["regressors"]:
            model.add_regressor(reg)

        return model

    def _make_future_dataframe(
            self,
            model: Prophet,
            train_df: pd.DataFrame,
            periods: int
    ) -> pd.DataFrame:
        """
        Erstellt Future-DataFrame mit Regressoren.

        Args:
            model: Trainiertes Prophet-Modell
            train_df: Training-Daten (mit Regressoren)
            periods: Anzahl zukünftiger Perioden

        Returns:
            Future-DataFrame mit Regressoren
        """
        # Basis Future DataFrame
        future = model.make_future_dataframe(periods=periods, include_history=True)

        regressors = self.prophet_spec["regressors"]
        if not regressors:
            return future

        # Erstelle kompletten Future-Regressor-DataFrame
        future_regs = pd.DataFrame(index=future.index)
        future_regs["ds"] = future["ds"]

        for reg in regressors:
            if reg == "month":
                future_regs[reg] = future["ds"].dt.month.astype("float64")
            elif reg == "dayofweek" or reg == "day_of_week":
                future_regs[reg] = future["ds"].dt.dayofweek.astype("float64")
            elif reg == "weekofyear" or reg == "week":
                future_regs[reg] = future["ds"].dt.isocalendar().week.astype("float64")
            elif reg == "is_weekend":
                future_regs[reg] = future["ds"].dt.dayofweek.isin([5, 6]).astype("float64")
            elif reg == "year":
                future_regs[reg] = future["ds"].dt.year.astype("float64")
            elif reg == "day":
                future_regs[reg] = future["ds"].dt.day.astype("float64")
            else:
                if reg in train_df.columns:
                    train_vals = train_df[["ds", reg]].copy()
                    train_vals[reg] = pd.to_numeric(train_vals[reg], errors='coerce')

                    merged = future_regs[["ds"]].merge(train_vals, on="ds", how="left")

                    # Fehlende Werte füllen
                    if reg.startswith("is_"):
                        future_regs[reg] = merged[reg].ffill().fillna(0).astype("float64")
                    else:
                        median_val = float(train_vals[reg].median())
                        future_regs[reg] = merged[reg].ffill().fillna(median_val).astype("float64")
                else:
                    future_regs[reg] = 0.0

        # Merge alles zusammen
        future = future.merge(future_regs.drop(columns=["ds"]), left_index=True, right_index=True)

        for reg in regressors:
            if reg in future.columns:
                future[reg] = future[reg].astype("float64")
                if future[reg].isna().any():
                    print(f"  ⚠ WARNUNG: {reg} hat noch NaN, fülle mit 0")
                    future[reg] = future[reg].fillna(0.0)

        return future

    def train_single_group(
            self,
            group_id: str,
            df: pd.DataFrame
    ) -> Tuple[Prophet, pd.DataFrame, Dict[str, float]]:
        """
        Trainiert Prophet-Modell für eine einzelne Gruppe.

        Args:
            group_id: Gruppe-ID (z.B. "DE", "Store_1")
            df: DataFrame für diese Gruppe

        Returns:
            (model, forecast, metrics)
        """
        print(f"\n[Prophet] Training: {group_id}")
        print(f"  Zeitreihen-Länge: {len(df)}")

        # Prophet-Format
        regressors = self.prophet_spec["regressors"]
        prophet_df = self._prepare_prophet_dataframe(df, regressors)

        # Modell erstellen und trainieren
        model = self._create_prophet_model()

        print(f"  Regressoren: {len(regressors)}")
        try:
            model.fit(prophet_df)
            print(f"  ✓ Training abgeschlossen")
        except Exception as e:
            print(f"  ✗ Training fehlgeschlagen: {e}")
            raise

        # Forecast erstellen
        prediction_length = self.prophet_spec["prediction_length"]
        future = self._make_future_dataframe(model, prophet_df, periods=prediction_length)

        forecast = model.predict(future)

        prophet_with_pred = prophet_df.merge(
            forecast[["ds", "yhat"]],
            on="ds",
            how="left"
        )

        y_true = prophet_with_pred["y"].values
        y_pred = prophet_with_pred["yhat"].values

        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        metrics = {
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100) if not (y_true == 0).any() else None,
        }

        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        if metrics['mape']:
            print(f"  MAPE: {metrics['mape']:.2f}%")

        return model, forecast, metrics

    def train_all_groups(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Trainiert Prophet für alle Gruppen.

        Args:
            df: Kompletter DataFrame mit allen Gruppen

        Returns:
            Summary mit Metriken und Metadaten
        """
        group_cols = self.prophet_spec["group_cols"]
        n_groups = self.prophet_spec["n_groups"]

        print(f"\n{'=' * 60}")
        print(f"PROPHET TRAINING")
        print(f"{'=' * 60}")
        print(f"Dataset: {_dataset_name}")
        print(f"Run ID: {self.run_id}")
        print(f"Anzahl Gruppen: {n_groups}")
        print(f"{'=' * 60}")

        all_metrics = {}

        if not group_cols:
            # Keine Gruppen: Trainiere auf gesamtem Dataset
            model, forecast, metrics = self.train_single_group("all", df)

            # Speichere Modell
            model_path = self.models_dir / "prophet_all.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Speichere Forecast
            forecast_path = self.forecasts_dir / "train_all.parquet"
            forecast.to_parquet(forecast_path, index=False)

            all_metrics["all"] = metrics

        else:
            # Iteriere über Gruppen
            for group_values, group_df in df.groupby(group_cols):
                # Gruppe-ID als String
                if isinstance(group_values, tuple):
                    group_id = "_".join(str(v) for v in group_values)
                else:
                    group_id = str(group_values)

                # Trainiere
                model, forecast, metrics = self.train_single_group(group_id, group_df)

                # Speichere Modell
                model_path = self.models_dir / f"prophet_{group_id}.pkl"
                with open(model_path, "wb") as f:
                    pickle.dump(model, f)

                # Speichere Forecast
                forecast_path = self.forecasts_dir / f"train_{group_id}.parquet"
                forecast.to_parquet(forecast_path, index=False)

                all_metrics[group_id] = metrics

        # Aggregierte Metriken
        avg_metrics = {
            "mae": np.mean([m["mae"] for m in all_metrics.values()]),
            "rmse": np.mean([m["rmse"] for m in all_metrics.values()]),
            "mape": np.mean([m["mape"] for m in all_metrics.values() if m["mape"] is not None])
        }

        # Summary erstellen
        summary = {
            "run_id": self.run_id,
            "dataset": _dataset_name,
            "timestamp": datetime.now().isoformat(),
            "n_groups": n_groups,
            "model_config": self.model_config,
            "prophet_spec": self.prophet_spec,
            "metrics": {
                "by_group": all_metrics,
                "average": avg_metrics
            }
        }

        # Speichere Summary
        summary_path = self.results_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 60}")
        print("TRAINING ABGESCHLOSSEN")
        print(f"{'=' * 60}")
        print(f"Durchschnittliche Metriken:")
        print(f"  MAE:  {avg_metrics['mae']:.2f}")
        print(f"  RMSE: {avg_metrics['rmse']:.2f}")
        if not np.isnan(avg_metrics['mape']):
            print(f"  MAPE: {avg_metrics['mape']:.2f}%")
        print(f"\n✓ Modelle gespeichert: {self.models_dir}")
        print(f"✓ Forecasts gespeichert: {self.forecasts_dir}")
        print(f"✓ Summary gespeichert: {summary_path}")
        print(f"{'=' * 60}")

        return summary


def load_model_config(config_path: Path) -> dict:
    """Lädt Model-Config aus YAML."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Hauptfunktion für Prophet-Training."""
    parser = argparse.ArgumentParser(description="Prophet Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Pfad zur Model-Config (z.B. configs/models/prophet/baseline.yaml)"
    )
    args = parser.parse_args()

    # Lade Configs
    model_config = load_model_config(Path(args.config))
    processed_dir = BASE_DIR / "data" / "processed" / _dataset_name

    # Lade Prophet-Spec
    spec_path = processed_dir / "prophet_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"prophet_spec.json nicht gefunden: {spec_path}\n"
            "Bitte vorher 'dataset_prophet.py' ausführen."
        )

    with open(spec_path, "r", encoding="utf-8") as f:
        prophet_spec = json.load(f)

    # Lade Train-Daten
    train_path = processed_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"train.parquet nicht gefunden: {train_path}")

    df_train = pd.read_parquet(train_path)

    # Training starten
    trainer = ProphetTrainer(model_config, prophet_spec)
    summary = trainer.train_all_groups(df_train)


if __name__ == "__main__":
    main()

# Aufruf einzeln:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
#   python -m src.modeling.trainer_prophet --config configs/models/prophet/booksales/baseline.yaml
#
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"
#   python -m src.modeling.trainer_prophet --config configs/models/prophet/walmart/baseline.yaml
