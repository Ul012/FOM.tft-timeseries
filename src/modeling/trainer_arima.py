"""
ARIMA Training Script

Trainiert ARIMA/SARIMA-Modelle für jede Gruppe mit auto_arima oder manuellen Parametern.

ARIMA = AutoRegressive Integrated Moving Average
- AR(p): Autoregressive Teil (Abhängigkeit von vergangenen Werten)
- I(d): Integrated (Differencing für Stationarität)
- MA(q): Moving Average (Abhängigkeit von vergangenen Residuen)

SARIMA = Seasonal ARIMA
- Zusätzliche saisonale Komponenten (P,D,Q,m)
- m = seasonal_period (7 für täglich, 52 für wöchentlich, 12 für monatlich)

Input:
    - data/processed/<dataset>/train.parquet
    - data/processed/<dataset>/arima_spec.json
    - configs/models/arima/<config>.yaml

Output:
    - results/arima/runs/run_<id>/models/arima_{group}.pkl
    - results/arima/runs/run_<id>/forecasts/train_{group}.parquet
    - results/arima/runs/run_<id>/summary.json

Aufruf:
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
    python -m src.modeling.trainer_arima --config configs/models/arima/baseline.yaml
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    from pmdarima import auto_arima

    HAS_AUTO_ARIMA = True
except ImportError:
    HAS_AUTO_ARIMA = False
    warnings.warn("pmdarima nicht installiert - auto_arima nicht verfügbar. Installieren mit: pip install pmdarima")

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema

_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]
_schema = get_schema(_dataset_config)

# Warnings filtern (ARIMA kann viele Convergence-Warnings geben)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class ARIMATrainer:
    """Trainiert ARIMA/SARIMA-Modelle für Time Series Forecasting."""

    def __init__(self, model_config: dict, arima_spec: dict):
        self.model_config = model_config
        self.arima_spec = arima_spec
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_arima_{model_config['name']}"
        self.results_dir = BASE_DIR / "results" / "arima" / "runs" / self.run_id
        self.models_dir = self.results_dir / "models"
        self.forecasts_dir = self.results_dir / "forecasts"

        # Erstelle Verzeichnisse
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.forecasts_dir.mkdir(parents=True, exist_ok=True)

    def _prepare_arima_data(
            self,
            df: pd.DataFrame,
            exog_vars: List[str]
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Konvertiert DataFrame zu ARIMA-Format.

        Args:
            df: Input-DataFrame
            exog_vars: Liste der exogenen Variablen

        Returns:
            (endog, exog) - endogene Variable (y) und exogene Variablen (X)
        """
        time_col = self.arima_spec["time_col"]
        target_col = self.arima_spec["target_col"]

        # Sortiere nach Zeit
        df = df.sort_values(time_col).reset_index(drop=True)

        # Endogene Variable (Target)
        endog = df[target_col].astype("float64")

        # Exogene Variablen (Regressoren)
        if exog_vars:
            exog = df[exog_vars].copy()
            # Konvertiere alles zu float64
            for col in exog.columns:
                exog[col] = pd.to_numeric(exog[col], errors='coerce').fillna(0).astype("float64")
        else:
            exog = None

        return endog, exog

    def _fit_auto_arima(
            self,
            endog: pd.Series,
            exog: pd.DataFrame,
            seasonal_period: int
    ):
        """
        Trainiert ARIMA mit automatischer Parametersuche (auto_arima).

        Args:
            endog: Endogene Variable
            exog: Exogene Variablen
            seasonal_period: m-Wert für SARIMA

        Returns:
            Fitted auto_arima model
        """
        if not HAS_AUTO_ARIMA:
            raise ImportError(
                "auto_arima benötigt pmdarima. Installieren mit: pip install pmdarima"
            )

        model_params = self.model_config.get("model", {})

        # seasonal aus Config lesen
        seasonal = model_params.get("seasonal", True)  # default True für Rückwärtskompatibilität

        model = auto_arima(
            endog,
            exogenous=exog,
            seasonal=seasonal,  # ← AUS CONFIG
            m=seasonal_period if seasonal else 1,  # ← m=1 wenn non-seasonal
            max_p=model_params.get("max_p", 3),
            max_q=model_params.get("max_q", 3),
            max_d=model_params.get("max_d", 2),
            max_P=model_params.get("max_P", 2) if seasonal else 0,  # ← 0 wenn non-seasonal
            max_Q=model_params.get("max_Q", 2) if seasonal else 0,
            max_D=model_params.get("max_D", 1) if seasonal else 0,
            start_p=0,  # Statt 1
            start_q=0,  # Statt 1
            start_P=0,  # Statt "1 if seasonal else 0"
            start_Q=0,  # Statt "1 if seasonal else 0"
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False,
        )

        return model

    def _fit_manual_arima(
            self,
            endog: pd.Series,
            exog: pd.DataFrame,
            order: Tuple[int, int, int],
            seasonal_order: Tuple[int, int, int, int]
    ):
        """
        Trainiert ARIMA mit manuellen Parametern (SARIMAX).

        Args:
            endog: Endogene Variable
            exog: Exogene Variablen
            order: (p, d, q)
            seasonal_order: (P, D, Q, m)

        Returns:
            Fitted SARIMAX model
        """
        model = SARIMAX(
            endog,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted_model = model.fit(disp=False)

        return fitted_model

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Berechnet Standard-Metriken."""
        # Entferne NaN/Inf
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            return {"mae": None, "rmse": None, "mape": None}

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        mape = None
        if not (y_true == 0).any():
            mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

        return {"mae": mae, "rmse": rmse, "mape": mape}

    def train_single_group(
            self,
            group_id: str,
            df: pd.DataFrame
    ) -> Tuple[Any, pd.DataFrame, Dict[str, float]]:
        """
        Trainiert ARIMA-Modell für eine einzelne Gruppe.

        Args:
            group_id: Gruppe-ID
            df: DataFrame für diese Gruppe

        Returns:
            (model, forecast_df, metrics)
        """
        print(f"\n[ARIMA] Training: {group_id}")
        print(f"  Zeitreihen-Länge: {len(df)}")

        # Daten vorbereiten
        exog_vars = self.arima_spec["exog_vars"]
        endog, exog = self._prepare_arima_data(df, exog_vars)

        print(f"  Exogene Variablen: {len(exog_vars) if exog_vars else 0}")

        # Training
        auto_arima_enabled = self.arima_spec["auto_arima"]
        seasonal_period = self.arima_spec["seasonal_period"]

        try:
            if auto_arima_enabled:
                print(f"  Training mit auto_arima (m={seasonal_period})...")
                model = self._fit_auto_arima(endog, exog, seasonal_period)
                print(f"  Beste Order: {model.order}, Seasonal Order: {model.seasonal_order}")
            else:
                order = tuple(self.arima_spec["order"])
                seasonal_order = tuple(self.arima_spec["seasonal_order"])
                print(f"  Training mit SARIMAX{order} × {seasonal_order}...")
                model = self._fit_manual_arima(endog, exog, order, seasonal_order)

            print(f"  ✓ Training abgeschlossen")

        except Exception as e:
            print(f"  ✗ Training fehlgeschlagen: {e}")
            raise

        # In-Sample Forecast (für Metriken)
        try:
            if auto_arima_enabled:
                y_pred = model.predict_in_sample(exogenous=exog)
            else:
                y_pred = model.fittedvalues

            y_true = endog.values

            # Metriken berechnen
            metrics = self._calculate_metrics(y_true, y_pred)

            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  RMSE: {metrics['rmse']:.2f}")
            if metrics['mape']:
                print(f"  MAPE: {metrics['mape']:.2f}%")

        except Exception as e:
            print(f"  ⚠ Metriken-Berechnung fehlgeschlagen: {e}")
            metrics = {"mae": None, "rmse": None, "mape": None}

        # Forecast-DataFrame erstellen (nur In-Sample für jetzt)
        time_col = self.arima_spec["time_col"]
        target_col = self.arima_spec["target_col"]

        forecast_df = pd.DataFrame({
            time_col: df[time_col].values,
            target_col: y_true,
            "yhat": y_pred
        })

        return model, forecast_df, metrics

    def train_all_groups(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Trainiert ARIMA für alle Gruppen.

        Args:
            df: Kompletter DataFrame mit allen Gruppen

        Returns:
            Summary mit Metriken und Metadaten
        """
        group_cols = self.arima_spec["group_cols"]
        n_groups = self.arima_spec["n_groups"]

        print(f"\n{'=' * 60}")
        print(f"ARIMA TRAINING")
        print(f"{'=' * 60}")
        print(f"Dataset: {_dataset_name}")
        print(f"Run ID: {self.run_id}")
        print(f"Anzahl Gruppen: {n_groups}")
        print(f"Auto-ARIMA: {self.arima_spec['auto_arima']}")
        print(f"Frequenz: {self.arima_spec['frequency']} (m={self.arima_spec['seasonal_period']})")
        print(f"{'=' * 60}")

        all_metrics = {}

        if not group_cols:
            # Keine Gruppen: Trainiere auf gesamtem Dataset
            model, forecast, metrics = self.train_single_group("all", df)

            # Speichere Modell
            model_path = self.models_dir / "arima_all.pkl"
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
                try:
                    model, forecast, metrics = self.train_single_group(group_id, group_df)

                    # Speichere Modell
                    model_path = self.models_dir / f"arima_{group_id}.pkl"
                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)

                    # Speichere Forecast
                    forecast_path = self.forecasts_dir / f"train_{group_id}.parquet"
                    forecast.to_parquet(forecast_path, index=False)

                    all_metrics[group_id] = metrics

                except Exception as e:
                    print(f"  ✗ Gruppe {group_id} übersprungen: {e}")
                    all_metrics[group_id] = {
                        "mae": None,
                        "rmse": None,
                        "mape": None,
                        "error": str(e)
                    }

        # Aggregierte Metriken
        valid_metrics = [m for m in all_metrics.values() if m.get("mae") is not None]

        if valid_metrics:
            avg_metrics = {
                "mae": np.mean([m["mae"] for m in valid_metrics]),
                "rmse": np.mean([m["rmse"] for m in valid_metrics]),
                "mape": np.mean([m["mape"] for m in valid_metrics if m.get("mape") is not None])
            }
        else:
            avg_metrics = {"mae": None, "rmse": None, "mape": None}

        # Summary erstellen
        summary = {
            "run_id": self.run_id,
            "dataset": _dataset_name,
            "timestamp": datetime.now().isoformat(),
            "n_groups": n_groups,
            "n_successful": len(valid_metrics),
            "n_failed": n_groups - len(valid_metrics),
            "model_config": self.model_config,
            "arima_spec": self.arima_spec,
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
        print(f"Erfolgreich: {len(valid_metrics)} / {n_groups}")
        if valid_metrics:
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
    """Hauptfunktion für ARIMA-Training."""
    parser = argparse.ArgumentParser(description="ARIMA Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Pfad zur Model-Config (z.B. configs/models/arima/baseline.yaml)"
    )
    args = parser.parse_args()

    # Lade Configs
    model_config = load_model_config(Path(args.config))
    processed_dir = BASE_DIR / "data" / "processed" / _dataset_name

    # Lade ARIMA-Spec
    spec_path = processed_dir / "arima_spec.json"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"arima_spec.json nicht gefunden: {spec_path}\n"
            "Bitte vorher 'dataset_arima.py' ausführen."
        )

    with open(spec_path, "r", encoding="utf-8") as f:
        arima_spec = json.load(f)

    # Lade Train-Daten
    train_path = processed_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"train.parquet nicht gefunden: {train_path}")

    df_train = pd.read_parquet(train_path)

    # Training starten
    trainer = ARIMATrainer(model_config, arima_spec)
    summary = trainer.train_all_groups(df_train)


if __name__ == "__main__":
    main()

# Aufruf:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.modeling.trainer_arima --config configs/models/arima/booksales/optuna_arima_booksales_best.yaml
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.modeling.trainer_arima --config configs/models/arima/booksales/optuna_arima_booksales_trial_11.yaml
#
#   # Oder für Walmart:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.modeling.trainer_arima --config configs/models/arima/walmart/optuna_arima_walmart_nonseasonal_best.yaml