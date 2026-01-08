# src/modeling/optuna_prophet_export_best.py
"""
Exportiert die beste Optuna-Konfiguration als YAML-Datei für Prophet.

Aufruf:
    Booksales:
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
    python -m src.modeling.optuna_prophet_export_best --study-name prophet_booksales

    Walmart:
    $env:DATASET_CONFIG="configs/datasets/walmart.yaml"
    python -m src.modeling.optuna_prophet_export_best --study-name prophet_walmart
"""

import argparse
from pathlib import Path

import optuna
import yaml

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config

_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]

OPTUNA_BASE_DIR = BASE_DIR / "results" / "prophet" / "optuna" / _dataset_name
OPTUNA_STORAGE = f"sqlite:///{OPTUNA_BASE_DIR}/prophet_studies.db"


def export_best_config(study_name: str, output_path: Path | None = None):
    """
    Lädt die beste Trial aus Optuna und exportiert sie als YAML.

    Args:
        study_name: Name der Optuna Study
        output_path: Ziel-Pfad für YAML (default: configs/models/prophet/<dataset>/optuna_<study-name>_best.yaml)
    """
    # Study laden
    study = optuna.load_study(study_name=study_name, storage=OPTUNA_STORAGE)

    if not study.best_trial:
        raise ValueError(f"Study '{study_name}' hat keine abgeschlossenen Trials!")

    best_params = study.best_params
    best_value = study.best_value
    best_trial_number = study.best_trial.number

    print("=" * 80)
    print(f"BESTE KONFIGURATION AUS STUDY: {study_name}")
    print("=" * 80)
    print()
    print(f"Trial Number:  {best_trial_number}")
    print(f"val_mae:       {best_value:.4f}")
    print()
    print("Hyperparameter:")
    print("-" * 80)
    for key, value in best_params.items():
        print(f"  {key:<30} = {value}")
    print()

    # YAML-Config erstellen
    config = {
        "type": "prophet",
        "name": f"optuna_best_trial{best_trial_number}",
        "description": f"Beste Config aus Optuna Study '{study_name}' (Trial {best_trial_number}, val_mae={best_value:.4f})",
        "model": {
            "growth": best_params["growth"],
            "seasonality_mode": best_params["seasonality_mode"],
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "changepoint_prior_scale": best_params["changepoint_prior_scale"],
            "seasonality_prior_scale": best_params["seasonality_prior_scale"],
            "holidays_prior_scale": best_params["holidays_prior_scale"],
            "interval_width": 0.95,
            "mcmc_samples": 0
        },
        "training": {
            "prediction_length": _dataset_config["prediction_length"]
        },
        "optuna_metadata": {
            "study_name": study_name,
            "trial_number": best_trial_number,
            "val_mae": best_value,
            "all_params": best_params,
        },
    }

    # Output-Pfad
    if output_path is None:
        output_path = BASE_DIR / "configs" / "models" / "prophet" / _dataset_name / f"optuna_{study_name}_best.yaml"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # YAML schreiben
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Config exportiert: {output_path}")
    print()
    print("NÄCHSTER SCHRITT:")
    print("-" * 80)
    print("Finales Training mit bester Config:")
    print(f"  $env:DATASET_CONFIG='configs/datasets/{_dataset_name}.yaml'")
    print(f"  python -m src.modeling.trainer_prophet --config {output_path}")
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Exportiere beste Optuna-Config als YAML für Prophet"
    )
    parser.add_argument(
        "--study-name", type=str, required=True, help="Name der Optuna Study"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output-Pfad (default: configs/models/prophet/<dataset>/optuna_<study_name>_best.yaml)",
    )

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    export_best_config(args.study_name, output_path)


if __name__ == "__main__":
    main()

# Aufruf:
#   Booksales:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
#   python -m src.modeling.optuna_prophet_export_best --study-name prophet_booksales
#
#   Walmart:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"
#   python -m src.modeling.optuna_prophet_export_best --study-name prophet_walmart