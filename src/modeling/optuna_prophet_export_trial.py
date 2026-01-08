# src/modeling/optuna_prophet_export_trial.py
"""
Exportiert eine spezifische Trial-Config als YAML für Prophet.

Nützlich wenn du z.B. Trial #15 nochmal trainieren willst.

Aufruf:
    Booksales:
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
    python -m src.modeling.optuna_prophet_export_trial --study-name prophet_booksales --trial-number 5

    Walmart:
    $env:DATASET_CONFIG="configs/datasets/walmart.yaml"
    python -m src.modeling.optuna_prophet_export_trial --study-name prophet_walmart --trial-number 12
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


def export_trial_config(study_name: str, trial_number: int, output_path: Path | None = None):
    """
    Lädt eine spezifische Trial aus Optuna und exportiert sie als YAML.

    Args:
        study_name: Name der Optuna Study
        trial_number: Trial-Nummer (0, 1, 2, ...)
        output_path: Ziel-Pfad für YAML
    """
    # Study laden
    study = optuna.load_study(study_name=study_name, storage=OPTUNA_STORAGE)

    # Trial finden
    trial = None
    for t in study.trials:
        if t.number == trial_number:
            trial = t
            break

    if trial is None:
        raise ValueError(
            f"Trial {trial_number} nicht gefunden in Study '{study_name}'!"
        )

    # Trial-Daten
    trial_params = trial.params
    trial_value = trial.value if trial.value is not None else "PRUNED/FAILED"
    trial_state = trial.state.name

    print("=" * 80)
    print(f"TRIAL-CONFIG EXPORT: {study_name} / Trial #{trial_number}")
    print("=" * 80)
    print()
    print(f"Trial Number:  {trial_number}")
    print(f"Trial State:   {trial_state}")
    print(f"val_mae:       {trial_value}")
    print()
    print("Hyperparameter:")
    print("-" * 80)
    for key, value in trial_params.items():
        print(f"  {key:<30} = {value}")
    print()

    if trial_state != "COMPLETE":
        print(f"⚠️  WARNUNG: Trial wurde {trial_state} (nicht COMPLETE)")
        print("   Training mit dieser Config könnte nicht erfolgreich sein.")
        print()

    # YAML-Config erstellen
    config = {
        "type": "prophet",
        "name": f"trial{trial_number}",
        "description": f"Trial {trial_number} aus Optuna Study '{study_name}' (State: {trial_state}, val_mae={trial_value})",
        "model": {
            "growth": trial_params["growth"],
            "seasonality_mode": trial_params["seasonality_mode"],
            "yearly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False,
            "changepoint_prior_scale": trial_params["changepoint_prior_scale"],
            "seasonality_prior_scale": trial_params["seasonality_prior_scale"],
            "holidays_prior_scale": trial_params["holidays_prior_scale"],
            "interval_width": 0.95,
            "mcmc_samples": 0
        },
        "training": {
            "prediction_length": _dataset_config["prediction_length"]
        },
        "optuna_metadata": {
            "study_name": study_name,
            "trial_number": trial_number,
            "trial_state": trial_state,
            "val_mae": trial_value if isinstance(trial_value, float) else None,
            "all_params": trial_params,
        },
    }

    # Output-Pfad
    if output_path is None:
        output_path = BASE_DIR / "configs" / "models" / "prophet" / _dataset_name / f"optuna_{study_name}_trial_{trial_number}.yaml"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # YAML schreiben
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Config exportiert: {output_path}")
    print()
    print("NÄCHSTER SCHRITT:")
    print("-" * 80)
    print("Training mit dieser Config:")
    print(f"  $env:DATASET_CONFIG='configs/datasets/{_dataset_name}.yaml'")
    print(f"  python -m src.modeling.trainer_prophet --config {output_path}")
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Exportiere spezifische Trial-Config als YAML für Prophet"
    )
    parser.add_argument(
        "--study-name", type=str, required=True, help="Name der Optuna Study"
    )
    parser.add_argument(
        "--trial-number", type=int, required=True, help="Trial-Nummer (0, 1, 2, ...)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output-Pfad (default: configs/models/prophet/<dataset>/optuna_<study_name>_trial_<N>.yaml)",
    )

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    export_trial_config(args.study_name, args.trial_number, output_path)


if __name__ == "__main__":
    main()

# Aufruf:
#   Booksales:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"
#   python -m src.modeling.optuna_prophet_export_trial --study-name prophet_booksales --trial-number 5
#
#   Walmart:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"
#   python -m src.modeling.optuna_prophet_export_trial --study-name prophet_walmart --trial-number 12