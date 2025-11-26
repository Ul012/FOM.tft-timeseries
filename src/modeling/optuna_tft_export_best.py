# src/modeling/optuna_tft_export_best.py
"""
Exportiert die beste Optuna-Konfiguration als YAML-Datei.

"""

import argparse
from pathlib import Path

import optuna
import yaml

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config

_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]

OPTUNA_BASE_DIR = BASE_DIR / "results" / "tft" / "optuna" / _dataset_name
OPTUNA_STORAGE = f"sqlite:///{OPTUNA_BASE_DIR}/tft_studies.db"


def export_best_config(study_name: str, output_path: Path | None = None):
    """
    Lädt die beste Trial aus Optuna und exportiert sie als YAML.

    Args:
        study_name: Name der Optuna Study
        output_path: Ziel-Pfad für YAML (default: configs/models/tft/optuna_<study-name>_best.yaml)
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
        print(f"  {key:<25} = {value}")
    print()

    # YAML-Config erstellen (im Format von baseline.yaml)
    config = {
        "type": "tft",
        "name": f"optuna_best_trial{best_trial_number}",
        "description": f"Beste Config aus Optuna Study '{study_name}' (Trial {best_trial_number}, val_mae={best_value:.4f})",
        "training": {
            "seed": 42,
            "max_epochs": 30,  # Für finales Training erhöhen
            "batch_size": best_params["batch_size"],
            "learning_rate": best_params["learning_rate"],
            "gradient_clip_val": best_params["gradient_clip_val"],
            "early_stopping_patience": 5,
            "accelerator": "gpu",
            "devices": 1,
            "num_workers": 4,
            "limit_train_batches": 1.0,
            "limit_val_batches": 1.0,
        },
        "model": {
            "loss": "quantile",
            "output_size": 7,
            "hidden_size": best_params["hidden_size"],
            "attention_head_size": best_params["attention_head_size"],
            "hidden_continuous_size": best_params["hidden_continuous_size"],
            "dropout": best_params["dropout"],
            "reduce_on_plateau_patience": 2,
        },
        "evaluation": {"batch_size": 256, "num_workers": 0},
        "optuna_metadata": {
            "study_name": study_name,
            "trial_number": best_trial_number,
            "val_mae": best_value,
            "all_params": best_params,
        },
    }

    # Output-Pfad
    if output_path is None:
        dataset_config = load_dataset_config()
        dataset_name = dataset_config["name"]
        output_path = BASE_DIR / "configs" / "models" / "tft" / dataset_name / f"optuna_{study_name}_best.yaml"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # YAML schreiben
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Config exportiert: {output_path}")
    print()
    print("NÄCHSTER SCHRITT:")
    print("-" * 80)
    print("Finales Training mit bester Config:")
    print(f"  python -m src.pipeline --dataset configs/datasets/booksales.yaml \\")
    print(f"                         --model {output_path} \\")
    print(f"                         --steps training")
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Exportiere beste Optuna-Config als YAML"
    )
    parser.add_argument(
        "--study-name", type=str, required=True, help="Name der Optuna Study"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output-Pfad (default: configs/models/tft/optuna_<study_name>_best.yaml)",
    )

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    export_best_config(args.study_name, output_path)


if __name__ == "__main__":
    main()

# Aufruf:
#   Booksales:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.modeling.optuna_tft_export_best --study-name tft_newyear
#
#   Walmart:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.modeling.optuna_tft_export_best --study-name walmart