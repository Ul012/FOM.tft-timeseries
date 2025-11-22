# src/modeling/optuna_tft_export_trial.py
"""
Exportiert eine spezifische Trial-Config als YAML (nicht nur die beste).

Nützlich wenn du z.B. Trial #15 nochmal trainieren willst.

Aufruf:
    python -m src.modeling.optuna_tft_export_trial --study-name tft_day --trial-number 15
    python -m src.modeling.optuna_tft_export_trial --study-name tft_day --trial-number 15 --output configs/models/tft/optuna_tft_day_trial_15.yaml
"""

import argparse
from pathlib import Path

import optuna
import yaml

from src.config import BASE_DIR

OPTUNA_STORAGE = "sqlite:///results/tft/optuna/tft_studies.db"


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
        print(f"  {key:<25} = {value}")
    print()

    if trial_state != "COMPLETE":
        print(f"⚠️  WARNUNG: Trial wurde {trial_state} (nicht COMPLETE)")
        print("   Training mit dieser Config könnte nicht erfolgreich sein.")
        print()

    # YAML-Config erstellen
    config = {
        "type": "tft",
        "name": f"trial{trial_number}",
        "description": f"Trial {trial_number} aus Optuna Study '{study_name}' (State: {trial_state}, val_mae={trial_value})",
        "training": {
            "seed": 42,
            "max_epochs": 30,
            "batch_size": trial_params["batch_size"],
            "learning_rate": trial_params["learning_rate"],
            "gradient_clip_val": trial_params["gradient_clip_val"],
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
            "hidden_size": trial_params["hidden_size"],
            "attention_head_size": trial_params["attention_head_size"],
            "hidden_continuous_size": trial_params["hidden_continuous_size"],
            "dropout": trial_params["dropout"],
            "reduce_on_plateau_patience": 2,
        },
        "evaluation": {"batch_size": 256, "num_workers": 0},
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
        output_path = BASE_DIR / "configs" / "models" / "tft" / f"trial_{trial_number}.yaml"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # YAML schreiben
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Config exportiert: {output_path}")
    print()
    print("NÄCHSTER SCHRITT:")
    print("-" * 80)
    print("Training mit dieser Config:")
    print(f"  python -m src.pipeline --dataset configs/datasets/booksales.yaml \\")
    print(f"                         --model {output_path} \\")
    print(f"                         --steps training")
    print()
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Exportiere spezifische Trial-Config als YAML"
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
        help="Output-Pfad (default: configs/models/tft/trial_<N>.yaml)",
    )

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None

    export_trial_config(args.study_name, args.trial_number, output_path)


if __name__ == "__main__":
    main()