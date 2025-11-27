# src/visualization/plot_tft_interpretation.py
"""
Visualisiert die Feature Importance und Attention Weights eines trainierten TFT-Modells.

Zeigt:
1. Variable Importance (Encoder & Decoder)
2. Attention Weights (welche vergangenen Zeitpunkte sind wichtig)
3. Static Variable Importance

Aufruf:
    Mit run-id:
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.visualization.plot_tft_interpretation --run-id run_20251125_003840_booksales_optuna_tft_day_best --split test

    Mit checkpoint (für Optuna-Trials):
    $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.visualization.plot_tft_interpretation --checkpoint "results\\tft\\optuna\\walmart\\trial_0020\\checkpoints\\tft-epoch=08-val_loss=790.6539.ckpt" --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting.models import TemporalFusionTransformer

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema, get_tft_config

# Lade Dataset-Config
_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]
_schema = get_schema(_dataset_config)

TIME_COL = _schema["time_col"]
ID_COLS = _schema["id_cols"]
TARGET_COL = _schema["target_col"]
TFT_DATASET = get_tft_config(_dataset_config)


# -----------------------------------------------------------------------------
# Argumente
# -----------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualisiert Feature Importance und Attention Weights eines TFT-Modells."
    )
    parser.add_argument(
        "--run-id",
        required=False,
        help="Run-ID wie in results/tft/runs/<run_id>/",
    )
    parser.add_argument(
        "--checkpoint",
        required=False,
        help="Direkter Pfad zum Checkpoint (für Optuna-Trials)",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["val", "test"],
        help="Zu analysierender Split (val oder test).",
    )
    args = parser.parse_args()

    if not args.run_id and not args.checkpoint:
        parser.error("Entweder --run-id oder --checkpoint muss angegeben werden.")

    return args


# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------
def _find_checkpoint(run_id: str) -> Path:
    """Findet den Checkpoint für einen Run."""
    ckpt_dir = BASE_DIR / "results" / "tft" / "runs" / run_id / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint-Verzeichnis nicht gefunden: {ckpt_dir}")

    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"Keine .ckpt-Dateien in {ckpt_dir} gefunden.")

    best_ckpts = [p for p in ckpts if "best" in p.name.lower()]
    return best_ckpts[0] if best_ckpts else ckpts[0]


def _load_split(split: str) -> pd.DataFrame:
    """Lädt den gewünschten Split."""
    path = BASE_DIR / "data" / "processed" / _dataset_name / f"{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Split-Datei nicht gefunden: {path}")
    return pd.read_parquet(path)


def _create_dataset_for_interpretation(
    model: TemporalFusionTransformer,
    df: pd.DataFrame,
) -> TimeSeriesDataSet:
    """Erstellt ein TimeSeriesDataSet mit den Parametern aus dem Modell-Checkpoint."""
    dp = model.hparams.dataset_parameters

    df = df.sort_values(by=ID_COLS + [TIME_COL]).reset_index(drop=True)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("float32")

    for cat_col in dp['static_categoricals']:
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype(str)

    time_idx_col = dp.get('time_idx', 'time_idx')

    dataset = TimeSeriesDataSet(
        df,
        time_idx=time_idx_col,
        target=TARGET_COL,
        group_ids=dp['group_ids'],
        max_encoder_length=dp['max_encoder_length'],
        max_prediction_length=dp['max_prediction_length'],
        min_encoder_length=dp.get('min_encoder_length', dp['max_encoder_length'] // 2),
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

    return dataset


def _get_interpretation(
    model: TemporalFusionTransformer,
    dataloader: torch.utils.data.DataLoader,
) -> Dict[str, Any]:
    """
    Führt Predictions durch und extrahiert Interpretation.
    """
    model.eval()

    # Predictions im raw-Modus für Interpretation
    raw_predictions = model.predict(dataloader, mode="raw", return_x=False)

    # Interpretation extrahieren
    interpretation = model.interpret_output(raw_predictions, reduction="sum")

    return interpretation


# -----------------------------------------------------------------------------
# Plot-Funktionen
# -----------------------------------------------------------------------------
def _plot_variable_importance(
    interpretation: Dict[str, Any],
    model: TemporalFusionTransformer,
    output_dir: Path,
    run_id: str,
) -> None:
    """
    Plottet die Variable Importance für Encoder und Decoder.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Encoder Variables (unknown reals + known reals im Encoder)
    if "encoder_variables" in interpretation:
        encoder_imp = interpretation["encoder_variables"]
        # Tensor zu numpy konvertieren
        if hasattr(encoder_imp, 'cpu'):
            encoder_imp = encoder_imp.cpu().numpy()

        # Feature-Namen aus dem Modell
        encoder_names = model.encoder_variables

        if len(encoder_imp) == len(encoder_names):
            # Sortieren nach Importance
            sorted_idx = np.argsort(encoder_imp)
            sorted_imp = encoder_imp[sorted_idx]
            sorted_names = [encoder_names[i] for i in sorted_idx]

            # Top 20 anzeigen
            n_show = min(20, len(sorted_names))
            axes[0].barh(range(n_show), sorted_imp[-n_show:])
            axes[0].set_yticks(range(n_show))
            axes[0].set_yticklabels(sorted_names[-n_show:])
            axes[0].set_xlabel("Importance")
            axes[0].set_title("Encoder Variable Importance (Top 20)")
        else:
            axes[0].text(0.5, 0.5, f"Shape mismatch: {len(encoder_imp)} vs {len(encoder_names)}",
                        ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title("Encoder Variables")

    # Decoder Variables (known reals im Decoder)
    if "decoder_variables" in interpretation:
        decoder_imp = interpretation["decoder_variables"]
        # Tensor zu numpy konvertieren
        if hasattr(decoder_imp, 'cpu'):
            decoder_imp = decoder_imp.cpu().numpy()

        decoder_names = model.decoder_variables

        if len(decoder_imp) == len(decoder_names):
            sorted_idx = np.argsort(decoder_imp)
            sorted_imp = decoder_imp[sorted_idx]
            sorted_names = [decoder_names[i] for i in sorted_idx]

            n_show = min(20, len(sorted_names))
            axes[1].barh(range(n_show), sorted_imp[-n_show:])
            axes[1].set_yticks(range(n_show))
            axes[1].set_yticklabels(sorted_names[-n_show:])
            axes[1].set_xlabel("Importance")
            axes[1].set_title("Decoder Variable Importance (Top 20)")
        else:
            axes[1].text(0.5, 0.5, f"Shape mismatch: {len(decoder_imp)} vs {len(decoder_names)}",
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title("Decoder Variables")

    fig.suptitle(f"TFT Variable Importance | Run: {run_id}", fontsize=12)
    fig.tight_layout()

    output_path = output_dir / f"{run_id}_variable_importance.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Variable Importance Plot: {output_path}")
    plt.close(fig)


def _plot_static_variables(
    interpretation: Dict[str, Any],
    model: TemporalFusionTransformer,
    output_dir: Path,
    run_id: str,
) -> None:
    """
    Plottet die Static Variable Importance (z.B. Store, Dept).
    """
    if "static_variables" not in interpretation:
        print("[INFO] Keine Static Variables im Modell.")
        return

    static_imp = interpretation["static_variables"]
    # Tensor zu numpy konvertieren
    if hasattr(static_imp, 'cpu'):
        static_imp = static_imp.cpu().numpy()

    static_names = model.static_variables

    if len(static_imp) != len(static_names):
        print(f"[WARNUNG] Static variables shape mismatch: {len(static_imp)} vs {len(static_names)}")
        return

    fig, ax = plt.subplots(figsize=(8, 4))

    sorted_idx = np.argsort(static_imp)
    sorted_imp = static_imp[sorted_idx]
    sorted_names = [static_names[i] for i in sorted_idx]

    ax.barh(range(len(sorted_names)), sorted_imp)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Importance")
    ax.set_title(f"Static Variable Importance | Run: {run_id}")

    fig.tight_layout()

    output_path = output_dir / f"{run_id}_static_importance.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Static Variable Importance Plot: {output_path}")
    plt.close(fig)


def _plot_attention_weights(
    interpretation: Dict[str, Any],
    output_dir: Path,
    run_id: str,
) -> None:
    """
    Plottet die Attention Weights als Heatmap.
    Zeigt, welche vergangenen Zeitpunkte für die Vorhersage wichtig sind.
    """
    if "attention" not in interpretation:
        print("[INFO] Keine Attention Weights verfügbar.")
        return

    attention = interpretation["attention"]
    # Tensor zu numpy konvertieren
    if hasattr(attention, 'cpu'):
        attention = attention.cpu().numpy()

    # attention shape: (n_samples, prediction_length, encoder_length)
    if attention.ndim == 3:
        # Durchschnitt über Samples
        avg_attention = attention.mean(axis=0)
    elif attention.ndim == 2:
        avg_attention = attention
    else:
        print(f"[WARNUNG] Unerwartete Attention Shape: {attention.shape}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(avg_attention, aspect='auto', cmap='Blues')

    ax.set_xlabel("Encoder Position (vergangene Zeitschritte)")
    ax.set_ylabel("Decoder Position (Vorhersagehorizont)")
    ax.set_title(f"Attention Weights | Run: {run_id}\n(Welche vergangenen Zeitpunkte beeinflussen die Vorhersage?)")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight")

    fig.tight_layout()

    output_path = output_dir / f"{run_id}_attention_heatmap.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Attention Heatmap: {output_path}")
    plt.close(fig)


def _plot_attention_summary(
    interpretation: Dict[str, Any],
    output_dir: Path,
    run_id: str,
    encoder_length: int,
) -> None:
    """
    Plottet einen aggregierten Attention-Verlauf:
    Zeigt für jeden vergangenen Zeitpunkt die durchschnittliche Attention.
    """
    if "attention" not in interpretation:
        return

    attention = interpretation["attention"]
    # Tensor zu numpy konvertieren
    if hasattr(attention, 'cpu'):
        attention = attention.cpu().numpy()

    if attention.ndim == 3:
        # Durchschnitt über Samples und Prediction Steps
        avg_attention = attention.mean(axis=(0, 1))
    elif attention.ndim == 2:
        avg_attention = attention.mean(axis=0)
    else:
        return

    fig, ax = plt.subplots(figsize=(10, 4))

    x_labels = [f"t-{encoder_length - i}" for i in range(len(avg_attention))]

    ax.bar(range(len(avg_attention)), avg_attention, color='steelblue')
    ax.set_xticks(range(0, len(avg_attention), max(1, len(avg_attention) // 10)))
    ax.set_xticklabels([x_labels[i] for i in range(0, len(avg_attention), max(1, len(avg_attention) // 10))],
                       rotation=45, ha='right')
    ax.set_xlabel("Vergangener Zeitpunkt")
    ax.set_ylabel("Durchschnittliche Attention")
    ax.set_title(f"Attention über Zeit | Run: {run_id}\n(Welche vergangenen Zeitpunkte sind im Durchschnitt am wichtigsten?)")

    fig.tight_layout()

    output_path = output_dir / f"{run_id}_attention_summary.png"
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Attention Summary: {output_path}")
    plt.close(fig)


def _save_interpretation_json(
    interpretation: Dict[str, Any],
    model: TemporalFusionTransformer,
    output_dir: Path,
    run_id: str,
) -> None:
    """Speichert die Interpretation als JSON für weitere Analyse."""
    result = {
        "run_id": run_id,
        "encoder_variables": {},
        "decoder_variables": {},
        "static_variables": {},
    }

    # Encoder Variables
    if "encoder_variables" in interpretation:
        encoder_imp = interpretation["encoder_variables"]
        if hasattr(encoder_imp, 'cpu'):
            encoder_imp = encoder_imp.cpu().numpy()
        encoder_names = model.encoder_variables
        if len(encoder_imp) == len(encoder_names):
            for name, imp in zip(encoder_names, encoder_imp):
                result["encoder_variables"][name] = float(imp)

    # Decoder Variables
    if "decoder_variables" in interpretation:
        decoder_imp = interpretation["decoder_variables"]
        if hasattr(decoder_imp, 'cpu'):
            decoder_imp = decoder_imp.cpu().numpy()
        decoder_names = model.decoder_variables
        if len(decoder_imp) == len(decoder_names):
            for name, imp in zip(decoder_names, decoder_imp):
                result["decoder_variables"][name] = float(imp)

    # Static Variables
    if "static_variables" in interpretation:
        static_imp = interpretation["static_variables"]
        if hasattr(static_imp, 'cpu'):
            static_imp = static_imp.cpu().numpy()
        static_names = model.static_variables
        if len(static_imp) == len(static_names):
            for name, imp in zip(static_names, static_imp):
                result["static_variables"][name] = float(imp)

    # Sortiert nach Importance ausgeben
    result["encoder_variables"] = dict(
        sorted(result["encoder_variables"].items(), key=lambda x: x[1], reverse=True)
    )
    result["decoder_variables"] = dict(
        sorted(result["decoder_variables"].items(), key=lambda x: x[1], reverse=True)
    )
    result["static_variables"] = dict(
        sorted(result["static_variables"].items(), key=lambda x: x[1], reverse=True)
    )

    output_path = output_dir / f"{run_id}_interpretation.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"✓ Interpretation JSON: {output_path}")

    # Kurzübersicht auf Konsole (Top 10)
    print("\n" + "=" * 50)
    print("FEATURE IMPORTANCE (Top 10)")
    print("=" * 50)

    print("\n📊 ENCODER (Historie):")
    for i, (name, imp) in enumerate(result["encoder_variables"].items()):
        if i >= 10:
            break
        print(f"  {i+1:2}. {name:<25} {imp:.2f}")

    print("\n📊 DECODER (Zukunft):")
    for i, (name, imp) in enumerate(result["decoder_variables"].items()):
        if i >= 10:
            break
        print(f"  {i+1:2}. {name:<25} {imp:.2f}")

    if result["static_variables"]:
        print("\n📊 STATIC (Gruppen):")
        for i, (name, imp) in enumerate(result["static_variables"].items()):
            print(f"  {i+1:2}. {name:<25} {imp:.2f}")

    print("=" * 50)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint nicht gefunden: {ckpt_path}")
        run_id = ckpt_path.parent.parent.name
    else:
        run_id = args.run_id
        ckpt_path = _find_checkpoint(run_id)

    split = args.split

    print(f"[plot_tft_interpretation] Checkpoint: {ckpt_path}")
    print(f"[plot_tft_interpretation] Split: {split}")

    # Modell laden
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))
    model.eval()

    # Daten laden
    df_split = _load_split(split)
    print(f"[plot_tft_interpretation] Geladene Daten: {len(df_split):,} Zeilen")

    # Dataset erstellen
    dataset = _create_dataset_for_interpretation(model, df_split)
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    # Interpretation
    print("[plot_tft_interpretation] Berechne Feature Importance...")
    interpretation = _get_interpretation(model, dataloader)

    # Output-Verzeichnis
    output_dir = BASE_DIR / "results" / "tft" / "plots" / "interpretation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plots erstellen
    _plot_variable_importance(interpretation, model, output_dir, run_id)
    _plot_static_variables(interpretation, model, output_dir, run_id)
    _plot_attention_weights(interpretation, output_dir, run_id)

    encoder_length = model.hparams.dataset_parameters.get('max_encoder_length', 16)
    _plot_attention_summary(interpretation, output_dir, run_id, encoder_length)

    # JSON speichern
    _save_interpretation_json(interpretation, model, output_dir, run_id)

    print(f"\n[plot_tft_interpretation] Alle Plots gespeichert in: {output_dir}")


if __name__ == "__main__":
    main()

# Aufruf:
#   Mit run-id:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.visualization.plot_tft_interpretation --run-id run_20251125_003840_booksales_optuna_tft_day_best --split test
#
#   Mit checkpoint (für Optuna-Trials):
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.visualization.plot_tft_interpretation --checkpoint "results\tft\optuna\walmart\trial_0020\checkpoints\tft-epoch=08-val_loss=790.6539.ckpt" --split test