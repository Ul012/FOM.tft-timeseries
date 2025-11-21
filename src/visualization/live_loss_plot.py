# src/visualization/live_loss_plot.py
#
# Liest einmalig logs/.../<run_id>/metrics.csv,
# plottet train_loss und val_loss über die Epochen,
# öffnet ein Plot-Fenster und speichert den Plot still als PNG:
#   results/tft/plots/eval/<run_id>_live.png
#
# Aufrufbeispiel:
#   python -m src.visualization.live_loss_plot --run_dir logs/tft/run_20251117_232558_lr001_mel120

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.config import BASE_DIR


def load_epoch_metrics(metrics_path: Path) -> pd.DataFrame:
    """Lädt metrics.csv und aggregiert auf Epoche (letzter Eintrag pro Epoche)."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv nicht gefunden: {metrics_path}")

    df = pd.read_csv(metrics_path)
    df = df.dropna(how="all")

    if df.empty:
        raise ValueError(f"metrics.csv enthält keine verwertbaren Daten: {metrics_path}")

    if "epoch" not in df.columns:
        # Fallback: einfach durchnummerieren
        df["epoch"] = range(len(df))

    # letzte Zeile je Epoche (robust ggü. Step-Logs)
    df_epoch = df.groupby("epoch").last().reset_index()
    return df_epoch


def detect_loss_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Versucht, Spalten für Trainings- und Validierungsloss zu finden."""
    train_candidates = ["train_loss_epoch", "train_loss_step", "train_loss", "loss"]
    val_candidates = ["val_loss", "val_loss_epoch"]

    train_col = next((c for c in train_candidates if c in df.columns), None)
    val_col = next((c for c in val_candidates if c in df.columns), None)

    if train_col is None and val_col is None:
        raise KeyError(
            f"Keine passenden Loss-Spalten gefunden. Verfügbare Spalten: {list(df.columns)}"
        )

    return train_col, val_col


def plot_loss_for_run(run_dir: Path) -> Path:
    """Erstellt den Loss-Plot für einen Run und speichert ihn als PNG."""
    metrics_path = run_dir / "metrics.csv"
    df_epoch = load_epoch_metrics(metrics_path)
    train_col, val_col = detect_loss_cols(df_epoch)

    run_id = run_dir.name

    # Output in training/ statt eval/
    output_dir = BASE_DIR / "results" / "tft" / "plots" / "training"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}_loss.png"

    fig, ax = plt.subplots(figsize=(10, 6))

    if train_col is not None:
        ax.plot(
            df_epoch["epoch"],
            df_epoch[train_col],
            marker="o",
            label="train_loss",
        )

    if val_col is not None:
        ax.plot(
            df_epoch["epoch"],
            df_epoch[val_col],
            marker="o",
            label="val_loss",
        )

    ax.set_title(f"Loss-Verlauf pro Epoche – Run {run_id}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    # Speichern
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot gespeichert: {output_path}")

    # Anzeigen
    plt.show()
    plt.close(fig)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Erzeuge einen einmaligen Loss-Plot (train/val) aus metrics.csv für einen Run."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Pfad zum Run-Ordner (enthält metrics.csv), z. B. logs/tft/run_20251117_232558_lr001_mel120",
    )

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run-Ordner nicht gefunden: {run_dir}")

    plot_loss_for_run(run_dir=run_dir)


if __name__ == "__main__":
    main()

# Aufruf (nach Training):
#   python -m src.visualization.live_loss_plot --run_dir logs/tft/run_20251121_123456_baseline
#   python -m src.visualization.live_loss_plot --run_dir logs/tft/run_20251121_150832_bs_small
#   python -m src.visualization.live_loss_plot --run_dir logs/tft/run_20251121_222313_lr_high
#
# Output: results/tft/plots/training/<run_id>_loss.png


