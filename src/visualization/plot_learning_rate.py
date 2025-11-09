# src/visualization/plot_learning_rate.py
"""
Visualisiert Trainings- und Validierungs-Loss über die Epochen
und zeigt die Learning Rate auf separater Achse.
"""

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import yaml


def load_cfg() -> dict:
    """Lädt die Basiskonfiguration aus configs/trainer_tft_baseline.yaml (falls vorhanden)."""
    configs_dir = Path("configs")
    yaml_files = list(configs_dir.glob("trainer_tft*.yaml"))
    if not yaml_files:
        return {}
    with open(yaml_files[0], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        help="Pfad zum Run-Ordner mit metrics.csv, z. B. logs/tft/run_20251109_120713",
    )
    args = parser.parse_args()

    csv_path = Path(args.run) / "metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics.csv nicht gefunden: {csv_path}")

    # CSV laden und vorbereiten
    df = pd.read_csv(csv_path)
    df = df.dropna(how="all", axis=0).reset_index(drop=True)
    if "epoch" not in df.columns:
        df["epoch"] = range(len(df))

    # Aggregation pro Epoche
    df_epoch = df.groupby("epoch").last().reset_index()

    # Plot vorbereiten
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Train-Loss
    if "train_loss_epoch" in df_epoch.columns:
        ax1.plot(df_epoch["epoch"], df_epoch["train_loss_epoch"], label="Train Loss", color="tab:blue")
    elif "train_loss_step" in df_epoch.columns:
        ax1.plot(df_epoch["epoch"], df_epoch["train_loss_step"], label="Train Loss", color="tab:blue")

    # Val-Loss
    if "val_loss" in df_epoch.columns:
        ax1.plot(df_epoch["epoch"], df_epoch["val_loss"], label="Val Loss", color="tab:red", alpha=0.8)

    ax1.set_xlabel("Epoche")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)

    # Learning Rate (rechte Achse)
    lr_cols = [c for c in df.columns if ("lr" in c.lower()) or ("learning_rate" in c.lower())]
    if not lr_cols:
        raise RuntimeError("Learning Rate-Spalte nicht gefunden. Ist LearningRateMonitor aktiviert?")
    lr_col = lr_cols[0]

    df_lr = df.groupby("epoch")[[lr_col]].last().reset_index()
    ax2 = ax1.twinx()
    ax2.plot(df_lr["epoch"], df_lr[lr_col], color="gray", linestyle="--", label="Learning Rate")
    ax2.set_ylabel("Learning Rate", color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")

    plt.title("Train/Val-Loss und Learning Rate über die Epochen")

    # Parameter unten rechts einfügen
    cfg = load_cfg()
    if cfg:
        param_text = (
            f"batch_size={cfg.get('batch_size', '?')} | "
            f"lr={cfg.get('learning_rate', '?')} | "
            f"max_epochs={cfg.get('max_epochs', '?')} | "
            f"patience={cfg.get('early_stopping_patience', '?')}"
        )
        plt.text(
            0.99,
            -0.22,
            param_text,
            ha="right",
            va="top",
            fontsize=9,
            color="gray",
            transform=ax1.transAxes,
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    # Ergebnis speichern
    run_name = Path(args.run).name
    plots_dir = Path("results") / "plots" / run_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    save_path = plots_dir / f"learning_curve_{run_name}_{timestamp}.png"

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"[plot_learning_rate] Plot gespeichert unter: {save_path}")


if __name__ == "__main__":
    main()




# python -m src.visualization.plot_learning_rate --run logs/tft/run_20251108_210539