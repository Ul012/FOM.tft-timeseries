# src/utils/load_trained_tft.py
"""
Utility zum Laden trainierter TFT-Modelle aus Checkpoints.

Beispiel-Aufruf:
    python -m src.utils.load_trained_tft

Hinweis: Wird selten direkt aufgerufen, meist via evaluate_tft.py
"""

from pathlib import Path
from pytorch_forecasting.models import TemporalFusionTransformer
import torch

from src.config import BASE_DIR


def load_trained_model(checkpoint_path: str | Path) -> TemporalFusionTransformer:
    """
    Lädt ein TFT-Modell aus einem gespeicherten Checkpoint (.ckpt-Datei).

    Args:
        checkpoint_path (str | Path): Pfad zur .ckpt-Datei.

    Returns:
        TemporalFusionTransformer: Geladenes Modellobjekt im eval-Modus.
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint-Datei nicht gefunden: {ckpt_path}")

    print(f"Lade TFT-Modell aus Checkpoint:\n  {ckpt_path}")

    # Modell laden (Lightning kümmert sich um alle internen Objekte)
    model = TemporalFusionTransformer.load_from_checkpoint(ckpt_path)
    model.eval()  # Modell in den Evaluierungsmodus (keine Gradienten, Dropout deaktiviert)

    print("Modell erfolgreich geladen.")
    print(f"Modellname: {model.__class__.__name__}")
    print(f"Geräteverwendung: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Anzahl Parameter: {sum(p.numel() for p in model.parameters()):,}")

    return model


def main():
    """
    Beispielhafte Nutzung: Lädt das zuletzt gespeicherte Checkpoint des TFT-Modells.
    """
    # Suche neuestes Checkpoint in results/tft/runs/
    runs_dir = BASE_DIR / "results" / "tft" / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Keine Runs gefunden in: {runs_dir}")

    # Finde alle Checkpoints in allen Run-Ordnern
    checkpoints = sorted(runs_dir.glob("*/checkpoints/*.ckpt"),
                         key=lambda p: p.stat().st_mtime,
                         reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"Keine Checkpoints gefunden in: {runs_dir}")

    latest_ckpt = checkpoints[0]
    model = load_trained_model(latest_ckpt)

    # Beispiel für spätere Nutzung:
    # predictions = model.predict(dataloader)
    # print(predictions[:5])

    print("Modell ist bereit für Inferenz oder Evaluierung.")


if __name__ == "__main__":
    main()

# Aufruf (selten direkt genutzt):
#   python -m src.utils.load_trained_tft
#
# Häufiger via evaluate_tft.py:
#   python -m src.evaluation.evaluate_tft --run-id <run_id>