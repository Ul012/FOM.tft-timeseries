# src/modeling/load_trained_tft.py
"""
Lädt ein bereits trainiertes Temporal Fusion Transformer (TFT) Modell aus einem
Checkpoint (.ckpt) und prüft, ob es erfolgreich geladen wurde.

Beispiel-Aufruf:
    python -m src.modeling.load_trained_tft
"""

from pathlib import Path
from pytorch_forecasting.models import TemporalFusionTransformer
import torch


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
    ckpt_dir = Path("data/processed/model_dataset/tft/checkpoints")
    checkpoints = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"Keine Checkpoints gefunden in: {ckpt_dir}")

    latest_ckpt = checkpoints[0]
    model = load_trained_model(latest_ckpt)

    # Beispiel für spätere Nutzung:
    # predictions = model.predict(dataloader)
    # print(predictions[:5])

    print("Modell ist bereit für Inferenz oder Evaluierung.")


if __name__ == "__main__":
    main()

# python -m src.modeling.load_trained_tft