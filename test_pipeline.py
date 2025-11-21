#!/usr/bin/env python3
# test_pipeline.py
"""
Quick-Test für die neue Pipeline-Struktur.

Testet ob:
1. Configs korrekt geladen werden
2. Pipeline-Script importierbar ist
3. Alle benötigten Module existieren

Aufruf:
    python test_pipeline.py
"""

from pathlib import Path
import sys
import yaml


def test_configs():
    """Prüft ob Config-Dateien existieren und valide sind."""
    print("\n" + "=" * 70)
    print("TEST 1: Config-Dateien")
    print("=" * 70)

    # Dataset-Config
    dataset_path = Path("configs/datasets/booksales.yaml")
    if not dataset_path.exists():
        print(f"❌ FEHLER: {dataset_path} nicht gefunden")
        return False

    dataset_cfg = yaml.safe_load(dataset_path.read_text())
    required_keys = ["name", "schema", "preprocessing", "split"]
    for key in required_keys:
        if key not in dataset_cfg:
            print(f"❌ FEHLER: Dataset-Config fehlt Key '{key}'")
            return False

    print(f"✓ Dataset-Config OK: {dataset_cfg['name']}")

    # Model-Config
    model_path = Path("configs/models/tft/baseline.yaml")
    if not model_path.exists():
        print(f"❌ FEHLER: {model_path} nicht gefunden")
        return False

    model_cfg = yaml.safe_load(model_path.read_text())
    required_keys = ["type", "name", "training", "model"]
    for key in required_keys:
        if key not in model_cfg:
            print(f"❌ FEHLER: Model-Config fehlt Key '{key}'")
            return False

    print(f"✓ Model-Config OK: {model_cfg['type']} / {model_cfg['name']}")

    return True


def test_pipeline_import():
    """Prüft ob Pipeline-Modul importierbar ist."""
    print("\n" + "=" * 70)
    print("TEST 2: Pipeline-Import")
    print("=" * 70)

    try:
        from src import pipeline
        print("✓ Pipeline-Modul importierbar")
        return True
    except ImportError as e:
        print(f"❌ FEHLER: Pipeline nicht importierbar: {e}")
        return False


def test_preprocessing_modules():
    """Prüft ob alle Preprocessing-Module existieren."""
    print("\n" + "=" * 70)
    print("TEST 3: Preprocessing-Module")
    print("=" * 70)

    modules = [
        "src.data.data_alignment",
        "src.data.data_cleaning",
        "src.data.feature_engineering",
        "src.data.cyclical_encoder",
        "src.data.lag_features",
    ]

    all_ok = True
    for module_name in modules:
        module_path = Path(module_name.replace(".", "/") + ".py")
        if not module_path.exists():
            print(f"❌ FEHLER: {module_path} nicht gefunden")
            all_ok = False
        else:
            print(f"✓ {module_path}")

    return all_ok


def test_modeling_modules():
    """Prüft ob alle Modeling-Module existieren."""
    print("\n" + "=" * 70)
    print("TEST 4: Modeling-Module")
    print("=" * 70)

    modules = [
        "src.modeling.model_dataset",
        "src.modeling.dataset_tft",
        "src.modeling.trainer_tft",
    ]

    all_ok = True
    for module_name in modules:
        module_path = Path(module_name.replace(".", "/") + ".py")
        if not module_path.exists():
            print(f"❌ FEHLER: {module_path} nicht gefunden")
            all_ok = False
        else:
            print(f"✓ {module_path}")

    return all_ok


def test_directory_structure():
    """Prüft ob wichtige Verzeichnisse existieren."""
    print("\n" + "=" * 70)
    print("TEST 5: Verzeichnis-Struktur")
    print("=" * 70)

    dirs = [
        "configs/datasets",
        "configs/models/tft",
        "data/raw",
        "data/interim",
        "data/processed",
        "src/data",
        "src/modeling",
        "src/utils",
    ]

    all_ok = True
    for dir_path in dirs:
        p = Path(dir_path)
        if not p.exists():
            print(f"❌ FEHLER: {dir_path}/ nicht gefunden")
            all_ok = False
        else:
            print(f"✓ {dir_path}/")

    return all_ok


def main():
    print("\n" + "#" * 70)
    print("# Pipeline Quick-Test")
    print("#" * 70)

    results = {
        "Configs": test_configs(),
        "Pipeline-Import": test_pipeline_import(),
        "Preprocessing-Module": test_preprocessing_modules(),
        "Modeling-Module": test_modeling_modules(),
        "Verzeichnisse": test_directory_structure(),
    }

    # Summary
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {test_name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ Alle Tests bestanden!")
        print("\nNächste Schritte:")
        print("1. Stelle sicher dass Rohdaten in data/raw/ liegen")
        print("2. Teste die Pipeline:")
        print("   python -m src.pipeline --dataset configs/datasets/booksales.yaml \\")
        print("                          --model configs/models/tft/baseline.yaml \\")
        print("                          --steps preprocessing")
        return 0
    else:
        print("\n❌ Einige Tests fehlgeschlagen - siehe Details oben")
        return 1


if __name__ == "__main__":
    sys.exit(main())