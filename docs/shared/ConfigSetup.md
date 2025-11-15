# Konfigurationen – Zusammenspiel von `config.py` und `configs/*.yaml`

Dieses Dokument beschreibt die Trennung zwischen **statischen Projektkonstanten** in `src/config.py` und **variablen Trainingsparametern** in den YAML-Konfigurationen unter `configs/`.  
Die klare Aufgabenteilung stellt sicher, dass die Pipeline reproduzierbar, konfigurierbar und schlank bleibt.

---

## Struktur und Zweck

- **`src/config.py`**  
  Enthält **statische Projektkonstanten**, darunter Dateipfade, Spaltennamen, Split-Konfigurationen und Feature-bezogene Einstellungen (Lag-Konfigurationen, Sequenzlängen für TFT-Datasets).  
  Diese Werte ändern sich selten und dienen als zentrale Referenz für die Daten- und Modellpipeline.

- **`configs/*.yaml`**  
  Enthalten **variierende Trainings- und Modellparameter**, die pro Experiment frei gewählt werden können.  
  Dazu zählen Batch-Größe, Lernrate, Epochenzahl oder Modellgrößen.

Kurz zusammengefasst:

- `config.py` definiert die **Projekt- und Datenstruktur**,  
- `configs/*.yaml` definieren das **Trainingsverhalten eines konkreten Laufs**.

---

## 1. Rolle von `src/config.py`

### 1.1 Verzeichnisse und Pfade (konkrete Inhalte aus `config.py`)

Die Datei `src/config.py` legt die grundlegende Verzeichnisstruktur des Projekts fest:

```python
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
```

- `RAW_DIR` enthält unveränderte Rohdaten.  
- `INTERIM_DIR` speichert Zwischenergebnisse der Datenpipeline.  
- `PROCESSED_DIR` enthält alle modellfertigen Daten (Splits, Specs, Features).

Weitere Pfade für Feature- und Dataset-Erzeugung:

```python
FEATURES_TRAIN_PATH = PROCESSED_DIR / "train_features.parquet"
MODEL_INPUT_PATH = PROCESSED_DIR / "train_features_cyc_lag.parquet"
```

- `FEATURES_TRAIN_PATH` ist ein optionaler Zwischenschritt.  
- `MODEL_INPUT_PATH` ist der verbindliche Input für `model_dataset.py`.

Diese Pfade steuern deterministisch die Übergabe zwischen den Modulen der Datenpipeline.

---

### 1.2 Schema / Spalten

Alle spaltenbezogenen Konstanten sind in `config.py` definiert:

```python
DATETIME_COLUMN = "date"
GROUP_COLS = ["country", "store", "product"]
TARGET_COL = "num_sold"

TIME_COL = DATETIME_COLUMN
ID_COLS = GROUP_COLS
```

- `TIME_COL` definiert die Zeitachse.  
- `ID_COLS` definieren die Identität der Zeitreihen.  
- `TARGET_COL` ist die Zielvariable des Forecasting-Problems.

Diese Werte werden in allen relevanten Modulen konsequent verwendet (`model_dataset.py`, `dataset_tft.py`, `trainer_tft.py`).

---

### 1.3 Split-Parameter

```python
VAL_START = None
TEST_START = None
SPLIT_RATIOS = (0.80, 0.10, 0.10)
```

Verwendung durch `model_dataset.py`:

- Wenn `VAL_START` und `TEST_START` gesetzt sind → feste zeitliche Grenzen für Validation und Test.  
- Sind beide `None`, erfolgt die zeitliche Aufteilung automatisch anhand der Ratios.

---

### 1.4 Feature-Konfigurationen (konkret aus config.py)

#### Lag-Konfiguration (`LAG_CONF`)

```python
LAG_CONF = {
    "target_col": TARGET_COL,
    "lags": [1, 7, 14],
    "roll_windows": [7],
    "roll_stats": ["mean"],
    "prefix": "lag_",
}
```

#### TFT-Dataset-Konfiguration (`TFT_DATASET`)

```python
TFT_DATASET = {
    "max_encoder_length": 28,
    "max_prediction_length": 7,
    "known_real_prefixes": ["cyc_"],
    "lag_prefixes": ["lag_"],
    "treat_calendar_as_known": True,
    "flag_cols": ["is_lockdown_period"],
}
```

Verwendung in `dataset_tft.py`:

- Ableitung von `time_varying_known_reals` und `time_varying_unknown_reals`  
- Steuerung der Heuristik über Prefixes und explizite Flag-Spalten  
- Festlegen der Encoder- und Prediction-Längen  

---

## 2. Rolle der Trainer-YAMLs in `configs/`

Die YAML-Konfigurationen definieren alle Trainingsparameter, die im Experiment variiert werden können.

Die Baseline-YAML lautet wie folgt:

```yaml
seed: 42
batch_size: 64
learning_rate: 0.001
max_epochs: 30
gradient_clip_val: 0.1
early_stopping_patience: 5

accelerator: "cpu"
devices: 1

limit_train_batches: 1.0
limit_val_batches: 1.0
num_workers: 4

model:
  loss: "quantile"
  output_size: 3
  hidden_size: 32
  attention_head_size: 4
  hidden_continuous_size: 16
  dropout: 0.1
  reduce_on_plateau_patience: 3
```

Diese Datei wird in `trainer_tft.py` ohne Fallbacks geladen und vollständig an das Modell und den Trainer weitergereicht.

Typische veränderliche Parameter:

- Lernrate  
- Batch-Größe  
- Anzahl der Trainings-Epochen  
- Modellgröße (z. B. Hidden Size)

---

## 3. Zusammenspiel im Trainingslauf

Ablauf eines Trainingslaufs:

```bash
python -m src.modeling.trainer_tft --config configs/trainer_tft_baseline.yaml
```

1. Laden der YAML-Konfiguration (Trainingseinstellungen)  
2. Laden statischer Konstanten aus `config.py` (Pfade, Spalten)  
3. Einlesen von `dataset_spec.json` aus `PROCESSED_DIR`  
4. Aufbau des TFT-Modells anhand der YAML  
5. Initialisierung des Lightning-Trainers  
6. Export von Trainingsartefakten:
   - Logs → `logs/tft/run_*`  
   - Checkpoints → `results/tft/checkpoints/run_*`  
   - Evaluation JSON → `results/evaluation/run_*`

Die Trennung zwischen Projektstruktur und Trainingsparametern sorgt für Konsistenz und Reproduzierbarkeit.

---

## 4. Best Practices für YAML-Konfigurationen

- Jede Variante als eigene YAML speichern.  
- Dateinamen funktional wählen, z. B.:
  - Konfiguration mit erhöhter Lernrate  
  - Konfiguration mit längerer Trainingsdauer  
  - Konfiguration mit größerer Modellarchitektur  
- Änderungen ausschließlich in YAML-Dateien vornehmen, nicht in `config.py`.  
- YAML-Dateien versionieren, um Trainingsläufe nachvollziehbar zu dokumentieren.

---

## 5. Aufrufschema

```bash
python -m src.modeling.trainer_tft --config configs/<datei>.yaml
```

Der Code bleibt unverändert, alle Varianten werden über YAML gesteuert.

---

Diese Struktur ermöglicht eine transparente, modulare und reproduzierbare Steuerung der gesamten Modeling-Pipeline.
