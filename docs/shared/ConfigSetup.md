# Konfigurationen – Zusammenspiel von `config.py` und `configs/*.yaml`

**Datum:** 2025-11-21 (aktualisiert)  
**Script:** —  
**Ziel & Inhalt:** Beschreibt die Trennung zwischen statischen Projektkonstanten (`config.py`) und variablen Trainingsparametern in YAML-Dateien. Erläutert Pfade, Spalten, Feature-Konfigurationen, Split-Parameter sowie den Ablauf eines Trainingslaufs.

---

## Struktur und Zweck

- **`src/config.py`**  
  Enthält **statische Projektkonstanten**: Dateipfade, Spaltennamen, Split-Konfigurationen, Feature-Einstellungen (Lag-Konfigurationen, Sequenzlängen für TFT).  
  Diese Werte ändern sich selten und dienen als zentrale Referenz für die Daten- und Modellpipeline.

- **`configs/datasets/*.yaml`**  
  **Dataset-spezifische Konfigurationen**: Schema, Preprocessing-Pipeline, Split-Strategie.  
  Ermöglicht Multi-Dataset-Support ohne Code-Änderungen.

- **`configs/models/tft/*.yaml`**  
  **Modell- und Trainingsparameter**: Batch-Größe, Lernrate, Epochenzahl, Modellarchitektur.  
  Pro Experiment frei wählbar.

**Kurz:**
- `config.py` → Projekt- und Datenstruktur  
- `configs/datasets/` → Dataset-Definition  
- `configs/models/` → Trainingsverhalten

---

## 1. Rolle von `src/config.py`

### 1.1 Verzeichnisse und Pfade

```python
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
```

- `RAW_DIR` → Rohdaten  
- `INTERIM_DIR` → Zwischenergebnisse  
- `PROCESSED_DIR` → Modellfertige Daten

```python
FEATURES_TRAIN_PATH = PROCESSED_DIR / "train_features.parquet"
MODEL_INPUT_PATH = PROCESSED_DIR / "train_features_cyc_lag.parquet"
```

### 1.2 Schema / Spalten

```python
DATETIME_COLUMN = "date"
GROUP_COLS = ["country", "store", "product"]
TARGET_COL = "num_sold"

TIME_COL = DATETIME_COLUMN
ID_COLS = GROUP_COLS
```

Verwendet in: `model_dataset.py`, `dataset_tft.py`, `trainer_tft.py`

### 1.3 Split-Parameter

```python
VAL_START = None
TEST_START = None
SPLIT_RATIOS = (0.80, 0.10, 0.10)
```

- Wenn `VAL_START`/`TEST_START` gesetzt → feste Grenzen  
- Wenn `None` → automatisch nach `SPLIT_RATIOS`

### 1.4 Feature-Konfigurationen

```python
LAG_CONF = {
    "target_col": TARGET_COL,
    "lags": [1, 7, 14],
    "roll_windows": [7],
    "roll_stats": ["mean"],
    "prefix": "lag_",
}

TFT_DATASET = {
    "max_encoder_length": 120,
    "max_prediction_length": 7,
    "known_real_prefixes": ["cyc_"],
    "lag_prefixes": ["lag_"],
    "treat_calendar_as_known": True,
    "flag_cols": ["is_lockdown_period"],
}
```

---

## 2. Neue Config-Hierarchie (seit 2025-11-21)

### 2.1 Dataset-Config (`configs/datasets/booksales.yaml`)

```yaml
name: "booksales"
paths:
  raw: "data/raw"
  interim: "data/interim"
  processed: "data/processed"

schema:
  time_col: "date"
  id_cols: ["country", "store", "product"]
  target_col: "num_sold"

preprocessing:
  - step: "alignment"
    enabled: true
  - step: "cleaning"
    enabled: true
  # ... weitere Steps

split:
  method: "ratio"
  ratios: [0.80, 0.10, 0.10]

tft:
  max_encoder_length: 120
  max_prediction_length: 7
```

### 2.2 Model-Config (`configs/models/tft/baseline.yaml`)

```yaml
type: "tft"
name: "baseline_v02"

training:
  seed: 42
  max_epochs: 30
  batch_size: 128
  learning_rate: 0.001
  # ...

model:
  loss: "quantile"
  hidden_size: 32
  attention_head_size: 4
  # ...
```

---

## 3. Aufruf-Beispiele

### Via Pipeline (empfohlen):

```bash
# Kompletter Run
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/baseline.yaml

# Nur Training
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/baseline.yaml \
    --steps training
```

### Einzeln (für Tests):

```bash
python -m src.modeling.trainer_tft \
    --config configs/models/tft/baseline.yaml
```

---

## 4. Best Practices

- Jedes Experiment = eigene YAML in `configs/models/tft/experiments/`
- Funktionale Namen: `lr_high.yaml`, `bs_small.yaml`, `model_large.yaml`
- Keine Änderungen in `config.py` für Experimente
- YAML-Dateien versionieren (Git)

---

## 5. Migration von alten Configs

**Alt:**
```
configs/trainer_tft_baseline02.yaml
```

**Neu:**
```
configs/models/tft/baseline.yaml
```

Alte Configs funktionieren weiter, sind aber deprecated.

---

Diese Struktur ermöglicht:
- ✅ Multi-Dataset-Support
- ✅ Klare Trennung Dataset ↔ Modell
- ✅ Reproduzierbare Experimente
- ✅ Einfache Erweiterung (ARIMA, Prophet)