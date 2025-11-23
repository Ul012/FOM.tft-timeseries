# Projektstruktur – TFT-TimeSeries

**Datum:** 2025-11-23 (aktualisiert)  
**Script:** –  
**Ziel & Inhalt:** Vollständige Übersicht über die Projektstruktur. Erklärt Ordner-Rollen, Datenfluss, Zuständigkeiten und Erweiterbarkeit.

---

## 🗂️ 1. `src/` – Hauptverzeichnis

```text
src/
├── data/           # Preprocessing
├── modeling/       # Training
├── evaluation/     # Bewertung
├── utils/          # Hilfsfunktionen
├── visualization/  # Plots
├── config.py       # Globale Konstanten (DEPRECATED – wird durch YAMLs ersetzt)
└── pipeline.py     # Orchestrierung
```

---

## 📊 2. `data/` – Datenaufbereitung

| Datei | Aufgabe |
|-------|---------|
| `load_raw.py` | Laden und Mergen von CSV-Dateien |
| `data_alignment.py` | Harmonisierung und Normalisierung |
| `data_cleaning.py` | Bereinigung, Outlier-Behandlung |
| `feature_engineering.py` | Kalender- und Feiertags-Features, time_idx |
| `cyclical_encoder.py` | Zyklische Kodierung (sin/cos) |
| `lag_features.py` | Lag- und Rolling-Features |
| `analyze_dataset.py` **(NEU)** | Automatische Datensatz-Analyse + YAML-Generierung |

**Output:** `data/processed/<dataset_name>/train_features_cyc_lag.parquet`

---

## 🤖 3. `modeling/` – Training

| Datei | Aufgabe |
|-------|---------|
| `model_dataset.py` | Split in Train/Val/Test |
| `dataset_tft.py` | TFT-Datensatz-Spezifikation + Dataset-spezifische Anpassungen |
| `trainer_tft.py` | TFT-Training mit PyTorch Lightning |
| *(geplant)* `trainer_arima.py` | ARIMA-Training |
| *(geplant)* `trainer_prophet.py` | Prophet-Training |

### Datenfluss:

```
model_dataset.py
  Eingabe: data/processed/<dataset_name>/train_features_cyc_lag.parquet
  Ausgabe: data/processed/<dataset_name>/{train,val,test}.parquet + meta.json

dataset_tft.py
  Eingabe: data/processed/<dataset_name>/{train,val,test}.parquet
  Funktionen:
    - Imputing (NaN-Werte mit festen Werten füllen)
    - Spalten-Exclusion (dataset-spezifisch)
    - Feature-Listen ableiten (static/known/unknown)
  Ausgabe: data/processed/<dataset_name>/dataset_spec.json

trainer_tft.py
  Eingabe: 
    - dataset_spec.json
    - configs/models/tft/*.yaml
  Funktionen:
    - Target-Normalisierung (config-gesteuert: softplus/standard/relu/log)
    - Training mit Early Stopping
    - Checkpoint-Speicherung
  Ausgabe:
    - Logs: logs/tft/run_YYYYMMDD_HHMMSS_<dataset>_<config>/
    - Checkpoints: results/tft/runs/run_YYYYMMDD_HHMMSS_<dataset>_<config>/checkpoints/
    - JSONs: results/tft/runs/run_YYYYMMDD_HHMMSS_<dataset>_<config>/{results,summary}.json
```

### Neue Features in `dataset_tft.py`:

**1. Imputing (`impute_cols`):**
```yaml
# In configs/datasets/walmart.yaml
tft:
  impute_cols:
    MarkDown1: 0.0
    MarkDown2: 0.0
```
→ Füllt NaN-Werte mit festen Werten (z.B. 0 = keine Promotion)

**2. Spalten-Exclusion (`exclude_cols`):**
```yaml
# In configs/datasets/walmart.yaml
tft:
  exclude_cols: ["lag_365"]
```
→ Entfernt Spalten vor Feature-Ableitung (z.B. lag_365 nur bei täglichen Daten)

### Neue Features in `trainer_tft.py`:

**Target-Normalisierung (dataset-spezifisch):**
```yaml
# In configs/models/tft/walmart/baseline.yaml
model:
  target_normalizer_transformation: null  # Standard statt softplus
```
Optionen:
- `"softplus"` (Default) - Für positive Werte (Booksales)
- `null` - Standard z-score (Walmart)
- `"relu"` - Clippt negative auf 0
- `"log"` - Für log-normalverteilte Daten

---

## 4. `evaluation/` – Bewertung

| Datei | Aufgabe |
|-------|---------|
| `evaluate_tft.py` | Berechnet Fehlermaße für einen Run (MAE, RMSE, MAPE, SMAPE) |
| `aggregate_tft_eval.py` | Aggregiert alle Evaluierungen |

**Ausgabe:**
```
results/tft/eval/
├── <run_id>/
│   └── eval_summary.json
├── eval_overview.csv
└── eval_overview.json
```

---

## 📈 5. `visualization/` – Plots

| Datei | Aufgabe |
|-------|---------|
| `data_alignment_plot.py` | Visualisierung der Harmonisierung |
| `data_cleaning_plot.py` | Vorher/Nachher-Vergleich |
| `plot_learning_rate.py` | Lernkurven |
| `plot_tft_eval_comparison.py` | Run-Vergleiche |
| `plot_tft_forecast_series.py` | Forecast-Beispiele |

**Ausgabe:** `results/tft/plots/`

---

## 🧰 6. `utils/` – Hilfsfunktionen

| Datei | Aufgabe |
|-------|---------|
| `config_loader.py` | YAML-Validierung und Laden |
| `load_dataset_config.py` | Dataset-Config-Loader |
| `json_results.py` | Metriken-Export |
| `load_trained_tft.py` | Checkpoint-Loader |

**Wichtig:** Utils werden NICHT direkt aufgerufen, sondern von anderen Scripts importiert.

---

## 🔄 7. `pipeline.py` – Orchestrierung

**Hauptfunktion:** Orchestriert alle Schritte von Preprocessing bis Training.

**Verfügbare Steps:**
1. `preprocessing` - Alle aktivierten Preprocessing-Steps
2. `model_dataset` - Train/Val/Test Split
3. `dataset_tft` - TFT-Spezifikation erstellen
4. `training` - TFT-Training

**Aufruf-Beispiele:**
```bash
# Kompletter Run
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/booksales/baseline.yaml

# Nur Training (Preprocessing bereits erledigt)
python -m src.pipeline \
    --dataset configs/datasets/walmart.yaml \
    --model configs/models/tft/walmart/baseline.yaml \
    --steps training

# Nur Preprocessing
python -m src.pipeline \
    --dataset configs/datasets/walmart.yaml \
    --steps preprocessing,model_dataset,dataset_tft
```

**Ausgabe:** `results/pipeline_runs/pipeline_YYYYMMDD_HHMMSS_manifest.json`

---

## ⚙️ 8. `config.py` – Zentrale Steuerung (DEPRECATED)

**Status:** Wird schrittweise durch YAML-Configs ersetzt.

**Aktuell noch enthält:**
- Pfade: `RAW_DIR`, `INTERIM_DIR`, `PROCESSED_DIR`, `BASE_DIR`
- Wird von allen Scripts importiert für Pfad-Konstanten

**Nicht mehr verwendet für:**
- Schema (jetzt in `configs/datasets/*.yaml`)
- Split-Parameter (jetzt in `configs/datasets/*.yaml`)
- Feature-Configs (jetzt in `configs/datasets/*.yaml`)
- Trainings-Hyperparameter (immer schon in `configs/models/*.yaml`)

---

## 📁 9. `configs/` – Konfigurationen

```
configs/
├── datasets/              # Dataset-spezifische Configs
│   ├── booksales.yaml     # Täglich, 3 ID-Spalten, EU-Feiertage
│   └── walmart.yaml       # Wöchentlich, 2 ID-Spalten, US-Feiertage
│       
└── models/
    └── tft/
        ├── booksales/
        │   ├── baseline.yaml
        │   └── optuna_tft_day_best.yaml
        └── walmart/
            └── baseline.yaml
```

### Dataset-Config Struktur (Beispiel: walmart.yaml)

```yaml
name: "walmart"
description: "Walmart Store Sales - Weekly forecasting"

paths:
  raw: "data/raw/walmart"
  interim: "data/interim/walmart"
  processed: "data/processed/walmart"

raw_data:
  type: "multiple_files"  # oder "single_file"
  files:
    - path: "data/raw/walmart/train.csv"
      role: "main"
    - path: "data/raw/walmart/features.csv"
      role: "features"
  merge:
    merge_on: ["Store", "Date"]
    how: "left"

schema:
  time_col: "Date"
  id_cols: ["Store", "Dept"]
  target_col: "Weekly_Sales"

preprocessing:
  - step: "load_raw"
    enabled: true
  - step: "feature_engineering"
    enabled: true
    params:
      country: "US"
  # ... weitere Steps

split:
  method: "ratio"
  ratios: [0.80, 0.10, 0.10]

tft:
  max_encoder_length: 16
  max_prediction_length: 4
  known_real_prefixes: ["cyc_"]
  lag_prefixes: ["lag_"]
  treat_calendar_as_known: true
  flag_cols: []
  
  # Walmart-spezifische Anpassungen
  impute_cols:
    MarkDown1: 0.0
    MarkDown2: 0.0
    MarkDown3: 0.0
    MarkDown4: 0.0
    MarkDown5: 0.0
  exclude_cols: ["lag_365"]
```

### Model-Config Struktur (Beispiel: baseline.yaml)

```yaml
type: "tft"
name: "baseline"
description: "Stabile Referenz für TFT"

training:
  seed: 42
  max_epochs: 30
  batch_size: 256
  learning_rate: 0.001
  gradient_clip_val: 0.1
  early_stopping_patience: 5
  accelerator: "gpu"
  devices: 1
  num_workers: 4

model:
  # Walmart-spezifisch: robustere Normalisierung
  target_normalizer_transformation: null
  
  loss: "quantile"
  output_size: 7
  hidden_size: 32
  attention_head_size: 4
  hidden_continuous_size: 16
  dropout: 0.1
  reduce_on_plateau_patience: 2
```

---

## 📂 10. `results/` – Outputs

```
results/
├── pipeline_runs/           # Pipeline-Manifests (modellübergreifend)
│   └── pipeline_YYYYMMDD_HHMMSS_manifest.json
│
└── tft/
    ├── runs/
    │   └── run_YYYYMMDD_HHMMSS_<dataset>_<config>/
    │       ├── checkpoints/
    │       │   └── tft-epoch=XX-val_loss=Y.YYYY.ckpt
    │       ├── results.json      # Epochen-weise Metriken
    │       └── summary.json      # Aggregierte Summary
    │
    ├── eval/
    │   ├── <run_id>/
    │   │   └── eval_summary.json
    │   ├── eval_overview.csv     # Vergleich aller Runs
    │   └── eval_overview.json
    │
    └── plots/
        └── eval/
            └── compare_test_smape.png
```

---

## 📚 11. `docs/` – Dokumentation

```
docs/
├── Pipeline.md                      # Pipeline-Orchestrierung
├── PipelineOrder.md                 # Workflow-Übersicht
├── DatasetTFT.md                    # TFT-Datensatz-Spezifikation
├── TrainerTFT.md                    # TFT-Training
├── ConfigSetup.md                   # Config-System
├── Projektstruktur.md               # Diese Datei
├── MLFlowKonzept.md                 # Geplante MLflow-Integration
└── ArimaProphetIntegration.md       # Geplante Modell-Erweiterung
```

**Zugriff via MkDocs:**
```bash
mkdocs serve
# → http://localhost:8000
```

---

## ✅ 12. Erweiterbarkeit

### Neues Modell hinzufügen (z.B. ARIMA):

1. **Config erstellen:** `configs/models/arima/baseline.yaml`
   ```yaml
   type: "arima"
   name: "baseline"
   training:
     order: [1, 1, 1]
     seasonal_order: [1, 1, 1, 7]
   ```

2. **Trainer erstellen:** `src/modeling/trainer_arima.py`
   - Nutzt dieselben Splits (`train.parquet`, `val.parquet`, `test.parquet`)
   - Speichert in `results/arima/runs/`

3. **Pipeline erweitern:** Modelltyp-Erkennung in `src/pipeline.py`

4. **Evaluation:** `src/evaluation/evaluate_arima.py` analog zu TFT

### Neuer Datensatz hinzufügen:

**Option 1: Manuell**
1. Rohdaten in `data/raw/<name>/` legen
2. Config nach Vorlage erstellen: `configs/datasets/<name>.yaml`
3. Pipeline starten

**Option 2: Automatisch (EMPFOHLEN)**
1. `analyze_dataset.py` ausführen:
   ```bash
   python -m src.data.analyze_dataset --path data/raw/<name>/train.csv
   ```
2. Generierte `<name>_proposed.yaml` prüfen und anpassen
3. Pipeline starten

---

## 🎯 13. Workflow-Übersicht

```
Preprocessing → model_dataset → dataset_tft → trainer_tft → evaluate_tft
     ↓              ↓               ↓              ↓             ↓
   interim/      processed/      spec.json    checkpoints/  eval_summary.json
     
Parallel: analyze_dataset.py → proposed.yaml (neuer Datensatz)
```

**Ausführungsmodi:**
- ✅ Einzelne Scripts (für Debugging)
- ✅ Pipeline-Orchestrierung (für Production)
- ✅ Automatische Datensatz-Analyse (für neue Daten)

---

## 🔑 14. Wichtige Konzepte

### Multi-Dataset-Support
- **Aktuell:** Booksales (täglich), Walmart (wöchentlich)
- **Erweiterbar:** Beliebige neue Datensätze via `analyze_dataset.py`

### Dataset-spezifische Anpassungen
- **Imputing:** NaN-Werte mit festen Werten füllen (`impute_cols`)
- **Exclusion:** Spalten entfernen (`exclude_cols`)
- **Normalisierung:** Target-Transformation wählen (`target_normalizer_transformation`)

### Reproduzierbarkeit
- ✅ Alle Parameter in YAML
- ✅ Seeds gesetzt
- ✅ Pipeline-Manifests
- ✅ Summary-JSONs

### Code-Prinzipien
- ✅ Minimalinvasiv
- ✅ Config-gesteuert (keine Hardcoded-Werte)
- ✅ Modular & testbar
- ✅ Umfassend dokumentiert