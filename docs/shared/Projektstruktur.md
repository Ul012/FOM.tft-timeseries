# Projektstruktur — TFT-TimeSeries

**Ziel & Inhalt:** Vollständige Übersicht über die Projektstruktur. Erklärt Ordner-Rollen, Datenfluss, Zuständigkeiten und Erweiterbarkeit.

---

## 🗂️ 1. `src/` — Hauptverzeichnis

```text
src/
├── data/           # Preprocessing
├── modeling/       # Training
├── evaluation/     # Metriken-Berechnung & Aggregation
├── visualization/  # Plots & Grafiken
├── utils/          # Hilfsfunktionen
├── config.py       # Globale Konstanten
└── pipeline.py     # Orchestrierung
```

---

## 📊 2. `data/` — Datenaufbereitung

| Datei | Aufgabe |
|-------|---------|
| `load_raw.py` | Laden und Mergen von CSV-Dateien |
| `data_alignment.py` | Harmonisierung und Normalisierung |
| `data_cleaning.py` | Bereinigung, Outlier-Behandlung |
| `feature_engineering.py` | Kalender- und Feiertags-Features, time_idx |
| `cyclical_encoder.py` | Zyklische Kodierung (sin/cos) |
| `lag_features.py` | Lag- und Rolling-Features |
| `analyze_dataset.py` | Automatische Datensatz-Analyse + YAML-Generierung |

**Output:** `data/processed/<dataset_name>/train_features_cyc_lag.parquet`

---

## 🤖 3. `modeling/` — Training

| Datei | Aufgabe |
|-------|---------|
| `model_dataset.py` | Split in Train/Val/Test |
| `dataset_tft.py` | TFT-Datensatz-Spezifikation + Dataset-spezifische Anpassungen |
| `trainer_tft.py` | TFT-Training mit PyTorch Lightning |
| `dataset_arima.py` | ARIMA-Datensatz-Vorbereitung |
| `trainer_arima.py` | ARIMA-Training |
| `dataset_prophet.py` | Prophet-Datensatz-Vorbereitung |
| `trainer_prophet.py` | Prophet-Training |
| `optuna_*.py` | Hyperparameter-Optimierung |

### Datenfluss:

```
model_dataset.py
  Eingabe: data/processed/<dataset_name>/train_features_cyc_lag.parquet
  Ausgabe: data/processed/<dataset_name>/{train,val,test}.parquet + meta.json

dataset_<model>.py
  Eingabe: data/processed/<dataset_name>/{train,val,test}.parquet
  Funktionen:
    - Imputing (NaN-Werte mit festen Werten füllen)
    - Spalten-Exclusion (dataset-spezifisch)
    - Feature-Listen ableiten (static/known/unknown)
  Ausgabe: data/processed/<dataset_name>/dataset_spec.json

trainer_<model>.py
  Eingabe: 
    - dataset_spec.json
    - configs/models/<model>/*.yaml
  Funktionen:
    - Target-Normalisierung (config-gesteuert)
    - Training mit Early Stopping
    - Checkpoint-Speicherung
  Ausgabe:
    - Logs: logs/<model>/run_YYYYMMDD_HHMMSS_<dataset>_<config>/
    - Checkpoints: results/<model>/runs/run_YYYYMMDD_HHMMSS_<dataset>_<config>/checkpoints/
    - JSONs: results/<model>/runs/run_YYYYMMDD_HHMMSS_<dataset>_<config>/{results,summary}.json
```

---

## 4. `evaluation/` — Metriken-Berechnung & Aggregation

| Datei | Aufgabe |
|-------|---------|
| `evaluate_tft.py` | Berechnet Fehlermaße für TFT-Runs (MAE, RMSE, MAPE, SMAPE, R²) |
| `evaluate_prophet.py` | Berechnet Fehlermaße für Prophet-Runs |
| `evaluate_arima.py` | Berechnet Fehlermaße für ARIMA-Runs |
| `aggregate_tft_eval.py` | Aggregiert alle TFT-Evaluierungen |
| `aggregate_prophet_eval.py` | Aggregiert alle Prophet-Evaluierungen |
| `aggregate_arima_eval.py` | Aggregiert alle ARIMA-Evaluierungen |
| `aggregate_all_models_eval.py` | Kombiniert alle Modelle zu Master-Tabelle |
| `analyze_optuna_*_trials.py` | Statistische Analyse der Optuna-Trials |

**Ausgabe:**
```
results/<model>/
├── runs/<run_id>/
│   ├── eval_val.json
│   └── eval_test.json
├── eval_overview.csv       # Pro Modell
└── eval_overview.json

results/eval/
├── model_comparison.csv    # Alle Modelle kombiniert
└── model_comparison.json
```

---

## 📈 5. `visualization/` — Plots & Grafiken

### Übergreifende Plots (Cross-Model)

| Datei | Aufgabe | Output |
|-------|---------|--------|
| `plot_cross_model_comparison.py` | Best Run Vergleich (TFT vs Prophet vs ARIMA) | `results/plots/` |
| `plot_multi_model_forecast.py` | Multi-Model Forecast Visualisierung | `results/plots/` |
| `plot_baseline_vs_optuna.py` | Optuna-Verbesserung über alle Modelle | `results/plots/` |

### Modell-spezifische Plots

| Datei | Aufgabe | Output |
|-------|---------|--------|
| `plot_tft_forecast_series.py` | TFT Forecast-Beispiele | `results/tft/plots/` |
| `plot_tft_eval_comparison.py` | TFT Run-Vergleiche | `results/tft/plots/` |
| `plot_tft_optuna_study.py` | TFT Optuna-Visualisierung | `results/tft/plots/` |
| `plot_prophet_optuna_study.py` | Prophet Optuna-Visualisierung | `results/prophet/plots/` |
| `plot_arima_optuna_study.py` | ARIMA Optuna-Visualisierung | `results/arima/plots/` |

### Data Exploration Plots

| Datei | Aufgabe | Output |
|-------|---------|--------|
| `plot_data_alignment.py` | Visualisierung der Harmonisierung | `results/plots/` |
| `plot_data_cleaning.py` | Vorher/Nachher-Vergleich | `results/plots/` |

**Ordnerstruktur-Logik:**
- **Cross-Model Plots** (vergleichen mehrere Modelle) → `results/plots/`
- **Modell-spezifisch** (nur ein Modell) → `results/<model>/plots/`

---

## 🧰 6. `utils/` — Hilfsfunktionen

| Datei | Aufgabe |
|-------|---------|
| `config_loader.py` | YAML-Validierung und Laden |
| `load_dataset_config.py` | Dataset-Config-Loader |
| `json_results.py` | Metriken-Export |
| `load_trained_tft.py` | Checkpoint-Loader |

**Wichtig:** Utils werden NICHT direkt aufgerufen, sondern von anderen Scripts importiert.

---

## 📄 7. `pipeline.py` — Orchestrierung

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

## ⚙️ 8. `config.py` — Zentrale Steuerung

**Aktuell enthält:**
- Pfade: `RAW_DIR`, `INTERIM_DIR`, `PROCESSED_DIR`, `BASE_DIR`
- Evaluation-Konstanten: `EVALUATION_METRICS`, `EVALUATION_SPLITS`
- Wird von allen Scripts importiert für Pfad-Konstanten

**Nicht mehr verwendet für:**
- Schema (jetzt in `configs/datasets/*.yaml`)
- Split-Parameter (jetzt in `configs/datasets/*.yaml`)
- Feature-Configs (jetzt in `configs/datasets/*.yaml`)
- Trainings-Hyperparameter (immer schon in `configs/models/*.yaml`)

---

## 📁 9. `configs/` — Konfigurationen

```
configs/
├── datasets/              # Dataset-spezifische Configs
│   ├── booksales.yaml     # Täglich, 3 ID-Spalten, EU-Feiertage
│   └── walmart.yaml       # Wöchentlich, 2 ID-Spalten, US-Feiertage
│       
└── models/
    ├── tft/
    │   ├── booksales/
    │   │   ├── baseline.yaml
    │   │   └── optuna_best.yaml
    │   └── walmart/
    │       └── baseline.yaml
    ├── prophet/
    │   └── booksales/
    │       └── baseline.yaml
    └── arima/
        └── booksales/
            └── baseline.yaml
```

### Dataset-Config Struktur

```yaml
name: "dataset_name"
description: "Dataset description"

paths:
  raw: "data/raw/dataset_name"
  interim: "data/interim/dataset_name"
  processed: "data/processed/dataset_name"

raw_data:
  type: "multiple_files"  # oder "single_file"
  files:
    - path: "data/raw/dataset_name/train.csv"
      role: "main"
  merge:
    merge_on: ["id_col1", "id_col2"]
    how: "left"

schema:
  time_col: "date"
  id_cols: ["group1", "group2"]
  target_col: "target"

preprocessing:
  - step: "load_raw"
    enabled: true
  - step: "feature_engineering"
    enabled: true
    params:
      country: "US"

split:
  method: "ratio"
  ratios: [0.80, 0.10, 0.10]

tft:
  max_encoder_length: 16
  max_prediction_length: 4
  known_real_prefixes: ["cyc_"]
  lag_prefixes: ["lag_"]
  treat_calendar_as_known: true
  impute_cols:
    col1: 0.0
    col2: 0.0
  exclude_cols: ["lag_365"]
```

### Model-Config Struktur

```yaml
type: "tft"  # oder "prophet", "arima"
name: "baseline"
description: "Baseline configuration"

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
  # Modell-spezifische Parameter
  target_normalizer_transformation: null
  loss: "quantile"
  output_size: 7
  hidden_size: 32
  attention_head_size: 4
  hidden_continuous_size: 16
  dropout: 0.1
```

---

## 📂 10. `results/` — Outputs

```
results/
├── plots/                    # Übergreifende Cross-Model Plots
│   ├── cross_model_comparison_best.png
│   ├── multi_model_forecast_*.png
│   └── baseline_vs_optuna.png
│
├── eval/                     # Cross-Model Aggregation
│   ├── model_comparison.csv
│   └── model_comparison.json
│
├── pipeline_runs/            # Pipeline-Manifests
│   └── pipeline_YYYYMMDD_HHMMSS_manifest.json
│
├── tft/
│   ├── runs/
│   │   └── run_YYYYMMDD_HHMMSS_<dataset>_<config>/
│   │       ├── checkpoints/
│   │       ├── predictions/
│   │       ├── results.json
│   │       ├── summary.json
│   │       ├── eval_val.json
│   │       └── eval_test.json
│   ├── eval_overview.csv
│   └── plots/                # TFT-spezifische Plots
│       ├── tft_forecast_series.png
│       └── tft_optuna_study.png
│
├── prophet/
│   ├── runs/
│   ├── eval_overview.csv
│   └── plots/                # Prophet-spezifische Plots
│
└── arima/
    ├── runs/
    ├── eval_overview.csv
    └── plots/                # ARIMA-spezifische Plots
```

---

## 📚 11. `docs/` — Dokumentation

```
docs/
├── Scripts.md                      # Script-Übersicht (ZENTRAL)
├── Projektstruktur.md              # Diese Datei
├── PipelineOrder.md                # Workflow-Übersicht
├── ORDNERSTRUKTUR.md               # Visualization-Ordnerlogik
├── OPTUNA.md                       # Hyperparameter-Tuning Guide
├── Pipeline.md                     # Pipeline-Orchestrierung
├── ConfigSetup.md                  # Config-System
├── DatasetTFT.md                   # TFT-Datensatz-Spezifikation
└── TrainerTFT.md                   # TFT-Training
```

**Wichtigste Dokumente:**
- **Scripts.md** — Alle verfügbaren Scripts mit Beispielen
- **ORDNERSTRUKTUR.md** — Erklärt visualization/ vs results/ Struktur

---

## ✅ 12. Erweiterbarkeit

### Neues Modell hinzufügen:

1. **Config erstellen:** `configs/models/<model>/baseline.yaml`
2. **Trainer erstellen:** `src/modeling/trainer_<model>.py`
3. **Evaluation erstellen:** `src/evaluation/evaluate_<model>.py`
4. **Aggregation erstellen:** `src/evaluation/aggregate_<model>_eval.py`
5. **Plots erstellen:** `src/visualization/plot_<model>_*.py` → `results/<model>/plots/`

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
Preprocessing → model_dataset → dataset_<model> → trainer_<model> → evaluate_<model>
     ↓              ↓               ↓                  ↓                 ↓
   interim/      processed/      spec.json        checkpoints/      eval_*.json
     
Aggregation-Flow:
  evaluate_<model> → aggregate_<model>_eval → aggregate_all_models_eval
                          ↓                           ↓
                  eval_overview.csv           model_comparison.csv
                          
Visualization-Flow:
  model_comparison.csv → plot_cross_model_comparison → results/plots/
  eval_overview.csv    → plot_<model>_*               → results/<model>/plots/
```

**Ausführungsmodi:**
- ✅ Einzelne Scripts (für Debugging)
- ✅ Pipeline-Orchestrierung (für Production)
- ✅ Automatische Datensatz-Analyse (für neue Daten)

---

## 🔑 14. Wichtige Konzepte

### Multi-Model-Support
- **TFT:** Deep Learning, Attention-basiert
- **Prophet:** Statistisch, Facebook
- **ARIMA:** Statistisch, Klassisch
- **Evaluation:** Einheitliche Metriken für faire Vergleiche

### Visualization-Hierarchie
- **Übergreifend:** Cross-Model Vergleiche → `results/plots/`
- **Modell-spezifisch:** Einzelmodell-Analysen → `results/<model>/plots/`

### Dataset-spezifische Anpassungen
- **Imputing:** NaN-Werte mit festen Werten füllen
- **Exclusion:** Spalten entfernen
- **Normalisierung:** Target-Transformation wählen

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