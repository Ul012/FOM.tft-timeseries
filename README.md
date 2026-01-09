# TFT-TimeSeries — Multi-Model Forecasting

Modulare, erweiterbare Pipeline für Zeitreihen-Forecasting mit **Temporal Fusion Transformer (TFT)**, **ARIMA** und **Prophet**. 

Der Fokus liegt auf einer klar strukturierten, konfigurationsgetriebenen und reproduzierbaren Umsetzung.

---

## 📊 Modelle

| Modell | Typ | Status |
|--------|-----|--------|
| **TFT** | Deep Learning (Attention-based) | ✅ Production |
| **ARIMA** | Statistisch (SARIMA mit auto_arima) | ✅ Production |
| **Prophet** | Statistisch (Facebook Prophet) | ✅ Production |

**Besonderheit:** Alle Modelle nutzen die gleiche Evaluation-Pipeline für faire Vergleiche.

---

## 📁 Datasets

Als **Beispiel-Datasets** dienen:
- **Booksales** (Kaggle Tabular Playground Series — Sep 2022) - Täglich
- **Walmart** (Kaggle Store Sales Forecasting) - Wöchentlich

Die Architektur unterstützt beliebig viele Datasets gleichzeitig.

### Download

1. Kaggle-Account erstellen (falls nicht vorhanden)
2. Dataset-Seite besuchen und "Download All" klicken
3. ZIP entpacken und Dateien in Ordner kopieren:

```
data/raw/booksales/
├── train.csv
└── test.csv (optional)

data/raw/walmart/
├── train.csv
├── features.csv
└── test.csv (optional)
```

**Hinweis:** Rohdaten werden nicht versioniert (siehe `.gitignore`).

---

## 🚀 Quick Start

### Installation

```bash
# Virtual Environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Dependencies
pip install -r requirements.txt
```

### Kompletter Durchlauf

```bash
# TFT
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/booksales/baseline.yaml

# ARIMA
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"
python -m src.modeling.dataset_arima
python -m src.modeling.trainer_arima --config configs/models/arima/booksales/baseline.yaml

# Prophet
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"
python -m src.modeling.dataset_prophet
python -m src.modeling.trainer_prophet --config configs/models/prophet/booksales/baseline.yaml
```

**Was passiert:**
1. Preprocessing (Load, Cleaning, Feature Engineering)
2. Model-spezifische Datenaufbereitung
3. Training
4. Checkpoint-Speicherung

---

## 📊 Evaluation

```bash
# Val Split
python -m src.evaluation.evaluate_tft --run-id <run_id> --split val
python -m src.evaluation.evaluate_arima --run-id <run_id> --split val
python -m src.evaluation.evaluate_prophet --run-id <run_id> --split val

# Test Split
python -m src.evaluation.evaluate_tft --run-id <run_id> --split test
python -m src.evaluation.evaluate_arima --run-id <run_id> --split test
python -m src.evaluation.evaluate_prophet --run-id <run_id> --split test
```

**Output:**
- `results/<model>/runs/<run_id>/eval_val.json`
- `results/<model>/runs/<run_id>/eval_test.json`

---

## 🎯 Hyperparameter-Tuning (Optuna)

```bash
# TFT (50 Trials, ~25h)
python -m src.modeling.optuna_tft --study-name tft_booksales --n-trials 50

# ARIMA (50 Trials, ~2.5h für Booksales)
python -m src.modeling.optuna_arima --study-name arima_booksales --n-trials 50

# Prophet (50 Trials, ~8h)
python -m src.modeling.optuna_prophet --study-name prophet_booksales --n-trials 50

# Beste Config exportieren
python -m src.modeling.optuna_<model>_export_best --study-name <study_name>

# Training mit bester Config
python -m src.pipeline \
    --model configs/models/<model>/optuna_best.yaml \
    --dataset configs/datasets/booksales.yaml \
    --steps training
```

**Details:** Siehe [docs/OPTUNA.md](docs/OPTUNA.md)

---

## 📂 Projektstruktur

```
TFT-TimeSeries/
├── src/
│   ├── data/              # Preprocessing
│   ├── modeling/          # Training (TFT, ARIMA, Prophet)
│   ├── evaluation/        # Metriken-Berechnung
│   ├── visualization/     # Plots
│   ├── utils/             # Hilfsfunktionen
│   └── pipeline.py        # Orchestrierung
│
├── configs/
│   ├── datasets/          # booksales.yaml, walmart.yaml
│   └── models/
│       ├── tft/           # TFT-Configs
│       ├── arima/         # ARIMA-Configs
│       └── prophet/       # Prophet-Configs
│
├── data/
│   ├── raw/<dataset>/     # Rohdaten (nicht versioniert)
│   ├── interim/<dataset>/ # Zwischenschritte
│   └── processed/<dataset>/ # Features, Splits
│
├── results/
│   ├── tft/               # TFT Runs, Optuna, Plots
│   ├── arima/             # ARIMA Runs, Optuna
│   └── prophet/           # Prophet Runs, Optuna
│
├── logs/                  # Training-Logs
│
└── docs/                  # Dokumentation
    ├── SCRIPTS.md         # ⭐ Alle Scripts auf einen Blick
    ├── OPTUNA.md          # Hyperparameter-Tuning Guide
    ├── OptunaTFT.md       # TFT Optuna Details
    ├── OptunaARIMA.md     # ARIMA Optuna Details
    └── OptunaProphet.md   # Prophet Optuna Details
```

---

## 📖 Dokumentation

### Zentrale Guides
- **[SCRIPTS.md](docs/SCRIPTS.md)** — Alle Scripts mit Beispiel-Aufrufen
- **[OPTUNA.md](docs/OPTUNA.md)** — Hyperparameter-Tuning Workflow

### Detaillierte Docs
- **[OptunaTFT.md](docs/OptunaTFT.md)** — TFT Optuna-Integration (technisch)
- **[OptunaARIMA.md](docs/OptunaARIMA.md)** — ARIMA Optuna-Integration (technisch)
- **[OptunaProphet.md](docs/OptunaProphet.md)** — Prophet Optuna-Integration (technisch)

### Legacy Docs (in `docs/`)
- LoadRaw.md, DataAlignment.md, DataCleaning.md
- FeatureEngineer.md, CyclicalEncoder.md, LagFeatures.md
- Pipeline.md, ConfigSetup.md, Projektstruktur.md

---

## 🔧 Konfiguration

### Dataset Config (`configs/datasets/<dataset>.yaml`)
- Schema (Spalten, time_col, target_col, group_cols)
- Raw Data Loading (single/multiple files)
- Preprocessing-Pipeline
- Split-Konfiguration
- Modell-Parameter

### Model Config (`configs/models/<model>/<config>.yaml`)
- **TFT:** Hyperparameter (learning_rate, hidden_size, dropout, etc.)
- **ARIMA:** auto_arima Parameter (max_p, max_q, seasonal, etc.)
- **Prophet:** Seasonality, Changepoints, Holidays

---

## 📊 Outputs

### Training
```
results/<model>/runs/<run_id>/
├── checkpoints/           # Beste Modelle
├── summary.json           # Training-Summary
├── eval_val.json          # Validation Metriken
└── eval_test.json         # Test Metriken
```

### Optuna
```
results/<model>/optuna/<dataset>/
├── <model>_studies.db     # SQLite mit allen Trials
├── trial_<n>/             # Pro Trial: Checkpoint + Summary
├── plots/                 # Visualisierungen
└── analysis/              # CSV-Exports
```

### Logs
```
logs/<model>/<run_id>/
└── metrics.csv            # PyTorch Lightning / Pandas
```

---

## 🎯 Typische Workflows

### Workflow 1: Baseline-Training
```bash
# 1. Dataset vorbereiten
python -m src.pipeline --dataset configs/datasets/booksales.yaml --steps preprocessing

# 2. Alle 3 Modelle trainieren
python -m src.pipeline --dataset configs/datasets/booksales.yaml --model configs/models/tft/booksales/baseline.yaml --steps training

$env:DATASET_CONFIG="configs/datasets/booksales.yaml"
python -m src.modeling.dataset_arima
python -m src.modeling.trainer_arima --config configs/models/arima/booksales/baseline.yaml

python -m src.modeling.dataset_prophet
python -m src.modeling.trainer_prophet --config configs/models/prophet/booksales/baseline.yaml

# 3. Evaluieren
python -m src.evaluation.evaluate_tft --run-id <run_id> --split test
python -m src.evaluation.evaluate_arima --run-id <run_id> --split test
python -m src.evaluation.evaluate_prophet --run-id <run_id> --split test
```

### Workflow 2: Hyperparameter-Tuning
```bash
# 1. Optuna durchführen (mehrere Stunden/Tage)
python -m src.modeling.optuna_tft --study-name tft_booksales --n-trials 50

# 2. Analysieren
python -m src.evaluation.analyze_optuna_tft_trials --study-name tft_booksales
python -m src.visualization.plot_tft_optuna_study --study-name tft_booksales

# 3. Beste Config exportieren
python -m src.modeling.optuna_tft_export_best --study-name tft_booksales

# 4. Finales Training
python -m src.pipeline \
    --model configs/models/tft/optuna_best.yaml \
    --dataset configs/datasets/booksales.yaml \
    --steps training
```

### Workflow 3: Neues Dataset hinzufügen
```bash
# 1. Rohdaten ablegen
# data/raw/neues_dataset/train.csv

# 2. Config erstellen (Kopie von bestehender Config als Vorlage)
Copy-Item configs/datasets/booksales.yaml configs/datasets/neues_dataset.yaml
# Dann anpassen: name, paths, schema

# 3. Pipeline ausführen
python -m src.pipeline \
    --dataset configs/datasets/neues_dataset.yaml \
    --model configs/models/tft/baseline.yaml
```

---

## 🔬 Modell-Vergleich

### Metriken
- **MAE** (Mean Absolute Error) - Hauptmetrik
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **SMAPE** (Symmetric MAPE)

### Evaluation-Outputs
Jedes Modell erstellt `eval_val.json` und `eval_test.json` mit:
```json
{
  "run_id": "...",
  "dataset": "booksales",
  "split": "test",
  "n_groups": 48,
  "metrics": {
    "by_group": {...},
    "overall": {
      "mae": 14.29,
      "rmse": 21.03,
      "mape": 6.35
    }
  }
}
```

---

## 🧪 Besonderheiten

### ARIMA
- **Resume-Training:** Bei Unterbrechung fortsetzen mit `resume_arima_training.py`
- **Seasonal:** m=7 (Booksales), m=52 (Walmart)
- **Training-Dauer:** Sehr variabel (Booksales: 3min, Walmart: Tage)

### Prophet
- **Additive/Multiplicative Seasonality**
- **Holidays:** Automatische Erkennung
- **Changepoints:** Flexible Trendänderungen

### TFT
- **Attention-Mechanismus:** Interpretierbare Vorhersagen
- **Multi-Horizon:** Mehrere Steps gleichzeitig
- **GPU-Accelerated:** ~10x schneller als CPU

---

## 🎓 Wissenschaftlicher Kontext

Dieses Projekt wurde im Rahmen einer Seminararbeit entwickelt:
- **Fokus:** Vergleich Deep Learning (TFT) vs. klassische Methoden (ARIMA, Prophet)
- **Datasets:** Unterschiedliche Frequenzen (täglich vs. wöchentlich)
- **Evaluation:** Faire Vergleiche durch einheitliche Metriken und Splits

---

## 🤝 Zusammenarbeit

Die modulare Struktur ermöglicht:
- Paralleles Arbeiten an verschiedenen Modellen
- Reproduzierbare Ergebnisse durch Configs und Seeds
- Einfache Erweiterung um neue Modelle/Datasets
- Klare Verantwortlichkeiten (Preprocessing, Modeling, Evaluation getrennt)

**Änderungen** werden minimalinvasiv umgesetzt, sodass die Gesamtstruktur stabil bleibt.

---

## 📝 Nächste Schritte

Siehe [SCRIPTS.md](docs/SCRIPTS.md) für vollständige Script-Übersicht und [OPTUNA.md](docs/OPTUNA.md) für Hyperparameter-Tuning.