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

Die Architektur unterstützt beliebig viele Datasets gleichzeitig mit unterschiedlichen Frequenzen und Strukturen.

### Beispiel-Datasets

Als Demonstration dienen zwei öffentliche Kaggle-Datasets:
- **Booksales** (Kaggle Tabular Playground Series — Sep 2022) - Täglich
- **Walmart** (Kaggle Store Sales Forecasting) - Wöchentlich

### Datenstruktur

```
data/raw/<dataset>/
├── train.csv
├── features.csv (optional)
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
    --dataset configs/datasets/<dataset>.yaml \
    --model configs/models/tft/<dataset>/baseline.yaml

# ARIMA
$env:DATASET_CONFIG="configs/datasets/<dataset>.yaml"
python -m src.modeling.dataset_arima
python -m src.modeling.trainer_arima --config configs/models/arima/<dataset>/baseline.yaml

# Prophet
$env:DATASET_CONFIG="configs/datasets/<dataset>.yaml"
python -m src.modeling.dataset_prophet
python -m src.modeling.trainer_prophet --config configs/models/prophet/<dataset>/baseline.yaml
```

**Was passiert:**
1. Preprocessing (Load, Cleaning, Feature Engineering)
2. Modell-spezifische Datenaufbereitung
3. Training
4. Checkpoint-Speicherung

---

## 📊 Evaluation & Visualization

### Evaluation Workflow

```bash
# 1. Einzelmodelle evaluieren
python -m src.evaluation.evaluate_tft --run-id <run_id> --split test
python -m src.evaluation.evaluate_arima --run-id <run_id> --split test
python -m src.evaluation.evaluate_prophet --run-id <run_id> --split test

# 2. Pro Modell aggregieren
python -m src.evaluation.aggregate_tft_eval
python -m src.evaluation.aggregate_prophet_eval
python -m src.evaluation.aggregate_arima_eval

# 3. Alle Modelle kombinieren
python -m src.evaluation.aggregate_all_models_eval
```

**Output:**
- `results/<model>/runs/<run_id>/eval_val.json`
- `results/<model>/runs/<run_id>/eval_test.json`
- `results/<model>/eval_overview.csv`
- `results/eval/model_comparison.csv`

### Visualization Workflow

```bash
# Cross-Model Vergleiche
python -m src.visualization.plot_cross_model_comparison
python -m src.visualization.plot_multi_model_forecast

# Modell-spezifische Analysen
python -m src.visualization.plot_tft_forecast_series --run-id <run_id>
python -m src.visualization.plot_tft_optuna_study --study-name <study_name>
```

**Output:**
- Übergreifend: `results/plots/`
- Modell-spezifisch: `results/<model>/plots/`

---

## 🎯 Hyperparameter-Tuning (Optuna)

```bash
# TFT
python -m src.modeling.optuna_tft --study-name tft_<dataset> --n-trials 50

# ARIMA
python -m src.modeling.optuna_arima --study-name arima_<dataset> --n-trials 50

# Prophet
python -m src.modeling.optuna_prophet --study-name prophet_<dataset> --n-trials 50

# Beste Config exportieren
python -m src.modeling.optuna_<model>_export_best --study-name <study_name>

# Training mit bester Config
python -m src.pipeline \
    --model configs/models/<model>/optuna_best.yaml \
    --dataset configs/datasets/<dataset>.yaml \
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
│   ├── evaluation/        # Metriken-Berechnung & Aggregation
│   ├── visualization/     # Plots & Grafiken
│   ├── utils/             # Hilfsfunktionen
│   └── pipeline.py        # Orchestrierung
│
├── configs/
│   ├── datasets/          # Dataset-Konfigurationen
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
│   ├── plots/             # Übergreifende Cross-Model Plots
│   ├── eval/              # Cross-Model Aggregationen
│   ├── tft/               # TFT Runs, Eval, Plots
│   ├── arima/             # ARIMA Runs, Eval, Plots
│   └── prophet/           # Prophet Runs, Eval, Plots
│
├── logs/                  # Training-Logs
│
└── docs/                  # Dokumentation
    ├── Scripts.md         # ⭐ Alle Scripts auf einen Blick
    ├── Projektstruktur.md # Detaillierte Struktur-Übersicht
    ├── PipelineOrder.md   # Pipeline-Workflow
    ├── ORDNERSTRUKTUR.md  # Visualization-Ordnerlogik
    └── OPTUNA.md          # Hyperparameter-Tuning Guide
```

---

## 📖 Dokumentation

### Zentrale Guides
- **[Scripts.md](docs/Scripts.md)** — Alle Scripts mit Beispiel-Aufrufen
- **[PipelineOrder.md](docs/PipelineOrder.md)** — Kompletter Workflow-Überblick
- **[ORDNERSTRUKTUR.md](docs/ORDNERSTRUKTUR.md)** — Visualization-Ordnerlogik
- **[OPTUNA.md](docs/OPTUNA.md)** — Hyperparameter-Tuning Workflow

### Detaillierte Docs
- **Projektstruktur.md** — Vollständige Struktur-Erklärung
- **ConfigSetup.md** — Config-System Details
- **OptunaTFT.md, OptunaARIMA.md, OptunaProphet.md** — Modell-spezifische Optuna-Integration

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
├── predictions/           # Predictions (falls gespeichert)
├── summary.json           # Training-Summary
├── eval_val.json          # Validation Metriken
└── eval_test.json         # Test Metriken
```

### Evaluation
```
results/<model>/
├── eval_overview.csv      # Alle Runs pro Modell
└── eval_overview.json

results/eval/
├── model_comparison.csv   # Alle Modelle kombiniert
└── model_comparison.json
```

### Visualization
```
results/plots/             # Übergreifende Cross-Model Plots
├── cross_model_comparison_best.png
├── multi_model_forecast_<dataset>.png
└── baseline_vs_optuna.png

results/<model>/plots/     # Modell-spezifische Plots
├── <model>_forecast_series.png
├── <model>_optuna_study.png
└── <model>_eval_comparison.png
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
python -m src.pipeline --dataset configs/datasets/<dataset>.yaml --steps preprocessing

# 2. Alle 3 Modelle trainieren
python -m src.pipeline --dataset configs/datasets/<dataset>.yaml --model configs/models/tft/<dataset>/baseline.yaml --steps training

$env:DATASET_CONFIG="configs/datasets/<dataset>.yaml"
python -m src.modeling.dataset_arima
python -m src.modeling.trainer_arima --config configs/models/arima/<dataset>/baseline.yaml

python -m src.modeling.dataset_prophet
python -m src.modeling.trainer_prophet --config configs/models/prophet/<dataset>/baseline.yaml

# 3. Evaluieren
python -m src.evaluation.evaluate_tft --run-id <run_id> --split test
python -m src.evaluation.evaluate_arima --run-id <run_id> --split test
python -m src.evaluation.evaluate_prophet --run-id <run_id> --split test

# 4. Aggregieren
python -m src.evaluation.aggregate_all_models_eval

# 5. Visualisieren
python -m src.visualization.plot_cross_model_comparison
```

### Workflow 2: Hyperparameter-Tuning

```bash
# 1. Optuna durchführen
python -m src.modeling.optuna_<model> --study-name <model>_<dataset> --n-trials 50

# 2. Analysieren
python -m src.evaluation.analyze_optuna_<model>_trials --study-name <model>_<dataset>
python -m src.visualization.plot_<model>_optuna_study --study-name <model>_<dataset>

# 3. Beste Config exportieren
python -m src.modeling.optuna_<model>_export_best --study-name <model>_<dataset>

# 4. Finales Training
python -m src.pipeline \
    --model configs/models/<model>/optuna_best.yaml \
    --dataset configs/datasets/<dataset>.yaml \
    --steps training
```

### Workflow 3: Neues Dataset hinzufügen

```bash
# 1. Rohdaten ablegen
# data/raw/<new_dataset>/train.csv

# 2. Config erstellen (Kopie von bestehender Config als Vorlage)
cp configs/datasets/<existing>.yaml configs/datasets/<new_dataset>.yaml
# Dann anpassen: name, paths, schema

# 3. Pipeline ausführen
python -m src.pipeline \
    --dataset configs/datasets/<new_dataset>.yaml \
    --model configs/models/tft/baseline.yaml
```

---

## 📬 Modell-Vergleich

### Metriken
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)
- **SMAPE** (Symmetric MAPE)
- **R²** (nur TFT)

### Evaluation-Outputs
Jedes Modell erstellt `eval_val.json` und `eval_test.json` mit:
```json
{
  "run_id": "...",
  "dataset": "<dataset>",
  "split": "test",
  "n_groups": 48,
  "metrics": {
    "by_group": {...},
    "overall": {
      "mae": 14.29,
      "rmse": 21.03,
      "mape": 6.35,
      "smape": 6.50
    }
  }
}
```

### Cross-Model Comparison
Die `model_comparison.csv` kombiniert alle Modelle für direkte Vergleiche:
```csv
model,dataset,type,run_id,val_smape,test_smape,val_mae,test_mae,...
TFT,booksales,Optuna,run_...,5.89,6.40,12.63,18.15,...
Prophet,booksales,Baseline,run_...,8.33,6.50,19.08,15.87,...
ARIMA,booksales,Optuna,run_...,11.06,20.61,23.64,45.30,...
```

---

## 🧪 Besonderheiten

### ARIMA
- **Resume-Training:** Bei Unterbrechung fortsetzen mit `resume_arima_training.py`
- **Seasonal:** Automatische Erkennung der Saisonalität
- **Training-Dauer:** Variabel je nach Datensatz-Größe

### Prophet
- **Additive/Multiplicative Seasonality**
- **Holidays:** Automatische Erkennung
- **Changepoints:** Flexible Trendänderungen

### TFT
- **Attention-Mechanismus:** Interpretierbare Vorhersagen
- **Multi-Horizon:** Mehrere Steps gleichzeitig
- **GPU-Accelerated:** Deutlich schneller als CPU

---

## 📊 Visualization-Hierarchie

### Übergreifende Plots (`results/plots/`)
Vergleichen **mehrere Modelle**:
- Cross-Model Performance Comparison
- Multi-Model Forecast Comparison
- Baseline vs. Optuna (alle Modelle)

### Modell-spezifische Plots (`results/<model>/plots/`)
Analysieren **ein einzelnes Modell**:
- Forecast Series Visualisierung
- Optuna Study Plots
- Evaluation Comparisons

**Regel:**  
- Cross-Model → `results/plots/`
- Single-Model → `results/<model>/plots/`

---

## 🤝 Zusammenarbeit

Die modulare Struktur ermöglicht:
- Paralleles Arbeiten an verschiedenen Modellen
- Reproduzierbare Ergebnisse durch Configs und Seeds
- Einfache Erweiterung um neue Modelle/Datasets
- Klare Verantwortlichkeiten (Preprocessing, Modeling, Evaluation, Visualization getrennt)

Änderungen werden minimalinvasiv umgesetzt, sodass die Gesamtstruktur stabil bleibt.

---

## 📝 Nächste Schritte

- Siehe [Scripts.md](docs/Scripts.md) für vollständige Script-Übersicht
- Siehe [PipelineOrder.md](docs/PipelineOrder.md) für Workflow-Details
- Siehe [OPTUNA.md](docs/OPTUNA.md) für Hyperparameter-Tuning
- Siehe [ORDNERSTRUKTUR.md](docs/ORDNERSTRUKTUR.md) für Visualization-Logik

---

## 📄 Lizenz

Dieses Projekt ist für Forschungs- und Bildungszwecke konzipiert.