# Pipeline Overview — TFT-TimeSeries

**Ziel & Inhalt:** Vollständige Übersicht über die Pipeline-Reihenfolge. Beschreibt Input, Output und Zweck aller Module von Rohdaten-Laden bis Training, Evaluation und Visualization.

---

## Ziel

Diese Übersicht beschreibt die **Ausführungsreihenfolge** der zentralen Module — von Rohdaten bis Visualization.  
Alle Schritte können einzeln getestet werden. Die Hauptpipeline umfasst Preprocessing, Training, Evaluation und Visualization.

---

## Projektlogik

| Ebene | Ordner | Aufgabe |
|-------|--------|---------|
| **data** | `src/data/` | Laden, Bereinigen und Feature-Erzeugung |
| **modeling** | `src/modeling/` | Splitten, Datensatz-Spezifikation, Modelltraining |
| **evaluation** | `src/evaluation/` | Metriken-Berechnung und Aggregation |
| **visualization** | `src/visualization/` | Plots für Daten, Training, Evaluation und Modellvergleiche |
| **utils** | `src/utils/` | Hilfsfunktionen — **nicht Teil der Pipeline** |
| **pipeline** | `src/pipeline.py` | Orchestrierung von Preprocessing und Training |

---

## Pipeline-Reihenfolge

### Phase 1: Preprocessing & Training

| # | Modul | Beschreibung | Input | Output |
|--:|-------|--------------|-------|--------|
| 1 | `load_raw.py` | Laden und Mergen von Rohdaten | `data/raw/<dataset>/*.csv` | `train_raw.parquet` |
| 2 | `data_alignment.py` | Skaliert/normalisiert Zeitreihen *(optional)* | Schritt 1 | `train_aligned.parquet` |
| 3 | `data_cleaning.py` | Bereinigt Target (Clip, NaN, dtype) | Schritt 1 oder 2 | `train_cleaned.parquet` |
| 4A | `feature_engineering.py` | Kalender-Features, time_idx, Feiertage | Schritt 3 | `train_features.parquet` |
| 4B | `cyclical_encoder.py` | Zyklische Sin/Cos-Kodierungen | Schritt 4A | `train_features_cyc.parquet` |
| 4C | `lag_features.py` | Lag-/Rolling-Features, Gruppen-Filter, NaN-Imputation | Schritt 4B | `train_features_cyc_lag.parquet` |
| 5 | `model_dataset.py` | Zeitbasierter Split (Train/Val/Test) | Schritt 4C | `train.parquet`, `val.parquet`, `test.parquet` |
| 6 | `dataset_<model>.py` | Modell-spezifische Datensatz-Spezifikation | Schritt 5 | `dataset_spec.json` |
| 7 | `trainer_<model>.py` | Modell-Training | Schritt 6 + Model-YAML | Checkpoints, Logs, JSONs |

### Phase 2: Evaluation

| # | Modul | Beschreibung | Input | Output |
|--:|-------|--------------|-------|--------|
| 8 | `evaluate_<model>.py` | Berechnet Fehlermaße (MAE, RMSE, MAPE, SMAPE, R²) | Checkpoint + Daten | `eval_val.json`, `eval_test.json` |
| 9 | `aggregate_<model>_eval.py` | Aggregiert Evaluierungen pro Modell | Schritt 8 | `eval_overview.csv` |
| 10 | `aggregate_all_models_eval.py` | Kombiniert alle Modelle | Schritt 9 | `model_comparison.csv` |

### Phase 3: Visualization

| # | Modul | Beschreibung | Input | Output |
|--:|-------|--------------|-------|--------|
| 11 | `plot_cross_model_comparison.py` | Cross-Model Performance Vergleich | `model_comparison.csv` | `results/plots/cross_model_comparison_*.png` |
| 12 | `plot_multi_model_forecast.py` | Multi-Model Forecast Visualisierung | `model_comparison.csv` + Predictions | `results/plots/multi_model_forecast_*.png` |
| 13 | `plot_<model>_*.py` | Modell-spezifische Analysen | Modell-Daten | `results/<model>/plots/*.png` |

---

## Datenfluss

```
data/raw/<dataset>/
        ↓
data/interim/<dataset>/
    train_raw.parquet → train_aligned.parquet → train_cleaned.parquet
        ↓
data/processed/<dataset>/
    train_features.parquet → train_features_cyc.parquet → train_features_cyc_lag.parquet
        ↓
    train.parquet, val.parquet, test.parquet, dataset_spec.json
        ↓
results/<model>/runs/<run_id>/
    checkpoints/, results.json, summary.json, eval_val.json, eval_test.json
        ↓
results/<model>/eval_overview.csv
        ↓
results/eval/model_comparison.csv
        ↓
results/plots/  (Cross-Model)
results/<model>/plots/  (Modell-spezifisch)
```

---

## Aufgabenverteilung

### Preprocessing
| Aufgabe | Modul | Beschreibung |
|---------|-------|--------------|
| Target clippen | `data_cleaning.py` | Negative Werte auf Minimum setzen |
| Target dtype | `data_cleaning.py` | Konvertierung zu float32 |
| Target-NaN entfernen | `data_cleaning.py` | Zeilen mit NaN im Target entfernen |
| Lag-NaN Imputation | `lag_features.py` | Median + Missing-Indicator |
| Gruppen filtern | `lag_features.py` | Zu kurze Zeitreihen entfernen |

### Evaluation
| Aufgabe | Modul | Beschreibung |
|---------|-------|--------------|
| Einzelmodell-Metriken | `evaluate_<model>.py` | MAE, RMSE, MAPE, SMAPE, R² pro Run |
| Modell-Aggregation | `aggregate_<model>_eval.py` | Alle Runs eines Modells kombinieren |
| Cross-Model Aggregation | `aggregate_all_models_eval.py` | Alle Modelle vergleichen |

### Visualization
| Aufgabe | Modul | Output-Pfad |
|---------|-------|-------------|
| Cross-Model Vergleich | `plot_cross_model_comparison.py` | `results/plots/` |
| Multi-Model Forecasts | `plot_multi_model_forecast.py` | `results/plots/` |
| Modell-spezifische Plots | `plot_<model>_*.py` | `results/<model>/plots/` |

---

## Aufruf-Beispiele

### Via Pipeline (empfohlen für Preprocessing + Training)

```bash
# Kompletter Run (Preprocessing + Training)
python -m src.pipeline \
    --dataset configs/datasets/<dataset>.yaml \
    --model configs/models/<model>/<dataset>/<config>.yaml

# Nur Preprocessing
python -m src.pipeline \
    --dataset configs/datasets/<dataset>.yaml \
    --steps preprocessing,model_dataset,dataset_<model>

# Nur Training (Preprocessing bereits erledigt)
python -m src.pipeline \
    --dataset configs/datasets/<dataset>.yaml \
    --model configs/models/<model>/<dataset>/<config>.yaml \
    --steps training
```

### Evaluation Workflow (manuell)

```bash
# 1. Einzelmodelle evaluieren
python -m src.evaluation.evaluate_tft --run-id <run_id> --split test
python -m src.evaluation.evaluate_prophet --run-id <run_id> --split test
python -m src.evaluation.evaluate_arima --run-id <run_id> --split test

# 2. Pro Modell aggregieren
python -m src.evaluation.aggregate_tft_eval
python -m src.evaluation.aggregate_prophet_eval
python -m src.evaluation.aggregate_arima_eval

# 3. Alle Modelle kombinieren
python -m src.evaluation.aggregate_all_models_eval
```

### Visualization Workflow (manuell)

```bash
# Übergreifende Cross-Model Plots
python -m src.visualization.plot_cross_model_comparison
python -m src.visualization.plot_multi_model_forecast

# Modell-spezifische Plots
python -m src.visualization.plot_tft_forecast_series --run-id <run_id>
python -m src.visualization.plot_tft_optuna_study --study-name <study_name>
python -m src.visualization.plot_prophet_optuna_study --study-name <study_name>
python -m src.visualization.plot_arima_optuna_study --study-name <study_name>
```

### Einzeln (für Tests/Debugging)

```bash
# Umgebungsvariable setzen
$env:DATASET_CONFIG="configs/datasets/<dataset>.yaml"

# Preprocessing-Schritte
python -m src.data.load_raw
python -m src.data.data_cleaning
python -m src.data.feature_engineering
python -m src.data.cyclical_encoder
python -m src.data.lag_features

# Modeling-Schritte
python -m src.modeling.model_dataset
python -m src.modeling.dataset_<model>
python -m src.modeling.trainer_<model> --config configs/models/<model>/<dataset>/<config>.yaml
```

---

## Output-Struktur

```
logs/<model>/<run_id>/
├── metrics.csv
└── hparams.yaml

results/<model>/runs/<run_id>/
├── checkpoints/
│   └── <model>-epoch=XX-val_loss=Y.YYYY.ckpt
├── predictions/          # Falls Predictions gespeichert
├── results.json
├── summary.json
├── eval_val.json
└── eval_test.json

results/<model>/
├── eval_overview.csv     # Pro Modell
└── plots/                # Modell-spezifische Plots

results/eval/
├── model_comparison.csv  # Alle Modelle kombiniert
└── model_comparison.json

results/plots/            # Übergreifende Cross-Model Plots
├── cross_model_comparison_best.png
├── cross_model_comparison_combined.png
├── multi_model_forecast_<dataset>.png
└── baseline_vs_optuna.png

results/pipeline_runs/    # Nur bei Pipeline-Aufruf
└── pipeline_<timestamp>_manifest.json
```

---

## Workflow-Hierarchie

### 1. Kompletter Workflow (alle Phasen)

```mermaid
Preprocessing → Training → Evaluation → Aggregation → Visualization
```

**Ausführung:**
```bash
# Phase 1: Preprocessing + Training (via Pipeline)
python -m src.pipeline --dataset <dataset>.yaml --model <model>.yaml

# Phase 2: Evaluation (manuell)
python -m src.evaluation.evaluate_<model> --run-id <run_id> --split test

# Phase 3: Aggregation (manuell)
python -m src.evaluation.aggregate_<model>_eval
python -m src.evaluation.aggregate_all_models_eval

# Phase 4: Visualization (manuell)
python -m src.visualization.plot_cross_model_comparison
```

### 2. Nur Evaluation + Visualization (Training bereits erledigt)

```bash
# Evaluation
python -m src.evaluation.evaluate_tft --run-id <run_id> --split test
python -m src.evaluation.evaluate_prophet --run-id <run_id> --split test
python -m src.evaluation.evaluate_arima --run-id <run_id> --split test

# Aggregation
python -m src.evaluation.aggregate_tft_eval
python -m src.evaluation.aggregate_prophet_eval
python -m src.evaluation.aggregate_arima_eval
python -m src.evaluation.aggregate_all_models_eval

# Visualization
python -m src.visualization.plot_cross_model_comparison
python -m src.visualization.plot_multi_model_forecast
```

### 3. Nur Visualization (Evaluation bereits erledigt)

```bash
# Nur Plots erstellen
python -m src.visualization.plot_cross_model_comparison
python -m src.visualization.plot_multi_model_forecast
python -m src.visualization.plot_tft_forecast_series --run-id <run_id>
```

---

## Hinweise

### Automatische vs. Manuelle Schritte

**Via Pipeline automatisch:**
- Schritte 1-7: Preprocessing + Training

**Immer manuell:**
- Schritte 8-10: Evaluation & Aggregation
- Schritte 11-13: Visualization

**Grund:** Evaluation und Visualization sind bewusst getrennt, um flexible Analysen zu ermöglichen.

### Reihenfolge beachten

- Schritte 1-7 müssen **sequenziell** ausgeführt werden
- Schritte 8-10 erfordern abgeschlossene Trainings (Schritt 7)
- Schritte 11-13 erfordern aggregierte Daten (Schritt 10)

### Pipeline-Manifests

Pipeline-Manifests dokumentieren den kompletten Workflow von Schritt 1-7:
```json
{
  "run_id": "pipeline_YYYYMMDD_HHMMSS",
  "dataset": "booksales",
  "model": "tft_baseline",
  "steps_executed": ["preprocessing", "model_dataset", "dataset_tft", "training"],
  "outputs": {
    "processed_data": "data/processed/booksales/",
    "training_run": "results/tft/runs/run_YYYYMMDD_HHMMSS_tft_baseline/",
    "checkpoints": ["..."]
  }
}
```

### Output-Hierarchie

```
results/
├── plots/                # ÜBERGREIFEND (Cross-Model)
├── eval/                 # ÜBERGREIFEND (Cross-Model)
├── <model>/
│   ├── runs/             # PRO RUN
│   ├── eval_overview.*   # PRO MODELL
│   └── plots/            # PRO MODELL
└── pipeline_runs/        # PIPELINE-LOGS
```

---

## Typische Use Cases

### Use Case 1: Neues Modell trainieren

```bash
python -m src.pipeline \
    --dataset configs/datasets/<dataset>.yaml \
    --model configs/models/<model>/<dataset>/<config>.yaml
```

### Use Case 2: Bestehendes Modell evaluieren

```bash
python -m src.evaluation.evaluate_<model> --run-id <run_id> --split test
```

### Use Case 3: Alle Modelle vergleichen

```bash
# Aggregiere alle
python -m src.evaluation.aggregate_all_models_eval

# Visualisiere
python -m src.visualization.plot_cross_model_comparison
```

### Use Case 4: Modell-spezifische Analyse

```bash
# Aggregiere nur ein Modell
python -m src.evaluation.aggregate_tft_eval

# Modell-spezifische Plots
python -m src.visualization.plot_tft_eval_comparison
python -m src.visualization.plot_tft_forecast_series --run-id <run_id>
```

---