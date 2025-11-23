# Pipeline Overview – FOM.tft-timeseries

**Datum:** 2025-11-23  
**Script:** —  
**Ziel & Inhalt:** Vollständige Übersicht über die Pipeline-Reihenfolge. Beschreibt Input, Output und Zweck aller Module von Rohdaten-Laden bis Training und optionaler Evaluation.

---

## Ziel

Diese Übersicht beschreibt die **Ausführungsreihenfolge** der zentralen Module – von Rohdaten bis Training und Evaluation.  
Alle Schritte können einzeln getestet werden. Schritte 1–7 bilden die Hauptpipeline.

---

## Projektlogik

| Ebene | Ordner | Aufgabe |
|------|--------|---------|
| **data** | `src/data/` | Laden, Bereinigen und Feature-Erzeugung (Kalender, Feiertage, Lags, zyklische Merkmale) |
| **modeling** | `src/modeling/` | Splitten, TFT-Datensatz, Modelltraining |
| **evaluation** | `src/evaluation/` | Ergebnisanalyse, Kennzahlen und Visualisierung |
| **visualization** | `src/visualization/` | Plots für Daten, Training und Evaluation |
| **utils** | `src/utils/` | Hilfsfunktionen – **nicht Teil der Pipeline** |
| **pipeline** | `src/pipeline.py` | Orchestrierung aller Schritte |

---

## Pipeline-Reihenfolge

| # | Modul | Beschreibung | Input | Output | Aufruf |
|--:|-------|--------------|-------|--------|--------|
| 1 | `load_raw.py` | Laden und Mergen von Rohdaten | `data/raw/<dataset_name>/*.csv` | `data/interim/<dataset_name>/train_raw.parquet` | Einzeln oder via Pipeline |
| 2 | `data_alignment.py` *(optional)* | Skaliert/normalisiert Zeitreihen | Schritt 1 | `data/interim/<dataset_name>/train_aligned.parquet` | Einzeln oder via Pipeline |
| 3 | `data_cleaning.py` *(optional)* | Bereinigt Ausreißer, glättet Lockdown | Schritt 1 oder 2 | `data/interim/<dataset_name>/train_cleaned.parquet` | Einzeln oder via Pipeline |
| 4A | `feature_engineering.py` | Kalender-Features, time_idx, Feiertage | Schritt 3 | `data/processed/<dataset_name>/train_features.parquet` | Einzeln oder via Pipeline |
| 4B | `cyclical_encoder.py` | Zyklische Sin/Cos-Kodierungen | `train_features.parquet` | `train_features_cyc.parquet` | Einzeln oder via Pipeline |
| 4C | `lag_features.py` | Lag- und Rolling-Features | `train_features_cyc.parquet` | `train_features_cyc_lag.parquet` | Einzeln oder via Pipeline |
| 5 | `model_dataset.py` | Zeitbasierter Split (Train/Val/Test) | Ergebnis aus 4C | `data/processed/<dataset_name>/train.parquet`, `val.parquet`, `test.parquet`, `meta.json` | Einzeln oder via Pipeline |
| 6 | `dataset_tft.py` | TFT-Datensatz erstellen | Schritt 5 | `dataset_spec.json` | Einzeln oder via Pipeline |
| 7 | `trainer_tft.py` | TFT-Training nach Config | Schritt 6 + `configs/models/tft/*.yaml` | Logs, Checkpoints, JSONs | Einzeln oder via Pipeline |
| 8 | `evaluate_tft.py` *(optional)* | Berechnet Fehlermaße (MAE, RMSE, MAPE, SMAPE) | Checkpoint + val/test.parquet | `results/tft/eval/<run_id>/eval_summary.json` | Nur einzeln |
| 9 | `aggregate_tft_eval.py` *(optional)* | Aggregiert alle eval_summary.json | Ordner aus 8 | `results/tft/eval/eval_overview.{csv,json}` | Nur einzeln |

---

## Detaillierte Outputs

### Schritt 7: `trainer_tft.py`

**Ausgabe-Struktur:**
```
logs/tft/run_YYYYMMDD_HHMMSS_<config>/
├── metrics.csv
├── hparams.yaml
└── ...

results/tft/runs/run_YYYYMMDD_HHMMSS_<config>/
├── checkpoints/
│   └── tft-epoch=XX-val_loss=Y.YYYY.ckpt
├── results.json        # Epochenweise Metriken
└── summary.json        # Aggregierte Trainings-Summary

results/pipeline_runs/  # Nur bei Pipeline-Aufruf
└── pipeline_YYYYMMDD_HHMMSS_manifest.json
```

---

## Aufruf-Beispiele

### Via Pipeline (empfohlen):

```bash
# Kompletter Run (Preprocessing + Training)
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/baseline.yaml

# Nur Preprocessing
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --steps preprocessing,model_dataset,dataset_tft

# Nur Training (Preprocessing bereits erledigt)
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/baseline.yaml \
    --steps training
```

### Einzeln (für Tests/Debugging):

```bash
# Preprocessing
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.load_raw
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.feature_engineering
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.cyclical_encoder
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.lag_features

# Modeling
python -m src.modeling.model_dataset
python -m src.modeling.dataset_tft
python -m src.modeling.trainer_tft --config configs/models/tft/baseline.yaml

# Evaluation (manuell)
python -m src.evaluation.evaluate_tft --run-id run_20251123_123456_baseline
python -m src.evaluation.aggregate_tft_eval
```

---

## Hinweise

- Schritt 1 ist **immer erforderlich** (lädt Rohdaten)
- Schritte 2–3 sind **optional** (abhängig vom Datensatz)
- Schritte 8–9 werden **nicht** automatisch ausgeführt (bewusst manuell)
- Pipeline-Manifests dokumentieren den kompletten Workflow
- Alte Arbeitsweise (einzelne Scripte) funktioniert parallel weiter

---