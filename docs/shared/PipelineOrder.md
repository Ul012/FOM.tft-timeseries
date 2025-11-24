# Pipeline Overview – TFT-TimeSeries

**Datum:** 2025-11-24 (aktualisiert)  
**Script:** –  
**Ziel & Inhalt:** Vollständige Übersicht über die Pipeline-Reihenfolge. Beschreibt Input, Output und Zweck aller Module von Rohdaten-Laden bis Training und Evaluation.

---

## Ziel

Diese Übersicht beschreibt die **Ausführungsreihenfolge** der zentralen Module – von Rohdaten bis Training und Evaluation.  
Alle Schritte können einzeln getestet werden. Schritte 1–7 bilden die Hauptpipeline.

---

## Projektlogik

| Ebene | Ordner | Aufgabe |
|-------|--------|---------|
| **data** | `src/data/` | Laden, Bereinigen und Feature-Erzeugung |
| **modeling** | `src/modeling/` | Splitten, TFT-Datensatz, Modelltraining |
| **evaluation** | `src/evaluation/` | Ergebnisanalyse und Kennzahlen |
| **visualization** | `src/visualization/` | Plots für Daten, Training und Evaluation |
| **utils** | `src/utils/` | Hilfsfunktionen – **nicht Teil der Pipeline** |
| **pipeline** | `src/pipeline.py` | Orchestrierung aller Schritte |

---

## Pipeline-Reihenfolge

| # | Modul | Beschreibung | Input | Output |
|--:|-------|--------------|-------|--------|
| 1 | `load_raw.py` | Laden und Mergen von Rohdaten | `data/raw/<dataset>/*.csv` | `train_raw.parquet` |
| 2 | `data_alignment.py` | Skaliert/normalisiert Zeitreihen *(optional)* | Schritt 1 | `train_aligned.parquet` |
| 3 | `data_cleaning.py` | Bereinigt Target (Clip, NaN, dtype) | Schritt 1 oder 2 | `train_cleaned.parquet` |
| 4A | `feature_engineering.py` | Kalender-Features, time_idx, Feiertage | Schritt 3 | `train_features.parquet` |
| 4B | `cyclical_encoder.py` | Zyklische Sin/Cos-Kodierungen | Schritt 4A | `train_features_cyc.parquet` |
| 4C | `lag_features.py` | Lag-/Rolling-Features, Gruppen-Filter, NaN-Imputation | Schritt 4B | `train_features_cyc_lag.parquet` |
| 5 | `model_dataset.py` | Zeitbasierter Split (Train/Val/Test) | Schritt 4C | `train.parquet`, `val.parquet`, `test.parquet` |
| 6 | `dataset_tft.py` | TFT-Datensatz-Spezifikation | Schritt 5 | `dataset_spec.json` |
| 7 | `trainer_tft.py` | TFT-Training | Schritt 6 + Model-YAML | Checkpoints, Logs, JSONs |
| 8 | `evaluate_tft.py` | Berechnet Fehlermaße *(optional)* | Checkpoint + Daten | `eval_summary.json` |
| 9 | `aggregate_tft_eval.py` | Aggregiert Evaluierungen *(optional)* | Schritt 8 | `eval_overview.csv` |

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
results/tft/runs/<run_id>/
    checkpoints/, results.json, summary.json
```

---

## Aufgabenverteilung im Preprocessing

| Aufgabe | Modul | Beschreibung |
|---------|-------|--------------|
| Target clippen | `data_cleaning.py` | Negative Werte auf Minimum setzen |
| Target dtype | `data_cleaning.py` | Konvertierung zu float32 |
| Target-NaN entfernen | `data_cleaning.py` | Zeilen mit NaN im Target entfernen |
| Lag-NaN Imputation | `lag_features.py` | Median + Missing-Indicator |
| Gruppen filtern | `lag_features.py` | Zu kurze Zeitreihen entfernen |

---

## Aufruf-Beispiele

### Via Pipeline (empfohlen)

```bash
# Kompletter Run (Preprocessing + Training)
python -m src.pipeline \
    --dataset configs/datasets/<dataset>.yaml \
    --model configs/models/tft/<dataset>/<config>.yaml

# Nur Preprocessing
python -m src.pipeline \
    --dataset configs/datasets/<dataset>.yaml \
    --steps preprocessing,model_dataset,dataset_tft

# Nur Training (Preprocessing bereits erledigt)
python -m src.pipeline \
    --dataset configs/datasets/<dataset>.yaml \
    --model configs/models/tft/<dataset>/<config>.yaml \
    --steps training
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
python -m src.modeling.dataset_tft
python -m src.modeling.trainer_tft --config configs/models/tft/<dataset>/<config>.yaml

# Evaluation (manuell)
python -m src.evaluation.evaluate_tft --run-id <run_id>
python -m src.evaluation.aggregate_tft_eval
```

---

## Output-Struktur

```
logs/tft/<run_id>/
├── metrics.csv
└── hparams.yaml

results/tft/runs/<run_id>/
├── checkpoints/
│   └── tft-epoch=XX-val_loss=Y.YYYY.ckpt
├── results.json
└── summary.json

results/pipeline_runs/  # Nur bei Pipeline-Aufruf
└── pipeline_<timestamp>_manifest.json
```

---

## Hinweise

- Schritt 1 ist **immer erforderlich** (lädt Rohdaten)
- Schritte 2–3 sind **optional** (abhängig vom Datensatz)
- Schritte 8–9 werden **nicht** automatisch ausgeführt (bewusst manuell)
- Pipeline-Manifests dokumentieren den kompletten Workflow
- Einzelne Scripts funktionieren parallel zur Pipeline

---