# Pipeline – Orchestrierung und Workflow

**Datum:** 2025-11-21  
**Script:** `src/pipeline.py`  
**Ziel & Inhalt:** Beschreibung der zentralen Pipeline-Orchestrierung. Erklärt die verfügbaren Steps, deren Reihenfolge, Abhängigkeiten und praktische Anwendung für reproduzierbare Experimente.

---

## Überblick
Die **Pipeline** orchestriert alle Schritte von der Rohdatenverarbeitung bis zum Modelltraining in einer einheitlichen, konfigurierbaren Ausführungsumgebung.  
Sie nutzt die vorhandenen Module als Subprozesse und dokumentiert jeden Durchlauf in einem Manifest für vollständige Reproduzierbarkeit.

Eingabe: Dataset-Config + Model-Config (beide als YAML)  
Ausgabe: Manifest (`results/pipeline_runs/pipeline_YYYYMMDD_HHMMSS_manifest.json`)

---

## Ziel
Ziel ist die **automatisierte, reproduzierbare Ausführung** aller Verarbeitungsschritte:

- klare Trennung zwischen Dataset-Definition und Modell-Konfiguration  
- flexible Auswahl der auszuführenden Steps  
- vollständige Dokumentation jedes Durchlaufs  
- parallele Unterstützung für manuelle Einzelausführung und automatisierte Workflows

Die Pipeline ermöglicht es, komplette Experimente mit einem einzigen Befehl zu starten und alle Zwischenergebnisse konsistent zu dokumentieren.

---

## Verfügbare Steps

Die Pipeline kennt vier Haupt-Phasen, die über den Parameter `--steps` gesteuert werden:

### 1. `preprocessing`
**Zweck:** Rohdatenverarbeitung bis zu modellfertigen Features

**Umfasst alle aktivierten Steps aus der Dataset-Config:**
- `alignment` – Normalisierung der Verkaufsniveaus auf Referenzjahr
- `cleaning` – Bereinigung von Ausreißern und Lockdown-Effekten
- `feature_engineering` – Kalender-, Feiertags- und Zeitindex-Features
- `cyclical_encoder` – Zyklische Sin/Cos-Kodierung für Zeitmerkmale
- `lag_features` – Lag- und Rolling-Features für historische Abhängigkeiten

**Eingabe:** `data/raw/<dataset_name>/train.csv`  
**Ausgabe:** `data/processed/<dataset_name>/train_features_cyc_lag.parquet`

**Steuerung:** Einzelne Steps können in der Dataset-Config aktiviert/deaktiviert werden:
```yaml
preprocessing:
  - step: "alignment"
    enabled: true
  - step: "cleaning"
    enabled: true
  # ...
```

---

### 2. `model_dataset`
**Zweck:** Zeitbasierter Split in Train/Val/Test

**Funktionalität:**
- Liest die vorbereiteten Features ein
- Teilt die Daten nach Zeit in drei Segmente (Train/Val/Test)
- Optional: Gruppenweise Z-Standardisierung auf ausgewählte Spalten
- Speichert drei separate Parquet-Dateien plus Metadaten

**Eingabe:** `data/processed/<dataset_name>/train_features_cyc_lag.parquet`  
**Ausgabe:**
- `data/processed/<dataset_name>/train.parquet`
- `data/processed/<dataset_name>/val.parquet`
- `data/processed/<dataset_name>/test.parquet`
- `data/processed/<dataset_name>/meta.json`

**Steuerung:** Split-Methode und Grenzen in Dataset-Config:
```yaml
split:
  method: "ratio"  # oder "fixed"
  ratios: [0.80, 0.10, 0.10]
  # alternativ:
  # val_start: "2020-10-01"
  # test_start: "2020-11-01"
```

---

### 3. `dataset_tft`
**Zweck:** TFT-spezifische Datensatz-Spezifikation

**Funktionalität:**
- Analysiert die Splits (Train/Val/Test)
- Leitet Feature-Listen heuristisch ab:
  - `static_categoricals` (ID-Spalten)
  - `time_varying_known_reals` (zyklische Features, Kalender, Feiertage)
  - `time_varying_unknown_reals` (Target, Lags, sonstige numerische)
- Speichert Spezifikation für den Trainer

**Eingabe:** 
- `data/processed/<dataset_name>/{train,val,test}.parquet`
- TFT-Config aus Dataset-YAML

**Ausgabe:** `data/processed/<dataset_name>/dataset_spec.json`

**Steuerung:** Feature-Präfixe und Parameter in Dataset-Config:
```yaml
tft:
  max_encoder_length: 120
  max_prediction_length: 7
  known_real_prefixes: ["cyc_"]
  lag_prefixes: ["lag_"]
  treat_calendar_as_known: true
  flag_cols: ["is_lockdown_period"]
```

---

### 4. `training`
**Zweck:** TFT-Modelltraining mit PyTorch Lightning

**Funktionalität:**
- Lädt Dataset-Spezifikation und erstellt TimeSeriesDataSet
- Initialisiert TFT-Modell mit Hyperparametern aus Model-Config
- Führt Training mit Early Stopping und Checkpointing durch
- Exportiert Trainings-Metriken und Summary

**Eingabe:**
- `data/processed/<dataset_name>/dataset_spec.json`
- `configs/models/tft/<config_name>.yaml`

**Ausgabe:**
- Logs: `logs/tft/run_YYYYMMDD_HHMMSS_<config>/metrics.csv`
- Checkpoints: `results/tft/runs/run_YYYYMMDD_HHMMSS_<config>/checkpoints/*.ckpt`
- Summary: `results/tft/runs/run_YYYYMMDD_HHMMSS_<config>/summary.json`

**Steuerung:** Alle Parameter in Model-Config:
```yaml
training:
  seed: 42
  max_epochs: 30
  batch_size: 128
  learning_rate: 0.001
  early_stopping_patience: 5
  
model:
  hidden_size: 32
  attention_head_size: 4
  dropout: 0.1
  loss: "quantile"
```

---

## Aufrufbeispiele

### Kompletter Durchlauf (alle Steps)
```bash
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/baseline.yaml
```
**Entspricht:** `--steps preprocessing,model_dataset,dataset_tft,training`

---

### Nur Preprocessing
```bash
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --steps preprocessing,model_dataset,dataset_tft
```
**Nutzen:** Daten vorbereiten, mehrere Modelle nacheinander trainieren

---

### Nur Training (Preprocessing bereits erledigt)
```bash
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/baseline.yaml \
    --steps training
```
**Nutzen:** Verschiedene Hyperparameter-Konfigurationen testen

---

### Einzelne Schritte kombinieren
```bash
# Nur Feature Engineering + Modell-Vorbereitung
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --steps preprocessing,model_dataset

# Nur Datensatz-Spec + Training
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/baseline.yaml \
    --steps dataset_tft,training
```

---

## Pipeline-Manifest

Jeder Pipeline-Durchlauf erzeugt ein Manifest mit vollständiger Dokumentation:

**Speicherort:** `results/pipeline_runs/pipeline_YYYYMMDD_HHMMSS_manifest.json`

**Inhalt:**
```json
{
  "pipeline_run_id": "pipeline_20251121_223046",
  "timestamp": "2025-11-21T22:30:46",
  "dataset": {
    "name": "booksales",
    "config": { /* vollständige Dataset-YAML */ }
  },
  "model": {
    "type": "tft",
    "name": "baseline_v02",
    "config": { /* vollständige Model-YAML */ }
  },
  "execution": {
    "steps_requested": ["preprocessing", "model_dataset", "dataset_tft", "training"],
    "steps_executed": {
      "preprocessing": ["alignment", "cleaning", "feature_engineering", 
                       "cyclical_encoder", "lag_features"],
      "model_dataset": true,
      "dataset_tft": true,
      "training": {"run_id": "run_20251121_223309_baseline"}
    }
  }
}
```

---

## Abhängigkeiten zwischen Steps

```
preprocessing  →  model_dataset  →  dataset_tft  →  training
     │                 │                │               │
     │                 │                │               └─ benötigt: dataset_spec.json + model config
     │                 │                └─ benötigt: train/val/test.parquet
     │                 └─ benötigt: train_features_cyc_lag.parquet
     └─ benötigt: raw data + dataset config
```

**Wichtig:** Steps müssen in dieser Reihenfolge ausgeführt werden!  
Die Pipeline prüft dies nicht automatisch – bei fehlenden Eingaben bricht der entsprechende Step mit Fehlermeldung ab.

---

## Workflow-Empfehlungen

### Neues Experiment starten
```bash
# 1. Kompletter Run für initiales Training
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/baseline.yaml

# 2. Weitere Modell-Varianten (Preprocessing wird übersprungen)
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/lr_high.yaml \
    --steps training
```

---

### Neuer Datensatz
```bash
# 1. Preprocessing einmalig durchführen
python -m src.pipeline \
    --dataset configs/datasets/neuer_datensatz.yaml \
    --steps preprocessing,model_dataset,dataset_tft

# 2. Verschiedene Modelle trainieren
python -m src.pipeline \
    --dataset configs/datasets/neuer_datensatz.yaml \
    --model configs/models/tft/baseline.yaml \
    --steps training
```

---

### Preprocessing-Parameter ändern
```bash
# Preprocessing neu durchführen bei geänderten Feature-Einstellungen
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --steps preprocessing,model_dataset,dataset_tft

# Training mit neuen Features
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/baseline.yaml \
    --steps training
```

---

## Fehlerbehandlung

Die Pipeline gibt klare Fehlermeldungen bei:

- **Fehlende Eingabedateien:** "Datei nicht gefunden: data/raw/..."
- **Ungültige Config:** "Dataset-Config muss 'name' enthalten"
- **Step-Fehler:** "Pipeline-Schritt fehlgeschlagen: feature_engineering"

**Best Practice:**
- Bei Fehlern: Einzelne Steps manuell ausführen für detaillierte Fehleranalyse
- Logs prüfen: Terminal-Output zeigt alle Zwischenschritte
- Manifests prüfen: Dokumentieren erfolgreiche Ausführungen

---

## Evaluation (nicht Teil der Pipeline)

**Wichtig:** Die Evaluation erfolgt bewusst **manuell** nach dem Training:

```bash
# Nach Training: Evaluation durchführen
python -m src.evaluation.evaluate_tft --run-id run_20251121_223309_baseline

# Alle Evaluationen aggregieren
python -m src.evaluation.aggregate_tft_eval

# Plots erstellen
python -m src.visualization.plot_tft_eval_comparison --metric smape --split test
```

**Grund:** Evaluation ist oft explorativer Natur und soll nicht automatisch bei jedem Training erfolgen.

---

## Ergebnis und Nutzen

Die Pipeline ermöglicht:

- **Reproduzierbarkeit:** Vollständige Dokumentation aller Parameter und Schritte
- **Effizienz:** Einzelne Steps überspringen bei wiederholten Runs
- **Flexibilität:** Beliebige Kombinationen von Steps ausführen
- **Multi-Dataset:** Einfacher Wechsel zwischen Datensätzen
- **Experimente:** Systematisches Testen verschiedener Konfigurationen

Die modulare Struktur bleibt erhalten – jeder Step kann weiterhin einzeln für Debugging ausgeführt werden.