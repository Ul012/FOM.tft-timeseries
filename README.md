# TFT-TimeSeries – Multi-Dataset Forecasting

Dieses Repository enthält eine **modulare, erweiterbare Pipeline** zur Modellierung von Zeitreihen auf Basis des **Temporal Fusion Transformer (TFT)**. Der Fokus liegt auf einer klar strukturierten, konfigurationsgetriebenen und teamfähigen Umsetzung.

Als **Beispiel-Datensätze** dienen:
- **Booksales** (Kaggle Tabular Playground Series – Sep 2022)
- **Walmart** (Kaggle Store Sales Forecasting)

Die Architektur ist so aufgebaut, dass weitere Datensätze einfach hinzugefügt werden können.

---

## 1. Datenbasis

Die Pipeline unterstützt **mehrere Datensätze** gleichzeitig.

**Beispiel: Booksales**
- Dateien: `train.csv`, optional `test.csv`
- Ablageort: `data/raw/booksales/`

**Beispiel: Walmart**
- Dateien: `train.csv`, `features.csv`, optional `test.csv`
- Ablageort: `data/raw/walmart/`

Rohdaten werden nicht versioniert.

**Erweiterbarkeit:** Jeder Datensatz erhält einen eigenen Unterordner (`data/raw/<dataset_name>/`) und eine eigene Config (`configs/datasets/<dataset_name>.yaml`).

---

## 2. Installation und Setup

### 2.1 Virtuelle Umgebung

```bash
python -m venv .venv
```

Aktivierung:

- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

### 2.2 Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

---

## 3. Projektstruktur

```text
src/
├── data/              # Preprocessing (Load, Alignment, Cleaning, Features)
├── modeling/          # Training (model_dataset, dataset_tft, trainer_tft)
├── evaluation/        # Metriken-Berechnung und Aggregation
├── visualization/     # Plots für Daten, Training und Evaluation
├── utils/             # Hilfsfunktionen
├── config.py          # Globale Konstanten
└── pipeline.py        # Orchestrierung aller Schritte

configs/
├── datasets/          # booksales.yaml, walmart.yaml (Dataset-Configs)
└── models/tft/        # baseline.yaml + Experimente

data/
├── raw/<dataset>/     # Rohdaten (nicht versioniert)
├── interim/<dataset>/ # Zwischenschritte (raw, aligned, cleaned)
└── processed/<dataset>/ # Features, Splits (train/val/test)

logs/tft/              # Training-Logs (metrics.csv)

results/tft/
├── runs/              # Checkpoints + Training-Summaries
├── eval/              # Evaluation-Ergebnisse
└── plots/             # Visualisierungen

docs/                  # Detaillierte Dokumentation
```

---

## 4. Pipeline – Ausführung

### 4.1 Via Pipeline (empfohlen)

```bash
# Kompletter Run (Preprocessing + Training) - Booksales
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/baseline.yaml

# Kompletter Run - Walmart
python -m src.pipeline \
    --dataset configs/datasets/walmart.yaml \
    --model configs/models/tft/baseline.yaml

# Nur Preprocessing
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --steps preprocessing,model_dataset,dataset_tft

# Nur Training (wenn Preprocessing bereits erledigt)
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/baseline.yaml \
    --steps training
```

### 4.2 Einzelne Schritte (manuell)

**Preprocessing:**
```bash
# Mit Umgebungsvariable (für einzelne Schritte)
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.load_raw
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.feature_engineering
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.cyclical_encoder
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.lag_features
```

**Modeling:**
```bash
python -m src.modeling.model_dataset
python -m src.modeling.dataset_tft
python -m src.modeling.trainer_tft --config configs/models/tft/baseline.yaml
```

**Evaluation:**
```bash
python -m src.evaluation.evaluate_tft --run-id <run_id>
python -m src.evaluation.aggregate_tft_eval
```

**Visualization:**
```bash
python -m src.visualization.live_loss_plot --run_dir logs/tft/<run_id>
python -m src.visualization.plot_tft_eval_comparison --metric smape --split test
```

---

## 5. Konfiguration

### `configs/datasets/<dataset_name>.yaml`
- Schema (Spaltennamen, time_col, id_cols, target_col)
- Raw Data Loading (single_file vs. multiple_files mit Merge)
- Preprocessing-Pipeline (Steps aktivieren/deaktivieren)
- Split-Konfiguration
- TFT-Parameter

### `configs/models/tft/*.yaml`
- Training-Hyperparameter (Epochen, Batch-Size, Learning Rate)
- Modell-Architektur (hidden_size, dropout, etc.)
- Hardware-Parameter (GPU/CPU, num_workers)

### `src/config.py`
- Verzeichnis-Struktur
- Projekt-übergreifende Konstanten

---

## 6. Logging & Ergebnisse

- **Training-Logs:**  
  `logs/tft/<run_id>/metrics.csv`

- **Checkpoints:**  
  `results/tft/runs/<run_id>/checkpoints/`

- **Training-Summary:**  
  `results/tft/runs/<run_id>/summary.json`

- **Evaluation:**  
  `results/tft/eval/<run_id>/eval_summary.json`  
  `results/tft/eval/eval_overview.csv`

- **Plots:**  
  `results/tft/plots/{data,training,eval}/`

---

## 7. Dokumentation

Detaillierte Dokumentation in `docs/`:

- LoadRaw.md – Laden und Mergen von Rohdaten
- DataAlignment.md – Normalisierung
- DataCleaning.md – Bereinigung
- FeatureEngineer.md – Kalender-Features
- CyclicalEncoder.md – Sin/Cos-Kodierung
- LagFeatures.md – Lag- und Rolling-Features
- Pipeline.md – Orchestrierung
- PipelineOrder.md – Reihenfolge aller Schritte
- ConfigSetup.md – Konfigurationssystem
- Projektstruktur.md – Vollständiger Überblick

---

## 8. Multi-Dataset Support

### Neuen Datensatz hinzufügen:

1. **Rohdaten ablegen:**
   ```
   data/raw/neuer_datensatz/
   ├── train.csv
   └── ...
   ```

2. **Config erstellen:**
   ```yaml
   # configs/datasets/neuer_datensatz.yaml
   name: "neuer_datensatz"
   
   raw_data:
     type: "single_file"  # oder "multiple_files"
     files:
       - path: "data/raw/neuer_datensatz/train.csv"
         role: "main"
   
   schema:
     time_col: "date"
     id_cols: ["id1", "id2"]
     target_col: "value"
   
   preprocessing:
     - step: "load_raw"
       enabled: true
     # ... weitere Steps
   ```

3. **Pipeline ausführen:**
   ```bash
   python -m src.pipeline \
       --dataset configs/datasets/neuer_datensatz.yaml \
       --model configs/models/tft/baseline.yaml
   ```

---

## 9. Geplante Erweiterungen

### Klassische Modelle
- Integration von **ARIMA** und **Prophet**
- Vergleich mit TFT

### Hyperparameter-Optimierung
- **Optuna** mit Random Search, TPE, Pruning
- Studien-Persistierung

### MLflow
- Tracking von Experimenten
- Model Registry

---

## 10. Zusammenarbeit

Die modulare Struktur ermöglicht paralleles Arbeiten und reproduzierbare Ergebnisse.

Alle Schritte sind konfigurationsgetrieben und klar dokumentiert. Änderungen werden minimalinvasiv umgesetzt, sodass die Gesamtstruktur stabil bleibt.

---

## 11. Aktueller Stand 23.11.2025

✅ **Komplett generalisiert:**
- load_raw.py (Single-File + Multi-File mit Merge)
- data_cleaning.py (Outlier-Dates + Lockdown aus YAML)
- feature_engineering.py (Country-spezifische Feiertage)
- cyclical_encoder.py (Periodicities aus YAML)
- lag_features.py (Bereits generalisiert)
- pipeline.py (Dataset-Config-Weitergabe)

✅ **Funktionierende Datensätze:**
- Booksales (tägliche Daten, 3 ID-Spalten)
- Walmart (wöchentliche Daten, 2 ID-Spalten, Merge von 2 CSVs)

🔄 **In Arbeit:**
- model_dataset.py (Split generalisieren)
- dataset_tft.py (TFT-Features generalisieren)
- trainer_tft.py (Training anpassen)