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
└── models/tft/        # Nach Dataset organisiert:
    ├── booksales/     #   - baseline.yaml, experiments/
    └── walmart/       #   - baseline.yaml, experiments/

data/
├── raw/<dataset>/     # Rohdaten (nicht versioniert)
├── interim/<dataset>/ # Zwischenschritte (raw, aligned, cleaned)
└── processed/<dataset>/ # Features, Splits (train/val/test)

logs/tft/              # Training-Logs: run_YYYYMMDD_HHMMSS_<dataset>_<config>/

results/tft/
├── runs/              # Checkpoints + Summaries: run_YYYYMMDD_HHMMSS_<dataset>_<config>/
├── eval/              # Evaluation-Ergebnisse
└── plots/             # Visualisierungen

docs/                  # Detaillierte Dokumentation
```

---

## 4. Pipeline – Ausführung

### 4.1 Kompletter Durchlauf (empfohlen)

```bash
# Alle Steps: Preprocessing + Modeling + Training
python -m src.pipeline \
    --dataset configs/datasets/walmart.yaml \
    --model configs/models/tft/walmart/baseline.yaml

python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/booksales/baseline.yaml
```

**Was passiert:** Führt automatisch alle Steps aus:
1. `preprocessing` (load_raw, alignment, cleaning, feature_engineering, cyclical_encoder, lag_features)
2. `model_dataset` (Train/Val/Test-Split)
3. `dataset_tft` (TFT-Spezifikation)
4. `training` (TFT-Training)

**Hinweis:** Wenn `--steps` nicht angegeben wird, werden alle Steps ausgeführt.

---

### 4.2 Nur Preprocessing

```bash
python -m src.pipeline \
    --dataset configs/datasets/walmart.yaml \
    --steps preprocessing,model_dataset,dataset_tft

python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --steps preprocessing,model_dataset,dataset_tft
```

**Nutzen:** Daten einmalig vorbereiten, danach verschiedene Modell-Configs trainieren.

---

### 4.3 Nur Training

```bash
# Training mit verschiedenen Hyperparametern (Preprocessing bereits erledigt)
python -m src.pipeline \
    --dataset configs/datasets/walmart.yaml \
    --model configs/models/tft/walmart/baseline.yaml \
    --steps training

python -m src.pipeline \
    --dataset configs/datasets/walmart.yaml \
    --model configs/models/tft/walmart/lr_high.yaml \
    --steps training
```

**Nutzen:** Verschiedene Modell-Konfigurationen testen ohne Preprocessing zu wiederholen.

---

### 4.4 Einzelne Schritte (manuell, für Debugging)

**Preprocessing:**
```bash
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.load_raw
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.feature_engineering
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.cyclical_encoder
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.lag_features
```

**Modeling:**
```bash
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.modeling.model_dataset
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.modeling.dataset_tft
python -m src.modeling.trainer_tft --config configs/models/tft/walmart/baseline.yaml
```

**Evaluation:**
```bash
python -m src.evaluation.evaluate_tft --run-id run_20251123_140000_walmart_baseline
python -m src.evaluation.aggregate_tft_eval
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

2. **Config erstellen (als Kopie einer bestehenden Config):**
   ```bash
   # Kopiere z.B. booksales.yaml als Vorlage
   Copy-Item configs/datasets/booksales.yaml configs/datasets/neuer_datensatz.yaml
   # Dann anpassen: name, paths, schema, preprocessing
   ```

3. **Pipeline ausführen:**
   ```bash
   python -m src.pipeline \
       --dataset configs/datasets/neuer_datensatz.yaml \
       --model configs/models/tft/baseline.yaml
   ```

**Details zur Config-Struktur:** Siehe bestehende Configs in `configs/datasets/` als Vorlage.

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
