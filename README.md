# TFT-TimeSeries – Book Sales Forecasting

Dieses Repository enthält eine **modulare, erweiterbare Pipeline** zur Modellierung von Zeitreihen auf Basis des **Temporal Fusion Transformer (TFT)**. Der Fokus liegt auf einer klar strukturierten, konfigurationsgetriebenen und teamfähigen Umsetzung.

Als **Beispiel-Datensatz** dient der Kaggle-Datensatz **"Tabular Playground Series – Sep 2022" (Book Sales)**. Die Architektur ist so aufgebaut, dass weitere Datensätze einfach hinzugefügt werden können.

---

## 1. Datenbasis

Die Pipeline nutzt als **Beispiel** den Kaggle-Datensatz **"Tabular Playground Series – Sep 2022"**.

Erforderliche Dateien:

- `train.csv`
- optional: `test.csv`

Ablageort:

```
data/raw/booksales/
```

Rohdaten werden nicht versioniert.

**Erweiterbarkeit:** Die modulare Struktur erlaubt das einfache Hinzufügen weiterer Datensätze. Jeder Datensatz erhält einen eigenen Unterordner (`data/raw/<dataset_name>/`) und eine eigene Config (`configs/datasets/<dataset_name>.yaml`).

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
├── data/              # Preprocessing (Alignment, Cleaning, Features)
├── modeling/          # Training (model_dataset, dataset_tft, trainer_tft)
├── evaluation/        # Metriken-Berechnung und Aggregation
├── visualization/     # Plots für Daten, Training und Evaluation
├── utils/             # Hilfsfunktionen
├── config.py          # Globale Konstanten
└── pipeline.py        # Orchestrierung aller Schritte

configs/
├── datasets/          # booksales.yaml (Dataset-Config)
└── models/tft/        # baseline.yaml + Experimente

data/
├── raw/booksales/     # Rohdaten (nicht versioniert)
├── interim/booksales/ # Zwischenschritte (aligned, cleaned)
└── processed/booksales/ # Features, Splits (train/val/test)

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
# Kompletter Run (Preprocessing + Training)
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
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
python -m src.data.data_alignment
python -m src.data.data_cleaning
python -m src.data.feature_engineering
python -m src.data.cyclical_encoder
python -m src.data.lag_features
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

### `configs/datasets/booksales.yaml`
- Schema (Spaltennamen)
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

- Pipeline-Übersicht
- Script-Beschreibungen
- Workflow-Guides
- Konfigurationsreferenz

---

## 8. Geplante Erweiterungen

### Weitere Datensätze
- Modulare Struktur erlaubt einfache Integration neuer Datensätze
- Jeder Datensatz: eigene Config + eigene Unterordner
- Beispiele: Retail Sales, E-Commerce, Energy Consumption

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

## 9. Zusammenarbeit

Die modulare Struktur ermöglicht paralleles Arbeiten und reproduzierbare Ergebnisse.

Alle Schritte sind konfigurationsgetrieben und klar dokumentiert. Änderungen werden minimalinvasiv umgesetzt, sodass die Gesamtstruktur stabil bleibt.