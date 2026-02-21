# TFT-TimeSeries

Multi-Modell Forecasting Framework für Zeitreihenprognosen mit **Temporal Fusion Transformer (TFT)**, **ARIMA** und **Prophet**.

Entwickelt im Rahmen einer FOM-Seminararbeit zum Thema *Zeitreihenprognosen im Consulting-Kontext: Entwicklung eines Frameworks zur Analyse und Bewertung unterschiedlicher Prognoseansätze*.

Das Framework ist dataset-agnostisch: Schritte von Preprocessing bis Evaluation werden über YAML-Konfigurationen gesteuert. Der Wechsel zwischen Datensätzen erfolgt ohne Code-Änderungen.

---

## 🧱 Architektur

Die drei Modellfamilien werden auf denselben vorverarbeiteten Daten trainiert und anschließend mit einheitlichen Metriken ausgewertet.

| Modell | Typ | Ansatz | Trainingsart |
|--------|-----|--------|--------------|
| **TFT** | Deep Learning | Attention-basiert, Multi-Horizon | Globales Modell |
| **ARIMA** | Statistisch | Auto-Regression + Moving Average | Modell pro Gruppe |
| **Prophet** | Statistisch | Decomposition (Trend + Seasonality) | Modell pro Gruppe |

Optional ist für jedes Modell eine Optuna-basierte Hyperparameter-Suche integrierbar (konfigurationsgetrieben).

---

## 🗂 Projektstruktur

```
TFT-TimeSeries/
├── configs/
│   ├── datasets/                  # Dataset-Definitionen (Schema, Preprocessing, Split)
│   └── models/                    # Modell-Konfigurationen (TFT/ARIMA/Prophet)
│
├── src/
│   ├── data/                      # Preprocessing (load, clean, features, lags)
│   ├── modeling/                  # Training + (optional) Optuna
│   ├── evaluation/                # Evaluation + Aggregation
│   ├── visualization/             # Plots (Cross-Model + modellspezifisch)
│   ├── utils/                     # Config-Loader, JSON-Export
│   ├── config.py                  # Projektweite Konstanten (Pfade, Defaults)
│   └── pipeline.py                # Orchestrierung
│
├── data/
│   ├── raw/<dataset>/
│   ├── interim/<dataset>/
│   └── processed/<dataset>/       # Splits + (TFT) dataset_spec.json
│
├── results/
│   ├── <model>/runs/<run_id>/     # Checkpoints, Summaries, Evaluation je Run
│   └── eval/                      # Cross-Model Aggregation (optional)
│
├── docs/                          # MkDocs-Dokumentation
├── mkdocs.yml                     # MkDocs-Konfiguration
└── README.md
```

---

## ✅ Voraussetzungen

- **Python 3.10+**
- Abhängigkeiten (Beispiele, nicht abschließend):
  - `torch`, `pytorch-lightning`, `pytorch-forecasting` (TFT)
  - `statsmodels`, `pmdarima` (ARIMA)
  - `prophet` (Prophet)
  - `optuna` (optional, HPO)
  - `pandas`, `numpy`, `pyyaml`

Dokumentation (optional):
- `mkdocs`, `mkdocs-material`, `pymdown-extensions`

---

## 🚀 Quickstart

### 1) Kompletter Durchlauf (Preprocessing + Training)

```bash
python -m src.pipeline   --dataset configs/datasets/<dataset>.yaml   --model configs/models/tft/<dataset>/<model_config>.yaml
```

### 2) Nur Preprocessing (einmalig pro Dataset)

```bash
python -m src.pipeline   --dataset configs/datasets/<dataset>.yaml   --steps preprocessing,model_dataset,dataset_tft
```

### 3) Nur Training (Preprocessing bereits erledigt)

```bash
python -m src.pipeline   --dataset configs/datasets/<dataset>.yaml   --model configs/models/tft/<dataset>/<model_config>.yaml   --steps training
```

### 4) ARIMA / Prophet (ohne Pipeline)

ARIMA und Prophet werden über eigene Trainer-Skripte ausgeführt. Typisch ist eine Dataset-Referenz über Umgebungsvariable oder CLI-Argumente (siehe Script-Übersicht in der Dokumentation).

---

## ⚙️ Config-System

Das Projekt trennt strikt zwischen:

- **Dataset-Config** (`configs/datasets/<dataset>.yaml`): Schema, Preprocessing, Split, dataset-nahe Einstellungen
- **Model-Config** (`configs/models/<model>/<dataset>/<name>.yaml`): Training und Modellparameter

Für TFT wird zusätzlich eine `dataset_spec.json` als Schnittstelle zwischen Preprocessing und Training generiert.

---

## 📚 Dokumentation (MkDocs)

Lokale Vorschau:

```bash
pip install mkdocs mkdocs-material pymdown-extensions
mkdocs serve
```

Navigation und Inhalte findest du unter:

- `docs/index.md`
- `mkdocs.yml`
- `docs/shared/` (übergreifend)
- `docs/project/` (modulspezifisch)

---

## 🔎 Weiterführende Dokumentation

- Script-Übersicht: `docs/shared/SCRIPTS.md`
- Pipeline-Reihenfolge: `docs/shared/PipelineOrder.md`
- Projektstruktur: `docs/shared/Projektstruktur.md`
- Konfigurationen: `docs/shared/ConfigSetup.md`
- Optuna (optional): `docs/shared/Optuna.md`
- Plots/Visualisierung: `docs/shared/PlottingAndVisualization.md`
