# TFT-TimeSeries – Book Sales Forecasting

Dieses Repository enthält eine modulare, reproduzierbare Pipeline zur Modellierung von Zeitreihen auf Basis des **Temporal Fusion Transformer (TFT)**. Der Fokus liegt auf einer klar strukturierten, konfigurationsgetriebenen und teamfähigen Umsetzung.

---

## 1. Datenbasis

Die Pipeline nutzt den Kaggle-Datensatz **"Tabular Playground Series – Sep 2022"**.

Erforderliche Dateien:

- `train.csv`
- optional: `test.csv`

Ablageort:

```
data/raw/
```

Rohdaten werden nicht versioniert.

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

Bei Bedarf:

```bash
pip freeze > requirements.txt
```

---

## 3. Projektstruktur

```text
src/
├── data/
│   ├── data_alignment.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── cyclical_encoder.py
│   ├── lag_features.py
│   └── view_data.py
│
├── modeling/
│   ├── model_dataset.py
│   ├── dataset_tft.py
│   ├── trainer_tft.py
│   ├── evaluate_tft.py
│   ├── forecasting_interface.py
│   └── diagnostics/
│
├── utils/
│   ├── config_loader.py
│   ├── json_results.py
│   └── load_trained_tft.py
│
├── visualization/
│   ├── plot_tft_forecast_series.py
│   ├── plot_training_curves.py
│   └── plot_data_overview.py
│
└── config.py
```

Weitere Verzeichnisse:

```text
configs/
│   ├── trainer_tft_baseline.yaml
│   ├── trainer_tft_optuna.yaml
│   └── * weitere Experimente *

data/
├── raw/
├── interim/
└── processed/

logs/
└── tft/run_*/

results/
├── tft/
│   ├── checkpoints/
│   └── runs/
└── evaluation/

docs/
└── MkDocs-Dokumentation
```

---

## 4. Pipeline – Ausführung

### 4.1 Datenaufbereitung

```bash
python -m src.data.data_cleaning
python -m src.data.data_alignment
python -m src.data.feature_engineering
python -m src.data.cyclical_encoder
python -m src.data.lag_features
```

### 4.2 Modell-Dataset

```bash
python -m src.modeling.model_dataset
```

### 4.3 TFT-Dataset

```bash
python -m src.modeling.dataset_tft
```

### 4.4 Training

```bash
python -m src.modeling.trainer_tft --config configs/trainer_tft_baseline.yaml
```

---

## 5. Konfiguration

### `config.py`
- Datei- und Verzeichnisstruktur  
- Feature-Definitionen  
- Sequenzlängen  
- Spaltennamen  

### YAML-Konfigurationen
Steuern Training, Modellarchitektur, Hardware-Parameter und Logging.

---

## 6. Logging & Ergebnisse

- Trainingsmetriken:  
  `logs/tft/run_*/metrics.csv`
- Checkpoints:  
  `results/tft/runs/<run_id>/checkpoints/`
- Evaluation:  
  `results/evaluation/<run_id>/summary.json`

---

## 7. Dokumentation (MkDocs)

Start:

```bash
mkdocs serve
```

Dokumentation:

```
docs/
```

---

## 8. Geplante Erweiterungen

### Klassische Modelle
- Integration von **ARIMA** und **Prophet**  
- Vergleich der Modelle mit TFT hinsichtlich Forecast-Horizont und Fehlermaßen

### Hyperparameter-Optimierung
- Einbindung von **Optuna** mit:
  - Random Search → 20–40 Trials  
  - TPE Bayesian Optimization → 50–100 Trials  
  - Pruning (MedianPruner / SuccessiveHalving)  
- Speicherung der Studien als `.pkl`

### MLflow
- Logging von Parametern und Trainingsmetriken  
- Versionierung von Modellen  
- Optionale Model Registry

---

## 9. Zusammenarbeit

Die modulare Struktur ermöglicht paralleles Arbeiten und reproduzierbare Ergebnisse.  
Alle Schritte sind klar dokumentiert und können über Konfigurationen gesteuert werden.

Die Entwicklung erfolgt in einer klar strukturierten, konfigurationsbasierten Pipeline.
Alle Skripte folgen einem einheitlichen Aufbau, verwenden ausschließlich zentrale Konfigurationen und vermeiden unnötige Komplexität. Änderungen werden minimalinvasiv umgesetzt, sodass die Gesamtstruktur stabil bleibt. Die Dokumentation der einzelnen Module unterstützt ein gemeinsames, reproduzierbares Arbeiten und ermöglicht eine einheitliche Erweiterung der Pipeline (z. B. durch ARIMA, Prophet, Optuna oder MLflow).

