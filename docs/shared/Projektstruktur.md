# Projektstruktur – FOM.tft-timeseries

**Datum:** 2025-11-16  
**Script:** –  
**Ziel & Inhalt:** Gibt eine vollständige Übersicht über die Struktur des gesamten Projekts. Erklärt die Rollen der Ordner `data`, `modeling`, `utils`, `visualization` sowie geplante Evaluation. Beschreibt Datenfluss, Zuständigkeiten und Erweiterbarkeit der Pipeline.


---

## 🗂️ 1. `src/` – Hauptverzeichnis

```text
src/
├── data/
├── modeling/
├── evaluation/
├── utils/
├── visualization/
└── config.py
```

---

## 📊 2. `data/` – Datenaufbereitung (Preprocessing)

Beinhaltet alle Schritte bis zur Erstellung eines modellfertigen Datensatzes.

| Datei | Aufgabe |
|-------|---------|
| `data_alignment.py` | Harmonisierung und optionale Normalisierung der Zeitachsen. |
| `data_cleaning.py` | Bereinigung, Imputation, Konsistenzprüfungen. |
| `feature_engineering.py` | Erstellung von Kalender- und Feiertags-Features. |
| `cyclical_encoder.py` | Zyklische Kodierung periodischer Variablen (sin/cos). |
| `lag_features.py` | Erzeugt Lag- und Rolling-Features per `groupby().shift()`. |
| `view_data.py` | Kurze visuelle Kontrolle der Roh- und Zwischendaten. |

**Ausgabe dieser Stufe:**  
- `data/processed/train_features_cyc_lag.parquet` (Features inkl. Zyklen und Lags)  
→ dient als Input für `model_dataset.py`.

---

## 🤖 3. `modeling/` – Modellierung und Training

Enthält alle Skripte zur Vorbereitung, Spezifikation und zum Training der Modelle.

| Datei | Aufgabe |
|-------|---------|
| `model_dataset.py` | Split in Train/Validation/Test, schreibt `train/val/test.parquet` und `meta.json`. |
| `dataset_tft.py` | Leitet Feature-Listen (known/unknown/static) ab, erstellt `dataset_spec.json`. |
| `trainer_tft.py` | Trainiert den Temporal Fusion Transformer, speichert Logs, Checkpoints und JSON-Reports. |
| *(geplant)* `trainer_arima.py` | ARIMA-Modelltraining auf aggregierten oder einzelnen Zeitreihen. |
| *(geplant)* `trainer_prophet.py` | Prophet-Training mit automatischer Saisonalitätserkennung. |

**Wichtige Datenflüsse:**

1. `model_dataset.py`:
   - Eingabe: `data/processed/train_features_cyc_lag.parquet`
   - Ausgabe:  
     - `data/processed/train.parquet`  
     - `data/processed/val.parquet`  
     - `data/processed/test.parquet`  
     - `data/processed/meta.json`

2. `dataset_tft.py`:
   - Eingabe: `train/val/test.parquet` aus `data/processed/`
   - Ausgabe: `data/processed/dataset_spec.json` (TFT-Datensatzspezifikation)

3. `trainer_tft.py`:
   - Eingabe: `dataset_spec.json` + YAML aus `configs/`
   - Ausgabe:
     - Logs: `logs/tft/run_YYYYMMDD_HHMMSS/metrics.csv`, `hparams.yaml`, …
     - Checkpoints: `results/tft/runs/run_YYYYMMDD_HHMMSS/checkpoints/*.ckpt`
     - Evaluations-JSONs: `results/tft/eval/run_YYYYMMDD_HHMMSS/{results,summary}.json`

---

## 4. evaluation/ – Kennzahlen für fertige Modelle

| Datei                   | Aufgabe                                                                |
| ----------------------- | ---------------------------------------------------------------------- |
| `evaluate_tft.py`       | Lädt ein TFT-Checkpoint und berechnet Fehlermaße auf Val/Test.         |
| `aggregate_tft_eval.py` | Aggregiert alle `eval_summary.json` zu einer tabellarischen Übersicht. |

**Ablage:**

results/
└─ tft/
   ├─ runs/              # Artefakte aus dem Training (Checkpoints, Summaries)
   ├─ eval/              # Ausgaben der Evaluationsskripte
   │  ├─ <run_id>/eval_summary.json
   │  └─ eval_overview.{csv,json}
   └─ plots/             # Plots für Training und Evaluation
      └─ eval/...

---

## 📈 5. `visualization/` – Plots und Diagnosen (Evaluationsebene)

Fasst alle Visualisierungen zusammen, die nach oder während des Trainings benötigt werden.

| Datei                         | Aufgabe                                                             |
| ----------------------------- | ------------------------------------------------------------------- |
| `data_alignment_plot.py`      | Visualisierung der Datenharmonisierung.                             |
| `data_cleaning_plot.py`       | Darstellung der Bereinigung (Vorher/Nachher).                       |
| `plot_learning_rate.py`       | Lernkurven und ggf. Lernratenverlauf aus dem Training.              |
| `plot_tft_eval_comparison.py` | Vergleich einer Kennzahl (z. B. Test-SMAPE) über mehrere Runs.      |
| `plot_tft_forecast_series.py` | Ist vs. Prognose für eine Beispiel-Zeitreihe mit Forecast-Horizont. |

Plots werden typischerweise unter `results/tft/plots/` abgelegt.

---

## 🧰 6. `utils/` – Hilfsfunktionen & Werkzeuge

Dient zur Wiederverwendung und modularen Wartung.

| Datei | Aufgabe |
|-------|---------|
| `config_loader.py` | Lädt und validiert YAML-Konfigurationen für den Trainer. |
| `json_results.py` | Aggregiert Metriken aus `metrics.csv` und exportiert JSON-Ergebnisse pro Run. |
| `load_trained_tft.py` | Utility zum Laden eines gespeicherten TFT-Checkpoints (optional). |
| `__init__.py` | Kennzeichnung als Paket; ggf. globale Utility-Imports. |

> Utils-Skripte werden meist importiert und nicht direkt als Pipeline-Schritt ausgeführt.

---


## ⚙️ 7. `config.py` – Zentrale Steuerung

- Globale Konstanten: `DATETIME_COLUMN`, `GROUP_COLS`, `TARGET_COL`  
- Pfade: `RAW_DIR`, `INTERIM_DIR`, `PROCESSED_DIR`, `MODEL_INPUT_PATH`  
- Feature-Konfigurationen: `LAG_CONF`, `TFT_DATASET` (Sequenzlängen, Prefixes, Flags)  
- Split-Parameter: `VAL_START`, `TEST_START`, `SPLIT_RATIOS`

**Wichtig:** Trainings- und Modell-Hyperparameter (Learning Rate, Batch Size, Epochen, Modellgrößen, Devices etc.) stehen **nicht** in `config.py`, sondern in den YAML-Dateien im Ordner `configs/`.

---

## ✅ 8. Einordnung und Erweiterbarkeit

- **Hauptpipeline:**  
  `src/data/` → `src/modeling/` → `src/evaluation/` (Datenaufbereitung bis Evaluation)

- **Unterstützend, optional:**  
  `src/utils/`, `src/visualization/`

- **Erweiterbar:**  
  Zusätzliche Trainer-Module (z. B. `trainer_arima.py`, `trainer_prophet.py`) können nach demselben Muster aufgebaut werden wie `trainer_tft.py`:
  - trainieren Modelle,
  - loggen Metriken in `logs/<modell>/run_*`,
  - schreiben Ergebnisse nach `results/<modell>/...`.

Damit bleibt das Projekt trotz Erweiterungen (mehr Modelle, mehr Szenarien) **übersichtlich, modular und gut dokumentierbar**.
