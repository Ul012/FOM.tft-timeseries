# TFT-TimeSeries – Book Sales Forecasting

Dieses Repository enthält eine modulare, reproduzierbare Pipeline zur Modellierung von Zeitreihen mit Schwerpunkt auf dem **Temporal Fusion Transformer (TFT)**.  
Die Pipeline umfasst Datenaufbereitung, Feature Engineering, Datensplitting, Erstellung eines TFT-Datasets sowie Training, Logging und Evaluierung.

Das Projekt ist so aufgebaut, dass alle Schritte klar strukturiert, minimalinvasiv und konfigurationsgetrieben sind.  
Der Code folgt Clean-Code-Prinzipien und vermeidet versteckte Defaults.

---

## 1. Datenbasis

Für das Projekt wird der Kaggle-Datensatz **"Tabular Playground Series – Sep 2022"** verwendet.

**Download-Link:**  
https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/data

Die Datei muss manuell in folgendes Verzeichnis gelegt werden:

```
data/raw/
```

Erforderlich für die Pipeline:

- `train.csv`
- optional: `test.csv`

Es wird empfohlen, die Rohdaten nicht im Repository zu versionieren.

---

## 2. Installation und Setup

### 2.1 Python-Version

Python **3.12.0** wird empfohlen.

### 2.2 Virtuelle Umgebung erstellen

```bash
python -m venv .venv
```

Aktivierung:

- Windows  
  ```bash
  .venv\Scripts\activate
  ```
- macOS / Linux  
  ```bash
  source .venv/bin/activate
  ```

### 2.3 Abhängigkeiten installieren

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Falls die Datei noch nicht existiert:

```bash
pip freeze > requirements.txt
```

Für PyTorch kann eine systemspezifische Installation notwendig sein:  
https://pytorch.org/get-started/locally/

---

## 3. Projektstruktur

```text
src/
├── data/                # Datenverarbeitung
│   ├── data_alignment.py
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── cyclical_encoder.py
│   ├── lag_features.py
│   └── view_data.py
├── modeling/            # Datensplitting, TFT-Dataset, Training
│   ├── model_dataset.py
│   ├── dataset_tft.py
│   └── trainer_tft.py
├── utils/               # YAML-Loader, JSON-Erzeugung
│   ├── config_loader.py
│   ├── json_results.py
│   └── load_trained_tft.py
├── visualization/       # Plot-Module
└── config.py            # Statische Konstanten
```

Weitere Ordner:

```text
data/                    # Daten
├── raw/
├── interim/
└── processed/

logs/                    # Training Logs
└── tft/run_*/

results/                 # Modelle und Evaluierungen
├── tft/checkpoints/
└── evaluation/
```

---

## 4. Pipeline – Ausführungsreihenfolge

### 4.1 Datenbereinigung und Feature Engineering

```bash
python -m src.data.data_cleaning
python -m src.data.data_alignment
python -m src.data.feature_engineering
python -m src.data.cyclical_encoder
python -m src.data.lag_features
```

Erzeugt:

```
data/processed/train_features_cyc_lag.parquet
```

---

### 4.2 Modell-Dataset erstellen

```bash
python -m src.modeling.model_dataset
```

Erzeugt:

- `train.parquet`  
- `val.parquet`  
- `test.parquet`  
- `meta.json`

---

### 4.3 TFT-Dataset konfigurieren

```bash
python -m src.modeling.dataset_tft
```

Erzeugt:

- `dataset_spec.json` (Feature-Zuordnung und Sequenzlängen)

---

### 4.4 Training

```bash
python -m src.modeling.trainer_tft --config configs/trainer_tft_baseline.yaml
```

Ergebnisse:

- Trainingsmetriken → `logs/tft/run_*/metrics.csv`
- Checkpoints → `results/tft/checkpoints/run_*/`
- Evaluation → `results/evaluation/run_*/summary.json`

---

## 5. Konfiguration

### 5.1 Statische Projektkonfiguration (`src/config.py`)

Enthält:

- Datenpfade  
- Spaltennamen  
- Split-Logik  
- Lag-Konfiguration  
- TFT-Dataset-Struktur  

Diese Werte definieren die Projektstruktur und ändern sich selten.

### 5.2 Trainingskonfiguration (`configs/*.yaml`)

Steuert:

- Trainingseinstellungen  
- Modellarchitektur  
- Hardware-Konfiguration  

Jeder Trainingslauf wird ausschließlich über YAML-Dateien gesteuert.

---

## 6. Logging und Ergebnisse

### 6.1 Training Logs

```
logs/tft/run_*/metrics.csv
```

### 6.2 Modell-Checkpoints

```
results/tft/checkpoints/run_*/
```

### 6.3 Evaluationsdateien

```
results/evaluation/run_*/summary.json
results/evaluation/run_*/results.json
```

Diese Dateien dienen zur Analyse und zum späteren Modellvergleich.

---

## 7. Dokumentation mit MkDocs

Das Projekt verwendet **MkDocs**, um eine klare, durchsuchbare und strukturierte Dokumentation bereitzustellen.

Die Dokumentation beschreibt u. a.:

- den vollständigen Pipeline-Ablauf  
- alle Skripte der Daten- und Modeling-Schicht  
- Konfigurationslogik (`config.py` + YAML)  
- Datenstruktur und verwendete Features  
- Beispiele für Trainingsläufe  
- Hinweise zu Logs, Ergebnissen und Best Practices  

Start der lokalen Dokumentation:

```bash
mkdocs serve
```

Der Source-Code der Dokumentation befindet sich im Ordner:

```
docs/
```

---

## 8. Lizenz / Nutzungshinweise

- Die Daten stammen von Kaggle und unterliegen den entsprechenden Lizenzbestimmungen.  
- Der Code kann erweitert oder angepasst werden.  
- Rohdaten sollten nicht im Repository abgelegt werden.

---

## 9. Zusammenarbeit

Die Pipeline wurde so gestaltet, dass alle Nutzer jederzeit reproduzierbare Ergebnisse erzeugen können.  
Durch klare Konfiguration, modulare Skripte und dokumentierte Abläufe ist ein gemeinsames Arbeiten problemlos möglich.
