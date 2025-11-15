# Integration von ARIMA und Prophet in die TFT-TimeSeries-Pipeline

**Datum:** 2025-11-15  
**Script:** –  
**Ziel & Inhalt:** Beschreibt, wie ARIMA und Prophet als zusätzliche Modelle in die bestehende TFT-Pipeline integriert werden sollen. Definiert Trainer-Skripte, Datenbasis, Ordnerstruktur, Konfiguration sowie die geplante Evaluationslogik für den späteren Modellvergleich.


Die Integration erfolgt so, dass:

- die bestehende Datenpipeline unverändert wiederverwendet wird,
- alle Modelle dieselben `train/val/test`-Splits nutzen,
- Training (Trainer-Skripte) und Auswertung (Evaluations-Skripte) klar getrennt bleiben.

---

## 1. Zielsetzung

- Erweiterung der bestehenden Modeling-Schicht um zusätzliche Trainer:
  - `trainer_arima.py`
  - `trainer_prophet.py`
- Nutzung der vorhandenen Datenpipeline:
  - keine neue Preprocessing-Logik
  - Wiederverwendung der Splits (`train/val/test`) aus `model_dataset.py`
- Vorbereitung für einen konsistenten Modellvergleich:
  - TFT vs. ARIMA vs. Prophet
  - gemeinsame Evaluationslogik in separaten Skripten

---

## 2. Einordnung in die Projektstruktur

Neue bzw. relevante Scripte:

```text
src/
├── data/
├── modeling/
│   ├── model_dataset.py
│   ├── dataset_tft.py
│   ├── trainer_tft.py
│   ├── trainer_arima.py         # neu
│   └── trainer_prophet.py       # neu
├── evaluation/                  # perspektivisch
│   ├── evaluate_tft.py          # geplant
│   ├── evaluate_arima.py        # geplant
│   └── evaluate_prophet.py      # geplant
├── utils/
└── visualization/
```

Zugehörige Konfigurationen:

```text
configs/
├── trainer_tft_baseline.yaml
├── trainer_arima_baseline.yaml      # neu
└── trainer_prophet_baseline.yaml    # neu
```

Logs und Ergebnisse:

```text
logs/
├── tft/run_*/...
├── arima/run_*/...                  # neu
└── prophet/run_*/...                # neu

results/
├── tft/checkpoints/run_*/...
├── arima/checkpoints/run_*/...      # neu
├── prophet/checkpoints/run_*/...    # neu
└── evaluation/
    ├── tft/run_*/...                # Auswertung (geplant)
    ├── arima/run_*/...              # Auswertung (geplant)
    └── prophet/run_*/...            # Auswertung (geplant)
```

---

## 3. Eingabedaten für ARIMA und Prophet

Die Trainer für ARIMA/Prophet verwenden dieselben vorbereiteten Dateien wie der TFT-Trainer:

- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`

Spalten basieren ausschließlich auf `src/config.py`:

- Zeitspalte: `TIME_COL`
- Identifikator-Spalten: `ID_COLS`
- Zielvariable: `TARGET_COL`

Wichtig:

- Es werden **keine** Pfade oder Spaltennamen im Code hart kodiert.
- Alle Konstanten stammen aus `src/config.py`.

---

## 4. Aggregation und Modellfokus

ARIMA und Prophet arbeiten typischerweise auf **univariaten** Zeitreihen. Für die erste Projektphase wird vereinbart:

- Der Trainer arbeitet auf einer **klar definierten Zielzeitreihe**, festgelegt über die YAML-Konfiguration, z. B.:
  - globale Aggregation: Summe über alle Kombinationen von `ID_COLS` pro Datum, oder
  - Filter auf eine konkrete Kombination (`country/store/product`) über einen Filterblock in der YAML.

Die Aggregationslogik selbst liegt im jeweiligen Trainer-Skript, die dafür benötigten Parameter (z. B. Filterwerte) werden aus der YAML gelesen.

---

## 5. Aufgaben der Trainer-Skripte (TFT, ARIMA, Prophet)

Alle Trainer folgen demselben Grundprinzip:

- Laden der vorbereiteten Daten (`train/val/test`)  
- Aufbau und Training des jeweiligen Modells  
- Speichern der trainierten Modellinstanz  
- Optional: einfache Lauf-Logs (Konsole, ggf. kurze Textdatei)

**Keine** der Trainer-Komponenten berechnet „offizielle“ Vergleichsmetriken oder schreibt `summary.json`.  
Diese Aufgaben sind bewusst in **Evaluationsskripte** ausgelagert.

### 5.1 TFT (`trainer_tft.py`)

- nutzt bereits:
  - Logging über Lightning → `logs/tft/run_*/metrics.csv`
  - Checkpoints → `results/tft/checkpoints/run_*/...`

### 5.2 ARIMA (`trainer_arima.py` – geplant)

- lädt `train/val/test.parquet`
- wendet den über die YAML konfigurierten Aggregationsmodus an
- trainiert ein ARIMA-Modell
- speichert das Modell, z. B.:
  - `results/arima/checkpoints/run_*/model.pkl`

### 5.3 Prophet (`trainer_prophet.py` – geplant)

- analog zu ARIMA:
  - Daten laden
  - Aggregation / Filter anwenden
  - Prophet-Modell trainieren
  - Modell speichern:
    - `results/prophet/checkpoints/run_*/model.pkl`

---

## 6. Aufgaben der Evaluationsskripte (modellübergreifend)

Die Evaluationsskripte sind für den **eigentlichen Modellvergleich** zuständig.  
Sie werden für TFT, ARIMA und Prophet nach dem gleichen Muster aufgebaut.

Geplante Struktur:

```text
src/evaluation/
├── evaluate_tft.py
├── evaluate_arima.py
└── evaluate_prophet.py
```

Jedes Evaluationsskript soll:

1. das passende Modell (Checkpoint bzw. Modell-Datei) laden,  
2. `val.parquet` und `test.parquet` einlesen,  
3. Vorhersagen für Validation und Test erzeugen,  
4. Metriken berechnen, z. B.:
   - MAE  
   - RMSE  
   - MAPE/SMAPE  
5. Ergebnisse in einer einheitlichen Struktur speichern, z. B.:

```text
results/evaluation/tft/run_*/summary.json
results/evaluation/arima/run_*/summary.json
results/evaluation/prophet/run_*/summary.json
```

Optional können zusätzlich Vorhersagedateien erzeugt werden:

```text
results/evaluation/<modell>/run_*/predictions_val.parquet
results/evaluation/<modell>/predictions_test.parquet
```

Die Evaluationsskripte sind damit der zentrale Ort für den späteren quantitativen Modellvergleich.

---

## 7. Schnittstellen und Konfigurationslogik

- **Pfade und Basis-Konfiguration**
  - stammen ausschließlich aus `src/config.py`.
- **Modell- und Laufparameter**
  - werden nur in den YAML-Dateien (`trainer_*.yaml`) definiert.
- Trainer-Skripte sollen nach dem gleichen Muster aufgebaut sein wie `trainer_tft.py`:
  1. Imports
  2. Konstante Importe aus `src.config`
  3. Funktions- und ggf. Klassen-Definitionen
  4. `main(config_path: str)`-Funktion
  5. `if __name__ == "__main__":`-Block mit Aufruf über `python -m src.modeling.<skript>`

---

## 8. Nächste Schritte für die Implementierung

1. YAML-Baseline-Dateien für ARIMA und Prophet anlegen.
2. `trainer_arima.py` implementieren.
3. `trainer_prophet.py` implementieren.
4. Evaluationsskripte für TFT, ARIMA, Prophet vorbereiten.
5. Konsistenz der Ordnerstruktur testen.

Diese Struktur stellt sicher, dass Training und Evaluation klar getrennt sind und ein späterer Modellvergleich technisch und konzeptionell sauber erfolgen kann.
