# Evaluator_TFT – Validierungs- und Testauswertung

**Datum:** 2025-11-16  
**Script:** `src/evaluation/evaluate_tft.py`  
**Ziel & Inhalt:** Dieses Dokument beschreibt die Aufgaben, Eingaben und Ausgaben des Evaluators für TFT-Modelle. Es wird erläutert, wie fertige Modelle auf Validierungs- und Testdaten ausgewertet werden und welche Artefakte dabei entstehen.

---

## 1. Rolle im Projektkontext

Der Evaluator ist klar vom Trainer und den Visualisierungs-Skripten getrennt.

- **Trainer (`trainer_tft.py`)**: führt das Training durch, schreibt Logs, Checkpoints und Trainings-Summaries.  
- **Evaluator (`evaluate_tft.py`)**: lädt ein fertig trainiertes Modell und berechnet Fehlermaße auf Validation und Test.  
- **Visualizer** (z. B. später Metrik-Plots): erzeugt Diagramme auf Basis von Logs und Evaluationsdaten.

Der Evaluator verändert keine Modellparameter und startet kein Training. Es wird ausschließlich mit bereits gespeicherten Artefakten gearbeitet.

---

## 2. Eingaben

### 2.1 Run-ID und Checkpoint

Die Evaluierung erfolgt run-basiert. Ein Run entspricht einem abgeschlossenen Training, das nach folgendem Muster abgelegt ist:

```
results/
└─ tft/
   └─ runs/
      └─ run_YYYYMMDD_HHMMSS_suffix/
         └─ checkpoints/
            └─ <checkpoint_datei>.ckpt
```

- Die **Run-ID** entspricht exakt dem Ordnernamen, z. B.:  
  `run_20251115_132824_baseline`  
- Der Evaluator sucht im Unterordner `checkpoints/` nach einer `.ckpt`-Datei.  
- Preferiert werden Dateien mit „best“ im Namen, ansonsten die erste gefundene Datei.

### 2.2 Datensplits

Für die Evaluierung werden die verarbeiteten Datensätze aus `data/processed/` verwendet:

- `val.parquet`  
- `test.parquet`

Diese werden durch `model_dataset.py` erzeugt.

### 2.3 Konfigurationsblöcke im Evaluator

Drei Konfigurationsblöcke steuern Pfade und Parameter:

#### a) `data_cfg`

```python
data_cfg = {
    "val_path": str(PROCESSED_DIR / "val.parquet"),
    "test_path": str(PROCESSED_DIR / "test.parquet"),
}
```

Legt fest, wo die Daten liegen.

#### b) `model_cfg`

```python
model_cfg = {
    "checkpoint_root": str(BASE_DIR / "results" / "tft" / "runs"),
    "run_id": run_id,
    "checkpoint_pattern": "*.ckpt",
}
```

Bestimmt, welches Modell evaluiert wird.

#### c) `eval_cfg`

```python
eval_cfg = {
    "eval_root": str(BASE_DIR / "results" / "tft" / "eval"),
}
```

Steuert, wohin die Evaluationsdateien geschrieben werden.

---

## 3. Ausgaben und Artefakte

### 3.1 Ordnerstruktur

Nach der Evaluierung entsteht:

```
results/
└─ tft/
   └─ eval/
      └─ run_YYYYMMDD_HHMMSS_suffix/
         └─ eval_summary.json
```

### 3.2 Inhalt von `eval_summary.json`

Beispiel:

```json
{
  "run_id": "run_20251115_132824_baseline",
  "checkpoint_path": "results/tft/runs/run_20251115_132824_baseline/checkpoints/tft-01-12.ckpt",
  "metrics": {
    "val": { "mae": 1.23, "rmse": 1.45, "mape": 12.34, "smape": 11.11 },
    "test": { "mae": 1.34, "rmse": 1.56, "mape": 13.20, "smape": 12.05 }
  },
  "meta": {
    "time_col": "date",
    "id_cols": ["country", "store", "product"],
    "target_col": "num_sold"
  }
}
```

Die Datei dient als Grundlage für Modell- und Hyperparametervergleiche.

---

## 4. Sequenzielles Arbeiten von PyTorch Forecasting

Der Temporal Fusion Transformer arbeitet **nicht zeilenweise**, sondern **sequenziell** über Encoder- und Decoder-Fenster.

In `src/config.py` befindet sich:

```python
TFT_DATASET = {
    "max_encoder_length": 28,
    "max_prediction_length": 7
}
```

Daraus folgen feste Fenstergrößen:

- Encoder-Fenster: 28 Zeitschritte Vergangenheit  
- Decoder-Fenster: 7 Zeitschritte Zukunft (Vorhersagehorizont)

### Warum das wichtig ist

Beim Aufruf von:

```python
model.predict(df_val)
```

macht PyTorch Forecasting Folgendes:

1. Die Daten werden intern in **Samples** zerlegt:  
   je Serie → *(letzte 28 Encoder-Schritte) → (Vorhersage der nächsten 7 Schritte)*  
2. Das Modell erstellt **nicht** für jede Zeile des DataFrames eine Vorhersage.  
3. Ergebnis ist ein Array mit:

```
Anzahl_Zeitreihen × max_prediction_length
```

Beispiel:

- 48 Zeitreihen  
- `max_prediction_length = 7`  
→ `48 × 7 = 336` Vorhersagen

Wenn der Validation-Frame 7000 Zeilen enthält, dann:

- `y_pred` = 336 Werte  
- `y_true` = muss ebenfalls 336 Werte haben, **genau dieselben Zeitpunkte**

### Konsequenz für die Evaluierung

`y_true` darf **nicht** alle Zeilen des Validation-Sets enthalten.

Sondern:

- pro Zeitreihe  
  - sortieren nach `TIME_COL`  
  - letzte **7** Zeitschritte nehmen (= Vorhersagehorizont)

Der Evaluator berücksichtigt dies jetzt.

---

## 5. Ablauf der Evaluierung

1. Checkpoint finden  
2. TFT-Modell laden  
3. Validation- und Testdaten laden  
4. Pro Serie die letzten `max_prediction_length` Zeitpunkte auswählen  
5. `model.predict()` aufrufen  
6. `y_pred` und `y_true` vergleichen  
7. Fehlermaße berechnen (MAE, RMSE, MAPE, SMAPE)  
8. `eval_summary.json` schreiben  
9. Ergebnis als Dictionary zurückgeben

---

## 6. Nutzung

```bash
python -m src.evaluation.evaluate_tft --run-id run_20251115_132824_baseline
```

Voraussetzungen:

- trainierter Run unter `results/tft/runs/<run_id>`  
- Checkpoint im Unterordner `checkpoints/`  
- `val.parquet` und `test.parquet` existieren in `data/processed/`

---

## 7. Erweiterbarkeit & MLflow-Vorbereitung

Die Struktur erlaubt:

- YAML-basierte Konfiguration  
- Austausch des Loggers (`EvalLogger`) gegen MLflow  
- Evaluierung weiterer Modelle (ARIMA, Prophet) mit gleicher Outputstruktur  
- Einbindung in zukünftige Hyperparameter-Suchen (Optuna)
- Ergänzende Auswertungsskripte:
  - `aggregate_tft_eval.py` zur Aggregation aller `eval_summary.json` 
  - Visualisierungsskripte wie `plot_tft_eval_comparison.py` und `plot_tft_forecast_series.py` (siehe `EvaluatorTFTPlot.md`)

