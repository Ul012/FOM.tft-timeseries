# EvaluatorTFT – Validierungs- und Testauswertung

**Datum:** 2025-11-17  
**Script:** `src/evaluation/evaluate_tft.py`  
**Ziel & Inhalt:** Beschreibung der Auswertung abgeschlossener TFT-Trainingsläufe anhand vorliegender Checkpoints und der Validierungs- und Testdaten.

---

## Überblick
Das Evaluationsskript lädt ein trainiertes Modell, wendet es auf die vorbereiteten Datensplits an und berechnet standardisierte Fehlerkennzahlen. Die Ergebnisse werden in einer strukturierten JSON-Datei abgelegt.

---

## Ziel
Ziel ist eine konsistente und reproduzierbare Bewertung der Modellleistung eines abgeschlossenen Trainingslaufs.

---

## Eingaben & Ausgaben

### Eingaben
- Run-ID eines abgeschlossenen Trainings  
- Checkpoint aus dem Run-Unterordner  
- `val.parquet`  
- `test.parquet`

### Ausgabe
- `eval_summary.json` im Evaluationsordner des jeweiligen Runs

---

## Vorgehen

### 1. Auswahl des Checkpoints  
Es wird die Datei mit „best“ im Namen bevorzugt, ansonsten die erste `.ckpt`-Datei im Ordner.

### 2. Laden der Daten  
Die Validierungs- und Testdaten werden eingelesen.

### 3. Ableitung der Zielzeitpunkte  
Für jede Zeitreihe werden die letzten `max_prediction_length` Schritte bestimmt.

### 4. Modellvorhersage  
Der TFT erstellt Vorhersagen über das definierte Decoderfenster.

### 5. Berechnung der numerischen Kennzahlen  
Aus Vergleich von Vorhersagen und Zielwerten entstehen die Metriken:
- MAE  
- RMSE  
- MAPE  
- SMAPE

### 6. Schreiben des Evaluationsartefakts  
Alle Ergebnisse werden in `eval_summary.json` gespeichert.

---

## Beispielaufruf
```bash
python -m src.evaluation.evaluate_tft --run-id <run_id>
```

---

## Ergebnis und Nutzen
- numerische Bewertung eines Runs  
- strukturierte Speicherung aller Kennzahlen  
- Grundlage für aggregierte Auswertungen und Modellvergleiche  
