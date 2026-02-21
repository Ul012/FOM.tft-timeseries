# EvaluatorTFT – Validierungs- und Testauswertung

**Datum:** 2026-02-21  
**Script:** `src/evaluation/evaluate_tft.py`  
**Ziel & Inhalt:** Beschreibung der Auswertung abgeschlossener TFT-Trainingsläufe anhand vorliegender Checkpoints sowie der Validierungs- und Testdaten.

---

## Überblick

`evaluate_tft.py` lädt ein trainiertes TFT-Modell und bewertet dessen Leistung auf Validation- und Testdaten.

Wesentliche Eigenschaft:  
Die Evaluation verwendet die im **Modell-Checkpoint gespeicherten Dataset-Parameter**.  
Es werden **nicht** die aktuellen Feature-Listen aus `dataset_spec.json` genutzt.

Dadurch ist sichergestellt, dass die Bewertung exakt mit den Trainingsbedingungen des jeweiligen Runs übereinstimmt.

---

## Ziel

Ziel ist eine konsistente und reproduzierbare Bewertung eines abgeschlossenen Trainingslaufs unabhängig von späteren Änderungen an der Dataset-Spezifikation.

---

## Eingaben

- Run-ID eines abgeschlossenen Trainings **oder** direkter Pfad zu einem Checkpoint  
- `val.parquet`  
- `test.parquet`  

Die Datensplits werden aus `data/processed/<dataset>/` geladen.

---

## Betriebsmodi

### 1. Run-ID-basiert
Checkpoint wird automatisch im Run-Verzeichnis gesucht.

### 2. Checkpoint-basiert
Direkter Pfad zu einer `.ckpt`-Datei (z. B. für Optuna-Trials).

---

## Verarbeitungsschritte

### 1. Checkpoint-Ermittlung
- Bevorzugt Datei mit „best“ im Namen  
- Andernfalls erste `.ckpt`-Datei im Checkpoint-Ordner  

---

### 2. Laden der Daten
Validation- und Testdaten werden eingelesen und nach Gruppen- und Zeitspalten sortiert.

---

### 3. Dataset-Konstruktion
- Feature-Listen stammen aus `model.hparams.dataset_parameters`  
- `GroupNormalizer` wird verwendet  
- `allow_missing_timesteps=True`  
- Dataset-Parameter entsprechen denen des Trainings  

---

### 4. Vorhersage
- Modell wird im Eval-Modus ausgeführt  
- Quantile-Predictions werden auf den Median reduziert  
- Ziel- und Prognosewerte werden für die Metrikberechnung geflattet  

Optional: Speicherung von `predictions_<split>.npy` und `actuals_<split>.npy` via `--save-predictions`.

---

### 5. Metrikberechnung
Unterstützte Kennzahlen (konfigurationsgesteuert):  
- MAE  
- RMSE  
- MAPE  
- SMAPE  
- R²  

---

### 6. Artefakte

Pro Run werden erzeugt:

```
results/tft/eval/<run_id>/
├── eval_summary.json
└── eval_summary.csv
```

`eval_summary.json` enthält Run-ID, Checkpoint-Pfad, Metriken (Val/Test) und Metainformationen (time_col, id_cols, target_col).  
`eval_summary.csv` enthält eine tabellarische Zeile pro Run.

---

## Beispielaufruf

```bash
# Run-ID
python -m src.evaluation.evaluate_tft --run-id <run_id>

# Direkter Checkpoint
python -m src.evaluation.evaluate_tft --checkpoint <path_to_ckpt>

# Mit Speicherung der Predictions
python -m src.evaluation.evaluate_tft --run-id <run_id> --save-predictions
```

---

## Ergebnis und Nutzen

- Reproduzierbare Bewertung einzelner Runs  
- Saubere Trennung von Training und Evaluation  
- Evaluationsartefakte unabhängig von Trainingslogs  
- Grundlage für strukturierte Modellvergleiche
