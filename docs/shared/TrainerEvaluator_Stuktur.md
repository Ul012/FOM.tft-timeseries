# Struktur und Verantwortlichkeiten: Trainer – Evaluator – Visualizer

**Datum:** 2025-11-16  
**Script:** –  
**Ziel & Inhalt:** Definiert die rollenbasierte Trennung zwischen Trainer, Evaluator und Visualizer. Beschreibt ihre Aufgaben, Ordnerstrukturen, Artefakte und den sequenziellen Workflow. Dient als Leitlinie für saubere Zuständigkeiten und reproduzierbare Ergebnisse.

---

## 🎯 1. Grundprinzipien

- Jede Komponente (Trainer, Evaluator, Visualizer) erfüllt **eine klar abgegrenzte Aufgabe**.  
- Jede Komponente schreibt ihre Ergebnisse in **eindeutig definierte Ordner**.  
- Kein Skript überschreibt oder dupliziert Zuständigkeiten anderer Module.  

Diese Struktur folgt dem Prinzip der **Separation of Concerns** und dem **Single Responsibility Principle (SRP)**.

---

## ⚙️ 2. Komponentenübersicht

| Komponente | Hauptaufgabe | Typische Dateien | Verantwortlich für |
|-----------|--------------|------------------|--------------------|
| **Trainer** | Modelltraining, Checkpoints, Trainings-Summaries | `src/modeling/trainer_tft.py` | Training, Laufzeit-Artefakte |
| **Evaluator** | Berechnung von Fehlermaßen für fertige Modelle | `src/evaluation/evaluate_tft.py`, `aggregate_tft_eval.py` | Val/Test-Metriken, Run-Übersichten |
| **Visualizer** | Grafische Darstellung von Training und Evaluation | `src/visualization/plot_learning_rate.py`, `plot_tft_eval_comparison.py`, `plot_tft_forecast_series.py` | Plots für Dokumentation |

---

## 🧩 3. Trainer

**Aufgabe:**  
Trainiert den Temporal Fusion Transformer auf Basis einer YAML-Config.  

**Verantwortlichkeiten:**

- Vorbereitung von Datensatz und Modell  
- Ausführen des Trainings (Epochen, Batches)  
- Speichern von Checkpoints (z. B. „best model“)  
- Schreiben von Trainings-Summaries pro Run (z. B. Loss/MAE in der letzten Epoche)

**Ordnerstruktur (relevant):**

```text
results/
└─ tft/
   └─ runs/
      └─ run_YYYYMMDD_HHMMSS_suffix/
         ├─ checkpoints/
         │  └─ tft-*.ckpt
         └─ training_summary.json   # oder vergleichbare Zusammenfassung je Run
```

**Ausgabe:**
- Modellgewichte (`*.ckpt`)
- Kompakte Trainingsinformationen pro Run (JSON/CSV)

Der Trainer führt **keine** finale Val/Test-Evaluation durch – dies übernimmt der Evaluator.

---

## 4. Evaluator

**Aufgabe:**  
Der Evaluator bewertet ausschließlich bereits trainierte Modelle. Er führt kein Training durch, verändert keine Modellparameter und arbeitet rein lesend auf fertigen artefakten.  
Er berechnet standardisierte Fehlermaße (MAE, RMSE, MAPE, SMAPE) für Validierung und Test.

### 4.1 `evaluate_tft.py`

**Eingaben:**
- TFT-Checkpoint:  
  `results/tft/runs/<run_id>/checkpoints/*.ckpt`
- Datensplits:  
  `data/processed/val.parquet`  
  `data/processed/test.parquet`

**Bedeutende interne Logik:**
- Berücksichtigung des sequentiellen TFT-Fensters:  
  Encoder: `max_encoder_length`  
  Decoder: `max_prediction_length`
- Aus den Daten werden pro Zeitreihe nur die relevanten Forecast-Fenster ausgewählt.
- Modellvorhersage via `model.predict(df)`.

**Ausgabe:**
```
results/tft/eval/<run_id>/eval_summary.json
```

Der Inhalt umfasst:
- Metriken für Val und Test
- Pfad zum verwendeten Checkpoint
- Meta-Informationen zu Zeit- und ID-Spalten

---

### 4.2 `aggregate_tft_eval.py`

**Aufgabe:**  
Es werden alle vorhandenen `eval_summary.json` Dateien gesammelt, um eine zentrale Übersicht zu erstellen.

**Ablauf:**  
- rekursive Suche unter `results/tft/eval/`
- Extraktion folgender Werte:
  - `val_mae`, `val_rmse`, `val_mape`, `val_smape`
  - `test_mae`, `test_rmse`, `test_mape`, `test_smape`
- Zusammenführung in einer sortierten Tabelle

**Ausgabe:**
```
results/tft/eval/eval_overview.csv
results/tft/eval/eval_overview.json
```

Diese Übersicht bildet die Grundlage für alle Run-Vergleiche.

---

## 5. Visualizer

Der Visualizer erzeugt grafische Darstellungen.  
Er nutzt ausschließlich Daten aus dem Training (z. B. Trainingssummaries) oder aus der Evaluation (z. B. `eval_overview.csv`).  
Er führt weder Training noch Bewertung durch.

### Beispiele:

#### `plot_learning_rate.py`
- Visualisiert Lernkurven, Validierungsfehler (falls geloggt) und Lernrate pro Epoche.

#### `plot_tft_eval_comparison.py`
- Nutzt `eval_overview.csv`
- Erzeugt Balkendiagramme eines gewählten Metrik-/Split-Paares (z. B. Test-SMAPE)

Ablage:
```
results/tft/plots/eval/compare_<split>_<metric>.png
```

#### `plot_tft_forecast_series.py`
- Zeigt Ist-Historie, Ist im Forecast-Bereich und Modellprognose für eine einzige Zeitreihe.
- Ideal für qualitative Beispiele, z. B. Peak-Verhalten (Weihnachten).

Ablage:
```
results/tft/plots/eval/<run_id>_<split>_forecast_series.png
```

---

## 🧠 6. Gesamtfluss (Trainer → Evaluator → Visualizer)

```
(1) Trainer
     trainiert TFT
     speichert Checkpoints unter results/tft/runs/<run_id>/
     schreibt Trainingssummaries

          ▼

(2) Evaluator
     lädt Checkpoints
     liest val/test.parquet
     berechnet MAE, RMSE, MAPE, SMAPE
     schreibt eval_summary.json je Run

          ▼

(3) Visualizer
     nutzt eval_overview.csv & training summaries
     erstellt Plots (Lernkurven, Run-Vergleiche, Forecast-Zeitreihen)
     speichert unter results/tft/plots/
```

Diese Struktur stellt sicher, dass:
- Training und Bewertung getrennt bleiben,
- Visualisierungen reproduzierbar sind,
- Modelle systematisch geprüft und verglichen werden können.

---

## 🧭 7. Zuordnung der Dateitypen

| Dateityp | Inhalt | Herkunft | Ablage |
|-----------|---------|-----------|---------|
| `.csv` | Metriken, Tabellen | Trainer, Evaluator | `logs/`, `results/` |
| `.ckpt` | Modellgewichte | Trainer | `logs/.../checkpoints/` |
| `.yaml` | Hyperparameter | Trainer | `logs/.../` |
| `.png` | Visualisierungen | Visualizer | `results/plots/` |
| `.md` | Dokumentation | Manuell / MkDocs | `docs/`, `notes/` |

---

## ✅ 8. Vorteile der Trennung

| Vorteil | Beschreibung |
|----------|---------------|
| **Klarheit** | Jede Komponente hat einen klaren Verantwortungsbereich |
| **Wartbarkeit** | Änderungen an Training oder Evaluation erfordern keine Eingriffe in andere Module |
| **Reproduzierbarkeit** | Jeder Run ist durch `logs/run_*` vollständig nachvollziehbar |
| **Automatisierung** | Pipeline-Skripte können gezielt einzelne Phasen (train, eval, visualize) anstoßen |
| **Erweiterbarkeit** | Weitere Modelle oder Evaluatoren können einfach ergänzt werden |

---

## 📚 9. Nächste Schritte

1. Sicherstellen, dass `LearningRateMonitor` im Trainer aktiv ist.  
2. `evaluate_tft.py` erweitern, um finale Metriken automatisch aus allen Runs zu extrahieren.  
3. Erste Visualisierung mit `runs_summary.csv` aufbauen.  
4. Alle Ergebnisse im `results/`-Ordner dokumentieren.

---

**Kurzfazit:**  
> Der **Trainer** produziert Laufzeitdaten (logs),  
> der **Evaluator** verdichtet diese zu aussagekräftigen Ergebnissen (results),  
> und der **Visualizer** macht sie sichtbar und verständlich.
