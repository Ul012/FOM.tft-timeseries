# Struktur und Verantwortlichkeiten: Trainer – Evaluator – Visualizer

**Projektkontext:** *TFT-TimeSeries*  
Ziel ist eine saubere Trennung von Verantwortlichkeiten zwischen den Komponenten des Projekts, um die Nachvollziehbarkeit, Erweiterbarkeit und Wartbarkeit zu sichern.

---

## 🎯 Grundprinzipien

- Jede Komponente (Trainer, Evaluator, Visualizer) erfüllt **eine klar abgegrenzte Aufgabe**.  
- Jede Komponente schreibt ihre Ergebnisse in **eindeutig definierte Ordner**.  
- Kein Skript überschreibt oder dupliziert Zuständigkeiten anderer Module.  

Diese Struktur folgt dem Prinzip der **Separation of Concerns** und dem **Single Responsibility Principle (SRP)**.

---

## ⚙️ Komponentenübersicht

| Komponente | Hauptaufgabe | Typische Datei | Verantwortlich für |
|-------------|---------------|----------------|--------------------|
| **Trainer** | Modell trainieren und Laufzeitmetriken erfassen | `src/modeling/trainer_tft.py` | Training, Logging, Checkpoints |
| **Evaluator** | Ergebnisse bewerten und zusammenfassen | `src/evaluation/evaluate_tft.py` | Berechnung und Aggregation der Metriken |
| **Visualizer** | Ergebnisse grafisch darstellen | `src/visualization/compare_runs.py` / `plot_learning_rate.py` | Plots, Reports, Trends |

---

## 🧩 1. Trainer

**Aufgabe:**  
Führt das Modelltraining aus, protokolliert Metriken und speichert alle laufzeitbezogenen Artefakte.

**Verantwortlichkeiten:**
- Initialisierung des Modells (`TemporalFusionTransformer.from_dataset(...)`)
- Definition von Verlust- und Bewertungsmetriken (`QuantileLoss`, `MAE`, `RMSE`, `MAPE`, `SMAPE`)
- Logging (Loss, Metriken, Lernrate)
- Speichern der Checkpoints
- Ausgabe eines `metrics.csv`-Logs

**Ordnerstruktur:**
```
logs/
└─ tft/
   └─ run_YYYYMMDD_HHMMSS/
      ├─ metrics.csv              # Laufzeitmetriken (train/val/lr)
      ├─ checkpoints/             # Beste Gewichte (ModelCheckpoint)
      ├─ hparams.yaml             # Hyperparameter pro Run
      └─ run_summary.csv          # Letzte Epoche zusammengefasst
```

**Ausgabe:**  
- `metrics.csv` → vollständige Metrikverläufe (Loss, MAE, RMSE, LR)  
- `run_summary.csv` → letzte Epoche als Kurzreport  
- `checkpoints/` → gespeicherte Modellgewichte  

---

## 🧮 2. Evaluator

**Aufgabe:**  
Analysiert abgeschlossene Trainingsläufe und erzeugt standardisierte Auswertungen.  

**Verantwortlichkeiten:**
- Lesen aller `metrics.csv` aus `logs/tft/run_*`
- Ermitteln der finalen Werte (z. B. letzte Zeile pro Run)
- Erstellen von Vergleichstabellen (`runs_summary.csv`)
- Optional: Berechnung zusätzlicher Kennzahlen aus gespeicherten Checkpoints

**Ordnerstruktur:**
```
results/
└─ evaluation/
   ├─ runs_summary.csv        # Kompakte Vergleichstabelle über alle Runs
   ├─ eval_metrics.csv        # Ergebnisse aus geladenen Checkpoints (optional)
   └─ reports/                # spätere Text-/PDF-Berichte
```

**Ausgabe:**  
- `results/evaluation/runs_summary.csv` → konsolidierte Übersicht über alle Runs  
- (optional) `results/evaluation/eval_metrics.csv` → nachträglich berechnete Kennzahlen

---

## 📊 3. Visualizer

**Aufgabe:**  
Erstellt aus Logs und Evaluationsergebnissen anschauliche Darstellungen (Lernkurven, Run-Vergleiche).

**Verantwortlichkeiten:**
- Plotten von Loss- und LR-Verläufen aus `metrics.csv`
- Plotten von Balkendiagrammen aus `runs_summary.csv`
- Speichern der Visualisierungen im Unterordner `results/plots/`

**Ordnerstruktur:**
```
results/
└─ plots/
   ├─ learning_curve_with_params_bottom.png
   ├─ runs_comparison.png
   └─ weitere Visualisierungen
```

**Ausgabe:**  
- Diagramme, die direkt aus CSV-Dateien erzeugt werden  
- Bereitstellung für Dokumentation (z. B. MkDocs oder README)

---

## 🧠 4. Gesamtfluss (Trainer → Evaluator → Visualizer)

```
(1) Trainer
     │
     ├── trainiert Modell
     ├── schreibt logs/tft/run_*/metrics.csv
     └── speichert Checkpoints
            ↓
(2) Evaluator
     │
     ├── liest alle metrics.csv
     ├── extrahiert finale Metriken
     └── schreibt results/evaluation/runs_summary.csv
            ↓
(3) Visualizer
     │
     ├── liest runs_summary.csv & metrics.csv
     ├── erstellt Plots
     └── speichert in results/plots/
```

---

## 🧭 5. Zuordnung der Dateitypen

| Dateityp | Inhalt | Herkunft | Ablage |
|-----------|---------|-----------|---------|
| `.csv` | Metriken, Tabellen | Trainer, Evaluator | `logs/`, `results/` |
| `.ckpt` | Modellgewichte | Trainer | `logs/.../checkpoints/` |
| `.yaml` | Hyperparameter | Trainer | `logs/.../` |
| `.png` | Visualisierungen | Visualizer | `results/plots/` |
| `.md` | Dokumentation | Manuell / MkDocs | `docs/`, `notes/` |

---

## ✅ 6. Vorteile der Trennung

| Vorteil | Beschreibung |
|----------|---------------|
| **Klarheit** | Jede Komponente hat einen klaren Verantwortungsbereich |
| **Wartbarkeit** | Änderungen an Training oder Evaluation erfordern keine Eingriffe in andere Module |
| **Reproduzierbarkeit** | Jeder Run ist durch `logs/run_*` vollständig nachvollziehbar |
| **Automatisierung** | Pipeline-Skripte können gezielt einzelne Phasen (train, eval, visualize) anstoßen |
| **Erweiterbarkeit** | Weitere Modelle oder Evaluatoren können einfach ergänzt werden |

---

## 📚 7. Nächste Schritte

1. Sicherstellen, dass `LearningRateMonitor` im Trainer aktiv ist.  
2. `evaluate_tft.py` erweitern, um finale Metriken automatisch aus allen Runs zu extrahieren.  
3. Erste Visualisierung mit `runs_summary.csv` aufbauen.  
4. Alle Ergebnisse im `results/`-Ordner dokumentieren.

---

**Kurzfazit:**  
> Der **Trainer** produziert Laufzeitdaten (logs),  
> der **Evaluator** verdichtet diese zu aussagekräftigen Ergebnissen (results),  
> und der **Visualizer** macht sie sichtbar und verständlich.
