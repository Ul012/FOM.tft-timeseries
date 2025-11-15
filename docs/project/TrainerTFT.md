# Trainer_TFT – Training, Artefakte & Ergebnisdateien

**Datum:** 2025-11-15  
**Script:** `src/modeling/trainer_tft.py`  
**Ziel & Inhalt:** Dieses Dokument beschreibt Aufbau, Ablauf und Ausgaben des TFT-Trainingsprozesses. Zusätzlich werden alle Kennzahlen der Datei `summary.json` erklärt und in den Kontext des Projekts eingeordnet.

---

# 1. Aufgaben des Trainers

Der Trainer übernimmt den vollständigen Trainingsprozess für das TFT-Modell:

- Laden des vorbereiteten Datensatzes  
- Erstellen des `TimeSeriesDataSet` (Encoder/Decoder-Fenster, Feature-Rollen, Lags)  
- Initialisierung des Temporal Fusion Transformer  
- Start und Steuerung der Trainingsschleife über PyTorch Lightning  
- Logging aller relevanten Metriken pro Epoche  
- Speichern des besten Modells (Checkpointing)  
- Erstellen der konsolidierten Ergebnisdatei `summary.json`  

Die Architektur ist bewusst modular: Datensatz, Features, Modell und Training werden getrennt definiert und gesteuert.

---

# 2. Eingaben des Trainers

### 2.1 Konfigurationsdatei (`trainer_tft_*.yaml`)
Enthält alle hyperparametrischen Einstellungen des Laufs:

- Modellparameter  
- Batchgröße  
- Lernrate  
- Fenstergrößen  
- Early-Stopping, Gradient Clipping  
- DataLoader-Parameter  
- Logging-Optionen

### 2.2 Verarbeitete Datensätze (`data/processed/…`)
Der Trainer lädt:

- Training  
- Validation  
- Test  

als Parquet-Dateien, erzeugt vom ModelDataset-Script.

### 2.3 Projektweite Konstanten (`src/config.py`)
Zentrale Pfade, Namen und Split-Grenzen.

---

# 3. Ausgaben des Trainers

Alle Artefakte eines Runs befinden sich unter:

```
results/tft/<run_id>/
```

Typische Verzeichnisstruktur:

```
│
├── checkpoints/
│   └── best.ckpt
│
├── metrics.csv
├── summary.json
└── full_config.yaml
```

---

# 4. Beschreibung der Ausgabedateien

## 4.1 metrics.csv
CSV-Logger von PyTorch Lightning.  
Pro Epoche wird gespeichert:

- `train_loss`  
- `val_loss`  
- optionale Lernrate (`lr-Adam`)  
- Epochennummer  

Diese Datei dient als Grundlage für:

- Visualisierung (Loss-Plots)  
- Evaluator-Skripte  
- Vergleich zwischen verschiedenen Runs  

---

## 4.2 summary.json  
Konsolidierte Zusammenfassung eines gesamten Runs.  
Sie enthält nur wenige, aber **entscheidende** Kennzahlen, mit denen Runs schnell miteinander verglichen werden können.

Die Werte werden **automatisch** aus der `metrics.csv` abgeleitet.

Die vollständige Erklärung der Metriken folgt unter Kapitel 5.

---

## 4.3 checkpoints/best.ckpt
Ein binärer Snapshot des Modells, der jene Epoche enthält, in der der Validierungs-Loss minimal war.

Wird verwendet für:

- Vorhersage  
- Evaluierung  
- Reloading für neue Experimente  
- Fine-Tuning  

---

## 4.4 full_config.yaml
Persistierte vollständige Konfiguration des Laufs.  
Wichtig für:

- Reproduzierbarkeit  
- vollständige Dokumentation  
- Wiederherstellung von Laufbedingungen  

---

# 5. Erklärung der summary.json-Metriken

Die Datei `summary.json` hat die Aufgabe, in kompakter Form darzustellen, wie gut ein Lauf insgesamt war.  
Sie aggregiert alle wichtigen Informationen aus `metrics.csv` in einem leicht vergleichbaren Format.

Die Werte sind generalisierbar und unabhängig von einem konkreten Run.

---

## 5.1 Loss-Werte – was sie bedeuten (je niedriger, desto besser)

### **final_train_loss**
- Durchschnittlicher Trainings-Loss der letzten vollständig durchlaufenen Epoche.  
- Gibt an, wie gut das Modell **auf den bekannten Trainingsdaten** gelernt hat.  
- Interpretation:  
  - sehr weit unter dem Validierungs-Loss → Gefahr von Overfitting  
  - ähnlich wie Validierungs-Loss → Training stabil, Lernverhalten stimmig

---

### **final_val_loss**
- Validierungs-Loss der letzten Epoche.  
- Zentrale Kennzahl dafür, wie gut das Modell auf **unbekannte Daten** generalisiert.  
- Wird später für Modellvergleiche genutzt.

---

## 5.2 Bestes Validierungsergebnis

### **best_val_loss**
- Niedrigster Validierungs-Loss, der während des gesamten Trainings erreicht wurde.  
- Dieser Wert bestimmt, welches Modell als „best.ckpt“ abgespeichert wird.  
- Die wichtigste Vergleichsmetrik zwischen verschiedenen Läufen.

---

### **best_val_epoch**
- Epoche, in der der beste Validierungs-Loss gemessen wurde.  
- Interpretation:  
  - sehr frühes Minimum → Modell lernt schnell, mögliche Gefahr von Overfitting  
  - spätes Minimum → längeres Training ist hilfreich, gute Lernkurve  

---

## 5.3 Lernraten-Werte

Diese Werte erscheinen nur, wenn ein Lernraten-Logger in den Trainer eingebunden ist.

### **initial_lr**
- Erste aufgezeichnete Lernrate des genutzten Optimizers.  
- Hilfreich zur Dokumentation oder zum Vergleich von Hyperparameter-Runs.

### **final_lr**
- Letzte aufgezeichnete Lernrate im Training.  
- Zeigt, ob ein Scheduler aktiv war (z. B. Reduktion bei schwacher Validierungsperformance).  
- Kann `NaN` sein, wenn keine Lernrate geloggt wurde.  
  → kein Fehler des Trainings, nur fehlendes Logging.

---

# 6. Schnellinterpretation der Loss-Werte

| Verhalten | Bedeutung |
|----------|-----------|
| Train-Loss ≈ Val-Loss, beide niedrig | gutes Fit, stabile Generalisierung |
| Train-Loss ≪ Val-Loss | Overfitting |
| beide Loss-Werte hoch | Underfitting oder unzureichende Features |
| best_val_epoch sehr früh | Modell lernt schnell, ggf. Early Stopping |
| best_val_epoch spät | länger trainieren lohnt sich |

---

# 7. Ablauf des Trainings (Kurzfassung)

1. Setzen des Seeds  
2. Laden des Datensatzes  
3. Erstellen des `TimeSeriesDataSet`  
4. Initialisieren des TFT-Modells  
5. Training:
   - Train/Val-Loss pro Epoche  
   - optional: Logging der Lernrate  
6. Speichern des besten Modells  
7. Export:
   - `metrics.csv`  
   - `summary.json`  
   - `full_config.yaml`  
8. Konsolenausgabe mit Best-Checkpoint und Abschlussinformationen  

---

*Diese Dokumentation beschreibt den Trainer, seine Konfiguration und die Bedeutung der erzeugten Artefakte.  
Für eine zeilenweise Erklärung eines tatsächlichen Trainingslaufs siehe „Trainer_TFT_Runprotokoll.md“.*

