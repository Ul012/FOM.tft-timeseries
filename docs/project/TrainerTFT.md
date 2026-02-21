# TrainerTFT – Training des Temporal Fusion Transformers

**Datum:** 2026-02-21  
**Script:** `src/modeling/trainer_tft.py`  
**Ziel & Inhalt:** Beschreibt Ablauf, Konfigurationslogik und Trainingsprozess des TFT-Modells.

---

## Überblick

Der Trainer initialisiert das TFT-Modell auf Basis der Model- und Dataset-Konfiguration, 
führt das Training mit PyTorch Lightning durch und speichert Artefakte wie Checkpoints und Metriken.

---

## Trainingsablauf

1. Laden der Konfiguration  
2. Initialisierung des Datasets  
3. Aufbau des TFT-Modells  
4. Training mit Lightning Trainer  
5. Speichern von Checkpoints  
6. Persistieren von Metriken

---

## Runprotokoll (Lightning Logs)

Während eines Trainingslaufs erzeugt PyTorch Lightning strukturierte Konsolenausgaben. 
Diese Logs dokumentieren die einzelnen Phasen des Trainingsprozesses.

### 1. Initialisierung

- Hinweise zu Lag-Features (NaN am Anfang von Sequenzen)  
- Geräteauswahl (CPU/GPU)  
- Logger-Initialisierung  
- Batch-Limits (Verarbeitung aller Batches pro Epoche)

### 2. Modellzusammenfassung

Lightning listet die Hauptkomponenten des TFT-Modells:

- Einbettungen für kategoriale Merkmale  
- Skalierung numerischer Merkmale  
- Variable-Selection-Netzwerke  
- LSTM-Encoder und Decoder  
- Attention-Mechanismen  
- Normierungs- und Gate-Strukturen  
- Lineare Ausgabeschichten  

Diese Zusammenfassung dient der strukturellen Verifikation des Modells.

### 3. Sanity-Check

Vor Beginn des eigentlichen Trainings führt Lightning einen kurzen Validierungsdurchlauf aus, 
um DataLoader und Modellkompatibilität zu prüfen.

### 4. Trainingsphase

Pro Epoche werden u. a. folgende Kennzahlen ausgegeben:

- train_loss_step  
- train_loss_epoch  
- val_loss  

Die Entwicklung des Validierungswerts gibt Aufschluss über Lernfortschritt 
und mögliche Überanpassung.

### 5. Checkpoints

Es werden typischerweise gespeichert:

- Letzter Checkpoint (aktueller Stand)  
- Bester Checkpoint (niedrigster Validierungs-Loss)  

---

## Ergebnis und Nutzen

Das Runprotokoll ermöglicht:

- Nachvollziehbarkeit des Trainingsverlaufs  
- Technische Verifikation der Modellstruktur  
- Reproduzierbare Dokumentation einzelner Runs
