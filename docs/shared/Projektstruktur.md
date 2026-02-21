# Projektstruktur – Ordner, Module und Konventionen

**Datum:** 2026-02-21  
**Ziel & Inhalt:** Überblick über Projektstruktur, Modulbereiche und die Zuordnung zentraler Scripts. Fokus: Navigierbarkeit und reproduzierbare Arbeitsweise.

---

## Überblick

Dieses Dokument beschreibt die Ordnerstruktur des Projekts sowie die wichtigsten Script-Bereiche (Datenpipeline, Modeling, Evaluation, Utilities). Es dient als Einstiegspunkt, um Dateien schnell zu finden und Verantwortlichkeiten einzuordnen.

---

## Ordnerstruktur (High-Level)

- `src/` – Quellcode (Datenpipeline, Modeling, Evaluation, Utilities)  
- `configs/` – YAML-Konfigurationen (Datasets, Modelle, Trainer/Evaluation)  
- `data/` – Rohdaten, Zwischenstände, verarbeitete Daten  
- `results/` – Run-Artefakte (Checkpoints, JSON-Summaries, Evaluation)  
- `logs/` – Logger-Ausgaben (z. B. metrics.csv, hparams.yaml)  
- `docs/` – Projektdokumentation (MkDocs)  

---

## Script-Übersicht (Zuordnung)

### Data / Preprocessing (`src/data/`)
- Laden, Bereinigung, Feature Engineering und Erstellung modellfertiger Splits/Spezifikationen  

### Modeling (`src/modeling/`)
- Modelltraining (z. B. TFT) und modellnahe Datenvorbereitung (z. B. Spezifikationen)  

### Evaluation (`src/evaluation/`)
- Auswertung abgeschlossener Runs auf Validation/Test  

### Utilities (`src/utils/`)
- Konfigurationsladen, Export von Run-Artefakten, Helper-Funktionen  

---

## Konventionen

- Konfigurationsgetrieben: Parameter stammen aus YAML, keine stillen Defaults im Code  
- Reproduzierbarkeit: Seeds/Run-IDs/Artefakte werden konsequent protokolliert  
- Artefaktorientiert: Training/Evaluation erzeugen standardisierte Dateien zur Weiterverarbeitung
