# MLflow – Konzeptskizze für das TFT-TimeSeries-Projekt

**Datum:** 2025-11-18  
**Ziel & Inhalt:** Überblick über die geplante Nutzung von MLflow als Experiment-Tracking-Tool innerhalb des Projekts.

---

## Überblick
MLflow bietet experimentelles Tracking, Vergleichbarkeit von Modellvarianten und konsistente Protokollierung von Parametern, Metriken und Modellen. Das Dokument skizziert eine mögliche zukünftige Einbindung.

---

## Ziel
Ziel ist die Definition potenzieller Einsatzbereiche von MLflow, ohne eine konkrete Implementierung festzulegen.

---

## Anwendungsbereiche

### Hauptfunktionen
- Protokollierung von Hyperparametern und Metriken  
- Versionierung von Modellen  
- Vergleich verschiedener Modellvarianten  
- optionale Integration in Hyperparameter-Suchen (Optuna)  
- zukünftige Erweiterbarkeit für Deployment-Szenarien  

### Einsatz in der Pipeline

#### 1. Preprocessing Tracking
- Protokollierung von Split-Parametern (val_start, test_start, ratios)
- Tracking von Feature-Engineering-Schritten (Lags, zyklische Features)
- Dokumentation der Datenqualität (Zeilenanzahlen, NaN-Behandlung)
- Versionierung der aufbereiteten Datensätze

#### 2. Modeling Tracking
- Logging aller Trainings-Hyperparameter (Learning Rate, Batch Size, Epochen)
- Automatisches Tracking von Modellarchitekturen (Hidden Size, Attention Heads)
- Speicherung von Trainingsmetriken (Loss-Kurven, Lernrate)
- Versionierung von Checkpoints und Modellgewichten
- Integration mit bestehenden YAML-Konfigurationen

#### 3. Evaluation Tracking
- Zentrale Speicherung von Evaluationsmetriken (MAE, RMSE, MAPE, SMAPE)
- Vergleichbarkeit über verschiedene Runs hinweg
- Automatische Verknüpfung von Evaluation mit zugehörigem Training-Run
- Tracking von Inferenz-Parametern (Batch Size für Evaluation)

---

## Architektur-Überlegungen

### Config-Verwaltung

**Aktueller Stand:**
- Training: YAML-basierte Konfiguration (`configs/models/tft/*.yaml`)
- Dataset: Separate YAML pro Datensatz (`configs/datasets/*.yaml`)
- Evaluation: Nutzt `summary.json` aus Training-Run (keine separate Config nötig)
- Alle Parameter werden in `summary.json` dokumentiert

**Mit MLflow (geplant):**
- Automatisches Logging aller YAML-Parameter
- Run-Vergleiche über MLflow UI
- Versionierung von Experimenten
- Keine Änderung an YAML-Struktur nötig - MLflow erweitert nur das Tracking

### Multi-Model-Setup
- Einheitliche Tracking-Struktur für TFT, ARIMA, Prophet
- Vergleichbare Metriken über verschiedene Modelltypen
- Zentrale Experiment-Verwaltung

### Optuna-Integration
- Best-Trial-Parameter werden in MLflow gespeichert
- Hyperparameter-Suche wird als Parent-Run protokolliert
- Einzelne Trials als Child-Runs

---

## Technische Hinweise

### Batch-Size-Unterschiede
- Training: kleinere Batches (Speicher für Gradienten)
- Evaluation: größere Batches (kein Backprop, weniger Memory)
- MLflow trackt beide separat

### Aktueller Status (Stand: 2025-11-23)
- **Preprocessing:** JSON-Dateien dokumentieren Splits (`meta.json`, `dataset_spec.json`)
- **Training:** CSV-Logs via Lightning Logger, Checkpoints lokal in `results/tft/runs/`
- **Evaluation:** JSON-basierte Summary (`eval_summary.json`), Aggregation in `eval_overview.csv`
- **Geplant:** Schrittweise Migration zu MLflow ohne Breaking Changes an bestehenden Scripts

---

## Ergebnis und Nutzen
- strukturierte Grundlage für spätere Implementierung  
- klar abgegrenzte Einsatzfelder innerhalb des Projekts
- nahtlose Integration in bestehende Pipeline ohne Umbauten
- Vorbereitung für Hyperparameter-Optimierung und Multi-Model-Vergleiche