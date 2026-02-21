# Optuna – Hyperparameter-Suche (projektweit)

**Datum:** 2026-02-21  
**Ziel & Inhalt:** Einheitliche Beschreibung der Optuna-Integration für Modellläufe (TFT, ARIMA, Prophet). Fokus: Ablauf, Schnittstellen, Artefakte. Keine konkreten Hyperparameterwerte, keine Ergebnisdiskussion.

---

## Überblick

Optuna wird im Projekt eingesetzt, um konfigurationsgetriebene Suchläufe reproduzierbar auszuführen. Die modell-spezifische Trial-Logik ist getrennt vom Training/Evaluation implementiert, sodass Runs als standardisierte Artefakte abgelegt und anschließend identisch ausgewertet werden können.

---

## Eingaben

- Dataset-Config (für Datenpfade/Schema)  
- Modell-/Optuna-Config (Search Space, Trial-Parameter, Stoppkriterien)  
- Vorbereitete Splits und Spezifikationen gemäß Pipeline  

---

## Ablauf (generisch)

1. Erzeugung eines Trials (Sample aus Search Space)  
2. Training des jeweiligen Modells mit Trial-Konfiguration  
3. Speicherung der Run-Artefakte (Checkpoint, Logs, Summary)  
4. Bewertung des Trials über definierte Objective-Logik  
5. Speicherung der Study-Ergebnisse  

---

## Modell-spezifische Hinweise

### TFT
- Trials steuern ausschließlich YAML/Config-Werte (keine impliziten Defaults im Code).  
- Checkpoints/Logs werden je Trial im Ergebnisbaum abgelegt.  

### ARIMA
- Trial-Logik umfasst Modellordnung/Seasonality-Parameter, bleibt jedoch vollständig konfigurationsgetrieben.  

### Prophet
- Trial-Logik steuert Prophet-Konfigurationen über definierte Suchräume.  

---

## Ausgaben

- Study-Artefakte (Optuna-Storage/Logs je nach Konfiguration)  
- Run-Artefakte pro Trial im Ergebnisbaum (Checkpoints, Metriken, Summaries)  

---

## Ergebnis und Nutzen

- Reproduzierbare Modellkonfigurationen und Suchläufe  
- Einheitliche Ablage von Trial-Runs als vergleichbare Artefakte  
- Saubere Trennung zwischen Suche, Training und Evaluation
