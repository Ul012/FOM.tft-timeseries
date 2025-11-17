# Integration von ARIMA und Prophet

**Datum:** 2025-11-17  
**Ziel & Inhalt:** Beschreibung der geplanten Erweiterung der Modeling-Schicht um ARIMA- und Prophet-Modelle auf Basis der bestehenden TFT-Pipeline.

---

## Überblick
ARIMA und Prophet sollen als zusätzliche Modellvarianten integriert werden, basierend auf derselben Datenpipeline und identischen Train/Val/Test-Splits. Dadurch wird ein konsistenter Modellvergleich ermöglicht.

---

## Ziel
Ziel ist eine klar strukturierte Einbindung beider Modelle ohne Änderung der bestehenden Vorverarbeitung.

---

## Modelltrainer

### 1. Trainer für ARIMA (`trainer_arima.py`)
- Laden der vorbereiteten Daten  
- optional Filterung oder Aggregation  
- Training eines ARIMA-Modells  
- Speichern des Modells im Run-Ordner  

### 2. Trainer für Prophet (`trainer_prophet.py`)
- analog zu ARIMA  
- Training eines Prophet-Modells  
- Speichern im Run-Ordner  

---

## Evaluation
Für beide Modelle sind Evaluationsskripte analog zu `evaluate_tft.py` vorgesehen:

- Laden des Modells  
- Erzeugen von Vorhersagen  
- Berechnung von MAE, RMSE, MAPE, SMAPE  
- Speicherung der Ergebnisse  

---

## Ergebnis und Nutzen
- erweiterte Modeling-Schicht mit konsistenten Vorgaben  
- Vergleichbarkeit von TFT, ARIMA und Prophet auf identischer Datenbasis  
