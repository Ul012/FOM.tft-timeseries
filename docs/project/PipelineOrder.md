# Pipeline Overview – FOM.tft-timeseries

**Datum:** 2025-11-15  
**Script:** –  
**Ziel & Inhalt:** Gibt eine vollständige Übersicht über die Pipeline-Reihenfolge des Projekts. Beschreibt Input, Output und Zweck aller Module von Alignment bis Training und optionaler Evaluation.


## Ziel
Diese Übersicht beschreibt die **Ausführungsreihenfolge** der zentralen Module – von Rohdaten bis Training und Evaluation.  
Alle Schritte können einzeln getestet werden. Schritte 1 – 6 bilden die Hauptpipeline.

---

## Projektlogik

| Ebene | Ordner | Aufgabe |
|------|--------|---------|
| **data** | `src/data/` | Laden, Bereinigen und Feature-Erzeugung (Kalender, Feiertage, Lags, zyklische Merkmale) |
| **modeling** | `src/modeling/` | TFT-spezifische Datenaufbereitung, Modelltraining |
| **evaluation** | `src/evaluation/` | Ergebnisanalyse, Kennzahlen und Visualisierung |
| **utils** | `src/utils/` | Hilfsfunktionen (Checkpoint-Handling, Metriken, Visualisierung) – **nicht Teil der Pipeline** |

---

## Pipeline-Reihenfolge

| # | Modul | Beschreibung | Input | Output | Hinweis |
|---:|------|--------------|-------|--------|--------|
| 1 | `data_alignment.py` *(optional)* | Skaliert/normalisiert Zeitreihen auf ein Vergleichsniveau. | `data/raw/*.csv` | `data/interim/train_aligned.parquet` | Nur falls nötig. Visualisierung zur Kontrolle: `src/visualization/data_alignment_plot.py`. |
| 2 | `data_cleaning.py` *(optional)* | Bereinigt Ausreißer und glättet den Lockdown-Zeitraum (Daily-Imputation je Zeitreihe). | Schritt 1 oder `data/raw/*.csv` | `data/interim/train_cleaned.parquet` | Optionale Visualisierung: `data_cleaning_overview.py`, `data_cleaning_compare.py`, `data_cleaning_diff.py`. |
| 3A | `feature_engineering.py` | **Kalender-Features**, `time_idx`, **deutsche Feiertage**. | `data/interim/train_cleaned.parquet` | `data/processed/train_features.parquet` | Basis-Feature-Set. |
| 3B | `cyclical_encoder.py` | **Zyklische Sin/Cos-Kodierungen** (z. B. dow, month). | `train_features.parquet` | `train_features_cyc.parquet` | Nach 3A ausführen. |
| 3C | `lag_features.py` | **Lag- und Rolling-Features** (`lag_1`, `lag_7`, …) per `groupby().shift()`. | `train_features_cyc.parquet` | `train_features_cyc_lag.parquet` | Ersetzt das frühere `features.py`. |
| 4 | `model_dataset.py` | Zeitbasierter Split (Train/Val/Test), Metadaten schreiben. | Ergebnis aus 3C | `train.parquet`, `val.parquet`, `test.parquet`, `meta.json` | Pflichtschritt. |
| 5 | `dataset_tft.py` | TFT-Datensatz erstellen (Featurelisten known/unknown/static automatisch). | Schritt 4 | `model_ready/{train,val,test}.parquet`, `dataset_spec.json` | Erkennt `lag_`-Spalten automatisch. |
| 6 | `trainer_tft.py` | TFT-Training nach Config/YAML, Logs & Checkpoints. | Schritt 5 + `configs/*.yaml` | `logs/tft/...`, `checkpoints/...`, `results/evaluation/<run_id>/*.json` | Kerntraining. |
| 7 | `load_trained_tft.py` *(optional)* | Lädt bestes Checkpoint zur Inferenz oder Analyse. | Checkpoint aus 6 | – | Werkzeug-Skript aus `src/utils/`. |
| 8 | `evaluate.py` *(optional)* | Bewertet Ergebnisse, berechnet Kennzahlen, aggregiert JSONs. | Resultate aus 6/7 | `results/evaluation/*` | Optional für Vergleiche. |
| 9 | `viz_predictions.py` *(optional)* | Visualisiert Prognosen vs. Istwerte. | Eval-Artefakte | PNGs | Optional. |


---
