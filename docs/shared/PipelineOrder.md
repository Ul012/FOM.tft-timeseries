# Pipeline Overview – FOM.tft-timeseries

**Datum:** 2025-11-17  
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
| **visualization** | `src/visualization/` | Plots für Daten, Training und Evaluation |
| **utils** | `src/utils/` | Hilfsfunktionen (Checkpoint-Handling, Metriken, Visualisierung) – **nicht Teil der Pipeline** |

---

## Pipeline-Reihenfolge

| # | Modul | Beschreibung | Input | Output | Hinweis                                                                                                     |
|---:|------|--------------|-------|--------|-------------------------------------------------------------------------------------------------------------|
| 1 | `data_alignment.py` *(optional)* | Skaliert/normalisiert Zeitreihen auf ein Vergleichsniveau. | `data/raw/*.csv` | `data/interim/train_aligned.parquet` | Optionale Visualisierung: `src/visualization/data_alignment_plot.py`.                                       |
| 2 | `data_cleaning.py` *(optional)* | Bereinigt Ausreißer und glättet den Lockdown-Zeitraum (Daily-Imputation je Zeitreihe). | Schritt 1 oder `data/raw/*.csv` | `data/interim/train_cleaned.parquet` | Optionale Visualisierung: `data_cleaning_overview.py`, `data_cleaning_compare.py`, `data_cleaning_diff.py`. |
| 3A | `feature_engineering.py` | **Kalender-Features**, `time_idx`, **deutsche Feiertage**. | `data/interim/train_cleaned.parquet` | `data/processed/train_features.parquet` | Basis-Feature-Set.                                                                                          |
| 3B | `cyclical_encoder.py` | **Zyklische Sin/Cos-Kodierungen** (z. B. dow, month). | `train_features.parquet` | `train_features_cyc.parquet` |                                                                                                             |
| 3C | `lag_features.py` | **Lag- und Rolling-Features** (`lag_1`, `lag_7`, …) per `groupby().shift()`. | `train_features_cyc.parquet` | `train_features_cyc_lag.parquet` |                                                                                                             |
| 4 | `model_dataset.py` | Zeitbasierter Split (Train/Val/Test), Metadaten schreiben. | Ergebnis aus 3C | `train.parquet`, `val.parquet`, `test.parquet`, `meta.json` | Pflichtschritt.                                                                                             |
| 5 | `dataset_tft.py` | TFT-Datensatz erstellen (Featurelisten known/unknown/static automatisch). | Schritt 4 | `model_ready/{train,val,test}.parquet`, `dataset_spec.json` | Erkennt `lag_`-Spalten automatisch.                                                                         |
| 6 | `trainer_tft.py` | TFT-Training nach Config/YAML, Logs & Checkpoints. | Schritt 5 + `configs/*.yaml` | `logs/tft/...`, `checkpoints/...`, `results/evaluation/<run_id>/*.json` | Kerntraining.                                                                                               |
| 7 | `evaluate_tft.py` *(optional)* | Berechnet Fehlermaße (MAE, RMSE, MAPE, SMAPE) für Val/Test anhand eines Checkpoints. | Checkpoint aus 6 + `val/test.parquet` | `results/tft/eval/<run_id>/eval_summary.json` | Pro Run ein Evaluations-JSON. Optionale Visualisierung: `plot_tft_eval_comparison.py` und `plot_tft_forecast_series.py`                  |
| 8 | `aggregate_tft_eval.py` *(optional)* | Aggregiert alle `eval_summary.json` zu einer Tabelle. | Ordner aus 7 | `results/tft/eval/eval_overview.{csv,json}` | Grundlage für Run-Vergleiche.                                                                               |

Die Schritte 7–8 sind optional und können bei Bedarf nach dem Training ausgeführt werden, ohne die Modellartefakte zu verändern.


---
