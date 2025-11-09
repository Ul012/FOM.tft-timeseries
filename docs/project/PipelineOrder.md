# Pipeline Overview – FOM.tft-timeseries

## Ziel
Diese Übersicht beschreibt die **Ausführungsreihenfolge** der zentralen Module – von Rohdaten bis Evaluation.  
Hinweis: Schritt **3** kann auf zwei Arten erfolgen: **(A) getrennte Skripte** oder **(B) integriertes Sammelskript**.

---

## Projektlogik

| Ebene | Ordner | Aufgabe |
|------|--------|---------|
| **data** | `src/data/` | Laden/Bereinigen, Feature-Erzeugung (Kalender, Feiertage, Lags, zyklische Merkmale) |
| **modeling** | `src/modeling/` | Datensatz für TFT aufbereiten, TFT trainieren |
| **evaluation** | `src/evaluation/` | Auswertung, Visualisierung, Artefakte/JSONs schreiben |

---

## Pipeline-Reihenfolge (präzise)

| # | Modul | Beschreibung | Input | Output | Hinweis |
|---:|------|--------------|-------|--------|--------|
| 1 | `data_alignment.py` *(optional)* | Skaliert/normalisiert Zeitreihen auf ein Vergleichsniveau. | `data/raw/*.csv` | `data/interim/train_aligned.parquet` | Nur falls nötig. |
| 2 | `data_cleaning.py` *(optional)* | Bereinigt Ausreißer, imputiert fehlende Werte. | Schritt 1 oder `data/raw/*.csv` | `data/interim/train_cleaned.parquet` | Optional. |
| 3A | `feature_engineering.py` | **Kalender-Features**, **time_idx**, **deutsche Feiertage** erzeugen. | `data/interim/train_cleaned.parquet` | `data/processed/train_features.parquet` | Separates Basisskript. |
| 3B | `cyclical_encoder.py` | **Zyklische Sin/Cos-Kodierungen** (z. B. dow/month/hour) hinzufügen. | `train_features.parquet` | `train_features_cyc.parquet` (oder in-place) | Nach 3A ausführen, falls getrennt gefahren. |
| 3C | **ODER:** `features.py` | **Integrierter Weg**: führt 3A (**FeatureEngineer**) und 3B (**CyclicalEncoder**) sowie optionale **Lag/Rolling-Features** in **einem Lauf** aus. | `data/raw/train.csv` & `test.csv` **oder** `dataset.csv` | `data/processed/features_train.parquet`, `features_test.parquet` | Empfohlen, wenn du alles zentral steuern willst. |
| 4 | `model_dataset.py` | Zeitbasierter Split (Train/Valid/Test), Metadaten schreiben. | Features aus 3A+3B **oder** 3C | `train.parquet`, `valid.parquet`, `test.parquet`, `meta.json` |  |
| 5 | `dataset_tft.py` | TFT-Datensatz bauen (Encodings/Scaler fit auf Train, apply auf Val/Test). | Schritt 4 | `model_ready/{train,valid,test}.parquet`, `schema.json`, `scaler.json`, `categories.json` | Leakage-sicher. |
| 6 | `trainer_tft.py` | TFT trainieren, Logs/Checkpoints schreiben; nach Fit: **results/evaluation/<run_id>/{results.json, summary.json}** erzeugen. | Schritt 5 + `configs/*.yaml` | `logs/…`, `checkpoints/…`, `results/evaluation/<run_id>/*` | JSON-Export integriert. |
| 7 | `evaluate.py` | Kennzahlen/Plots/Test-Eval. | Modell + Daten | `results/evaluation/*` |  |
| 8 | `viz_predictions.py` *(optional)* | Prognosen vs. Istwerte visualisieren. | Eval-Artefakte | PNGs |  |

---