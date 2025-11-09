# Pipeline Overview – FOM.tft-timeseries

## Ziel
Diese Übersicht beschreibt die logische Ausführungsreihenfolge aller zentralen Module im Projekt **FOM.tft-timeseries**.  
Sie dient als Orientierung für den Workflow von der Datenbasis bis zur Evaluation, ohne dass Nummern in Dateinamen erforderlich sind.

---

## Projektlogik

| Ebene | Ordner | Aufgabe |
|--------|---------|----------|
| **data** | `src/data/` | Laden und Bereinigen der Rohdaten, Feature-Erzeugung (zeitliche, zyklische, lag-basierte Merkmale) |
| **modeling** | `src/modeling/` | Aufbau und Training der Modelle (z. B. Temporal Fusion Transformer) |
| **evaluation** | `src/evaluation/` | Bewertung und Visualisierung der Ergebnisse |

---

## Pipeline-Reihenfolge

| Schritt | Modul | Beschreibung | Input | Output |
|---|---|---|---|---|
| 1 | `data_alignment.py` *(optional)* | Skaliert historische Verkaufszahlen oder Zeitreihen auf ein einheitliches Vergleichsniveau. | `train.csv` | `train_aligned.parquet` |
| 2 | `data_cleaning.py` *(optional)* | Bereinigt Ausreißer und imputiert fehlende Werte. | `train_aligned.parquet` oder `train.csv` | `train_cleaned.parquet` |
| 3 | `features.py` | Erstellt **alle Features** in einem Lauf: Zeit-Features, Lag/Rolling-Werte (intern), sowie zyklische Sin/Cos-Kodierungen. Unterstützt zwei Dateien (*train/test*) oder eine Gesamtdatei mit internem Split. | `train.csv`/`test.csv` **oder** `dataset.csv` | `features_train.parquet`, `features_test.parquet` |
| 4 | `model_dataset.py` | Führt den zeitbasierten Split auf **Train/Valid** durch und leitet Testdaten harmonisiert weiter. Erstellt `meta.json` mit Datensatz-Infos. | `features_train.parquet`, `features_test.parquet` | `train.parquet`, `valid.parquet`, `test.parquet`, `meta.json` |
| 5 | `dataset_tft.py` | Bereitet die Daten modellgerecht auf: kategorische Kodierung (Mapping aus Train), numerische Skalierung (fit auf Train, transform auf Valid/Test). Speichert Artefakte wie Schema, Scaler und Kategorien. | `train/valid/test.parquet`, `meta.json` | `model_ready/{train,valid,test}.parquet`, `schema.json`, `scaler.json`, `categories.json` |
| 6 | `trainer_tft.py` | Startet das Training des Temporal Fusion Transformer auf Basis der `model_ready`-Daten. Nutzt Parameter aus `configs/*.yaml` und Konstanten aus `config.py`. | `model_ready/*`, `configs/*.yaml` | `model.pt`, Logs |
| 7 | `evaluate.py` | Berechnet Kennzahlen und erzeugt Plots auf Basis der Validierungs- und Testdaten. | `model.pt`, Valid/Test-Daten | `evaluation_results.json`, PNGs |
| 8 | `viz_predictions.py` *(optional)* | Visualisiert Vorhersagen im Vergleich zu den Ist-Werten. | `evaluation_results.json` | PNGs |

---

## Hinweise
- Bis einschließlich Schritt 5 werden Artefakte im **Parquet-Format** und begleitende **JSON-Metadaten** erzeugt.  
- `features.py` integriert alle Feature-Schritte: Zeit-, Lag- und zyklische Features. Separate Module wie `cyclical_encoding.py` oder `lag_features.py` sind nicht mehr erforderlich.  
- Der **Trainer** liest zur Laufzeit YAML-Konfigurationen aus dem Ordner `configs/` und kombiniert sie mit Konstanten aus `src/config.py`.  
- Alle Verarbeitungsschritte sind **leakage-sicher**: Transformationen und Parameter werden ausschließlich auf dem Trainings-Split bestimmt und anschließend auf Validierung und Test angewendet.  
- Evaluations- und Visualisierungsroutinen befinden sich im Ordner `src/evaluation/` und speichern Ergebnisse in `results/` und `plots/`.
