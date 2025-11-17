# ModelDataset – Erstellung eines modellfertigen Datensatzes

**Datum:** 2025-11-17  
**Script:** `src/modeling/model_dataset.py`  
**Ziel & Inhalt:** Zeitbasierte Aufteilung der aufbereiteten Daten in Train-, Validation- und Testbereiche sowie Erzeugung eines Metadatenmanifests.

---

## Überblick
`model_dataset.py` strukturiert die vorbereiteten Zeitreihen in drei zeitlich sortierte Teilmengen. Der Schritt erzeugt keine neuen Features, sondern organisiert die Daten für ein reproduzierbares Modelltraining.

---

## Ziel
Ziel ist eine klare und nachvollziehbare Datenaufteilung entlang der Zeitachse, um getrennte Bereiche für Training, Hyperparameterabstimmung und finale Bewertung zu definieren.

---

## Eingaben & Ausgaben

**Eingabe:**  
- `train_features_cyc_lag.parquet` (oder eine andere final vorbereitete Datei)

**Ausgaben:**  
- `train.parquet`  
- `val.parquet`  
- `test.parquet`  
- `manifest.json` (Informationen zu Zeitgrenzen, Zeilenzahlen und Spalten)

---

## Vorgehen

### 1. Einlesen und Sortieren  
Der Datensatz wird geladen und nach Zeitspalte (`TIME_COL`) sowie Identitätsspalten (`ID_COLS`) sortiert.

### 2. Bestimmung der Zeitgrenzen  
Die Splitgrenzen werden entweder  
- über feste Datumswerte (`VAL_START`, `TEST_START`) oder  
- über Verhältnisangaben (`SPLIT_RATIOS`) festgelegt.

### 3. Zeitbasierter Split  
Die Daten werden entlang der Zeitachse in drei aufeinanderfolgende Zeitbereiche geteilt.

### 4. (Optional) Skalierung  
Falls in `SCALE_COLS` angegeben, werden ausgewählte numerische Spalten gruppenweise z-standardisiert.

### 5. Schreiben der Ausgaben  
Die Teilmengen sowie das Manifest mit Metadaten werden in das Zielverzeichnis gespeichert.

---

## Konfiguration
Die relevanten Parameter werden in `config.py` definiert:

- `TIME_COL`  
- `ID_COLS`  
- `TARGET_COL`  
- `VAL_START`, `TEST_START`  
- `SPLIT_RATIOS`  
- `SCALE_COLS`  
- `DATASETS_DIR`

---

## Beispielaufruf
```bash
python -m src.modeling.model_dataset
```

---

## Ergebnis und Nutzen
- klar definierte zeitliche Strukturierung der Daten  
- reproduzierbare Splits für Training, Validation und Test  
- Manifest zur Dokumentation aller relevanten Metadaten  
