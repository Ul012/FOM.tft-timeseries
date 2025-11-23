# LoadRaw – Laden und Mergen von Rohdaten

**Datum:** 2025-11-23  
**Script:** `src/data/load_raw.py`  
**Ziel & Inhalt:** Beschreibt das Laden von Rohdaten aus CSV-Dateien mit optionalem Merge mehrerer Dateien. Erklärt Single-File- vs. Multiple-Files-Modus, Validierung und Output-Format.

---

## Überblick
Das Modul **LoadRaw** lädt die Rohdaten basierend auf der Dataset-Config und speichert sie als Parquet-Datei für die weitere Verarbeitung.

**Zwei Modi werden unterstützt:**
- **single_file:** Eine einzelne CSV-Datei (z.B. Booksales)
- **multiple_files:** Mehrere Dateien mit Merge (z.B. Walmart: train.csv + features.csv)

Eingabe: `data/raw/<dataset_name>/*.csv`  
Ausgabe: `data/interim/<dataset_name>/train_raw.parquet`

---

## Ziel
Ziel des LoadRaw-Schritts ist:

- einheitliches Laden verschiedener Datenquellen
- optionales Mergen mehrerer Dateien (z.B. Hauptdatei + Features)
- Spalten-Validierung gemäß Config
- konsistentes Parquet-Format als Basis für Preprocessing

---

## Funktionsweise

### 1. Single File Modus

**Verwendung:** Wenn alle Daten in einer Datei liegen

**Config-Beispiel (booksales.yaml):**
```yaml
raw_data:
  type: "single_file"
  files:
    - path: "data/raw/booksales/train.csv"
      role: "main"
```

**Vorgehen:**
- Lädt die CSV-Datei
- Validiert optionale Spalten
- Speichert als Parquet

---

### 2. Multiple Files Modus

**Verwendung:** Wenn Daten über mehrere Dateien verteilt sind

**Config-Beispiel (walmart.yaml):**
```yaml
raw_data:
  type: "multiple_files"
  files:
    - path: "data/raw/walmart/train.csv"
      role: "main"
      columns: ["Store", "Dept", "Date", "Weekly_Sales", "IsHoliday"]
    - path: "data/raw/walmart/features.csv"
      role: "features"
      columns: ["Store", "Date", "Temperature", "Fuel_Price", ...]
  merge:
    merge_on: ["Store", "Date"]
    how: "left"
    drop_from_right: ["IsHoliday"]
```

**Vorgehen:**
1. Lädt Hauptdatei (role: "main")
2. Lädt Feature-Dateien (role: "features")
3. Entfernt Duplikate gemäß `drop_from_right`
4. Merged Dateien auf gemeinsamen Spalten
5. Speichert als Parquet

---

## Spalten-Validierung

Falls in der Config `columns` definiert sind, werden die erwarteten Spalten validiert:

```python
expected_cols = ["Store", "Dept", "Date", "Weekly_Sales"]
missing = set(expected_cols) - set(df.columns)
if missing:
    raise ValueError(f"Fehlende Spalten: {missing}")
```

Dies verhindert Fehler in späteren Pipeline-Schritten.

---

## Merge-Optionen

### merge_on
Liste der Spalten für den Join (z.B. `["Store", "Date"]`)

**Wichtig:** Verwende `merge_on` statt `on` (YAML-Keyword-Problem!)

### how
Merge-Typ: `"left"`, `"right"`, `"inner"`, `"outer"` (Standard: `"left"`)

### drop_from_right
Liste von Spalten, die aus der rechten Datei entfernt werden sollen (z.B. Duplikate wie `IsHoliday`)

---

## Beispielaufruf

```bash
# Via Pipeline (empfohlen)
python -m src.pipeline --dataset configs/datasets/walmart.yaml --steps preprocessing

# Einzeln (mit Umgebungsvariable)
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.load_raw
```

**Output:**
```
[load_raw] Dataset: walmart
[load_raw] Type: multiple_files
[load_raw] Main-Datei: train.csv (Shape: (421570, 5))
[load_raw] Feature-Datei: features.csv (Shape: (8190, 12))
[load_raw] Nach Merge: (421570, 14)

✓ Raw-Daten geladen und gespeichert: data\interim\walmart\train_raw.parquet
  Zeilen: 421,570
  Spalten: 14
  Zeitraum: 2010-02-05 bis 2012-10-26
  Zeitreihen: 3,331
```

---

## Ergebnis und Nutzen

- Flexibles Laden verschiedener Datenquellen
- Automatisches Mergen bei Multi-File-Datasets
- Spalten-Validierung verhindert Fehler
- Konsistentes Parquet-Format für Preprocessing
- Einheitlicher Einstiegspunkt für alle Datasets