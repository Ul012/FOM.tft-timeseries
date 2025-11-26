# AnalyzeDataset – Automatische Dataset-Analyse und YAML-Generierung

**Datum:** 2025-11-25  
**Script:** `src/data/analyze_dataset.py`  
**Ziel & Inhalt:** Automatische Analyse neuer Datensätze mit Erkennung von Zeitreihen-Eigenschaften, Gruppen-Spalten und TFT-Parametern. Generiert eine vorgeschlagene YAML-Konfiguration.

---

## Überblick
Das Modul **AnalyzeDataset** analysiert unbekannte Datensätze und erstellt automatisch eine passende Dataset-YAML:

- **Datetime-Erkennung** – Findet die Zeit-Spalte automatisch
- **Frequenz-Erkennung** – Täglich, wöchentlich oder monatlich
- **ID-Spalten-Erkennung** – String/Category automatisch, Integer interaktiv
- **Target-Vorschläge** – Basierend auf Spaltennamen und Datentypen
- **TFT-Parameter** – Encoder/Prediction Length, Lags, Rolling Windows
- **Split-Validierung** – Prüft ob Val/Test lang genug für TFT
- **Datenqualitäts-Check** – NaN, Inf, Ausreißer, negative Werte

Eingabe: Beliebige CSV- oder Parquet-Datei  
Ausgabe: `configs/datasets/<name>_proposed.yaml`

---

## Vorgehen

### 1. Datetime-Spalte erkennen
```python
detect_datetime_column(df)
```
- Prüft zuerst auf `datetime64` Dtype
- Versucht dann String-Spalten zu parsen

### 2. Frequenz erkennen
```python
detect_frequency(df, time_col)
```
- Berechnet Median der Zeitabstände
- Klassifiziert: täglich (1d), wöchentlich (7d), monatlich (30d)

### 3. ID-Spalten identifizieren
```python
detect_id_columns(df, time_col)
```
- **String/Category**: Automatisch als ID erkannt
- **Integer**: Interaktive Rückfrage an den Nutzer

### 4. Target vorschlagen
```python
suggest_target(df, time_col, id_cols)
```
- Sucht numerische Spalten mit Keywords: `sales`, `value`, `amount`, `revenue`, `sold`, `demand`

### 5. TFT-Parameter berechnen
```python
calculate_tft_params(group_stats, freq_days)
```

Frequenzbasierte Standardwerte:

| Frequenz | Encoder | Prediction | Lags | Rolling |
|----------|---------|------------|------|---------|
| Täglich | 60 (~2 Monate) | 7 (1 Woche) | [1, 7, 14, 30] | [7, 14] |
| Wöchentlich | 26 (~6 Monate) | 4 (~1 Monat) | [1, 4, 8, 12] | [4, 8] |
| Monatlich | 24 (2 Jahre) | 6 (6 Monate) | [1, 3, 6, 12] | [3, 6] |

### 6. Split-Validierung
Prüft ob die kürzeste Gruppe nach dem Split noch lang genug für TFT ist:
```
min_required = max_encoder_length + max_prediction_length
```

### 7. Datenqualitäts-Check
- NaN-Raten pro Spalte
- Inf-Werte
- Ausreißer (>5 Standardabweichungen)
- Negative Werte im Target

---

## Interaktiver Modus

Bei Integer-Spalten fragt der Script nach:

```
Mögliche Gruppen-Spalten gefunden:
  - Store: 45 verschiedene Werte
  - Dept: 81 verschiedene Werte

Ist 'Store' eine Gruppen-Spalte (z.B. Store-ID, Produkt-ID)? [Y/n]: y
Ist 'Dept' eine Gruppen-Spalte (z.B. Store-ID, Produkt-ID)? [Y/n]: y

✓ Gruppen-Spalten: Store, Dept
```

---

## Beispielaufruf

```powershell
# Interaktiv (default)
python -m src.data.analyze_dataset --path data/raw/walmart/train.csv
python -m src.data.analyze_dataset --path data/raw/booksales/train.csv

# Mit explizitem Namen
python -m src.data.analyze_dataset --path data/raw/walmart/train.csv --name walmart

# Ohne Rückfragen (alle Integer-Kandidaten als ID annehmen)
python -m src.data.analyze_dataset --path data/raw/walmart/train.csv --no-interactive
```

---

## Parameter

| Parameter | Beschreibung |
|-----------|--------------|
| `--path` | Pfad zur CSV/Parquet-Datei (erforderlich) |
| `--name` | Dataset-Name (optional, wird sonst abgeleitet) |
| `--no-interactive` | Keine Rückfragen, alle Integer-Kandidaten als ID |

---

## Beispielausgabe

```
======================================================================
Dataset-Analyse: walmart
Datei: data/raw/walmart/train.csv
Zeilen: 421,570 | Spalten: 5 | Memory: 16.1 MB

Zeitreihen-Eigenschaften:
  time_col: Date (2010-02-05 bis 2012-10-26)
  Frequenz: weekly (erkannt)
  Zeitschritte: 143 weekly

Gruppen (schema.id_cols):
  Erkannt (String/Category): keine
  
  Mögliche Gruppen-Spalten gefunden:
    - Store: 45 verschiedene Werte
    - Dept: 81 verschiedene Werte

Target-Vorschläge:
  1. Weekly_Sales (numerisch, Keyword: sales)

TFT-Parameter (wöchentlich):
  max_encoder_length: 26 (~6 Monate Historie)
  max_prediction_length: 4 (~1 Monat Vorhersage)
  Empfohlene Lags: [1, 4, 8, 12]
  
✅ Vorgeschlagene Config erstellt: configs/datasets/walmart_proposed.yaml
======================================================================
```

---

## Generierte YAML-Struktur

```yaml
name: "walmart"
description: "Auto-generated config"

paths:
  raw: "data/raw/walmart"
  interim: "data/interim/walmart"
  processed: "data/processed/walmart"

schema:
  time_col: "Date"
  id_cols: ["Store", "Dept"]
  target_col: "Weekly_Sales"

preprocessing:
  - step: "load_raw"
    enabled: true
  - step: "cleaning"
    enabled: true
    params:
      clip_target_min: 0
      remove_nan: true
  - step: "feature_engineering"
    enabled: true
    params:
      country: "DE"
  - step: "lag_features"
    enabled: true
    params:
      lags: [1, 4, 8, 12]
      roll_windows: [4, 8]
      min_group_length: 35

split:
  method: "ratio"
  ratios: [0.80, 0.10, 0.10]

tft:
  max_encoder_length: 26
  max_prediction_length: 4
```

---

## Warnungen

Der Script gibt Warnungen aus bei:

- **Zu kurze Gruppen**: Gruppen mit weniger Zeitschritten als `min_group_length`
- **Hohe NaN-Raten**: Spalten mit >50% fehlenden Werten
- **Datenqualitätsprobleme**: Negative Werte, Inf, Ausreißer im Target
- **Split-Probleme**: Val/Test zu kurz für TFT-Anforderungen

---

## Ergebnis und Nutzen

- Schneller Einstieg bei neuen Datensätzen
- Automatische Erkennung aller relevanten Eigenschaften
- Vorgeschlagene TFT-Parameter basierend auf Frequenz
- Validierung der Datenqualität vor dem Training
- Basis-YAML, die manuell verfeinert werden kann
