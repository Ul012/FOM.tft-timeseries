# FeatureEngineer – Zweck und Funktionsweise

**Datum:** 2025-11-26 (aktualisiert)  
**Script:** `src/data/feature_engineering.py`  
**Ziel & Inhalt:** Beschreibung der Erstellung von Kalender-, Feiertags- und Zeitindex-Features als Grundlage für die spätere TFT-Modellierung.

---

## Übersicht
Der **FeatureEngineer** erweitert den bereinigten Datensatz um zusätzliche zeitliche Strukturmerkmale.  
Auf Basis der Verkaufsdaten werden unter anderem Kalendermerkmale, Wochenendkennzeichnung, ein fortlaufender Zeitindex sowie Feiertage erzeugt.

Eingabe: `data/interim/<dataset>/train_cleaned.parquet`  
Ausgabe: `data/processed/<dataset>/train_features.parquet`

---

## Ziel
Ziel ist die explizite und maschinenlesbare Darstellung zeitlicher Muster, damit nachfolgende Modelle wiederkehrende Zusammenhänge erkennen können:

- saisonale Effekte (z. B. höhere Verkäufe im Dezember)  
- Wochenmuster (z. B. Unterschiede zwischen Werktagen und Wochenende)  
- kalenderabhängige Effekte (z. B. gesetzliche Feiertage)  
- zeitliche Abfolge über einen numerischen Index (`time_idx`)

Damit wird die Basis für das spätere TimeSeriesDataSet und den Temporal Fusion Transformer geschaffen.

---

## Vorgehen

### 1. ID-Spalten zu String konvertieren (NEU)

PyTorch Forecasting erwartet kategorische Variablen als Strings. Bei Datensätzen mit numerischen ID-Spalten (z.B. Walmart: `Store`, `Dept` als Integer) werden diese automatisch konvertiert:

```python
for col in self.id_cols:
    if col in out.columns and out[col].dtype in ["int64", "int32", "float64"]:
        out[col] = out[col].astype(str)
```

**Hinweis:** Bei Datensätzen mit String-IDs (z.B. Booksales) passiert nichts, da die Bedingung nicht greift.

---

### 2. Kalendermerkmale (`add_calendar_features`)
Es werden klassische Kalender-Spalten ergänzt:

- `year`, `month`, `day`  
- `dayofweek` (0–6, Montag=0)  
- `weekofyear` (ISO-Kalenderwoche 1–53)  
- `is_weekend` (1 bei Samstag oder Sonntag, sonst 0)

Diese Merkmale beschreiben grobe saisonale und wöchentliche Strukturen.

---

### 3. Zeitindex (`add_time_index`) – AKTUALISIERT

Der fortlaufende numerische Index `time_idx` wird frequenzunabhängig erstellt:

```python
unique_dates = out[self.date_col].drop_duplicates().sort_values().reset_index(drop=True)
date_to_idx = {d: i for i, d in enumerate(unique_dates)}
out["time_idx"] = out[self.date_col].map(date_to_idx).astype("int64")
```

**Warum diese Änderung?**

Die ursprüngliche Implementierung verwendete `dt.days`:
```python
# ALT (fehlerhaft für wöchentliche Daten)
out["time_idx"] = (out[self.date_col] - first_date).dt.days.astype("int64")
```

Dies erzeugte bei **wöchentlichen Daten** Lücken im Index:

| Datum | time_idx (ALT) | time_idx (NEU) |
|-------|----------------|----------------|
| 2010-02-05 | 0 | 0 |
| 2010-02-12 | 7 | 1 |
| 2010-02-19 | 14 | 2 |

TFT interpretierte die Lücken als fehlende Zeitschritte und filterte fast alle Gruppen heraus.

**Vorteil der neuen Implementierung:**
- Funktioniert für täglich, wöchentlich und monatlich
- Fortlaufender Index ohne Lücken
- Keine Anpassung pro Datensatz nötig

---

### 4. Feiertage (`add_holiday_features`)

Mit Hilfe der Bibliothek `holidays` werden länderspezifische Feiertage markiert:

- Flag-Spalte `is_holiday` (1/0)  
- optional: `holiday_name` (bei `include_holiday_name=True`)

**Unterstützte Länder:**
- `DE` – Deutschland (bundesweite Feiertage)
- `US` – USA (inkl. Thanksgiving, Super Bowl etc.)
- `EU` – Vereinfachte EU-Feiertage (basierend auf DE)

Die Konfiguration erfolgt über YAML:
```yaml
feature_engineering:
  params:
    country: "US"  # oder "DE"
    include_holiday_name: true
```

---

### 5. Custom Date Flags (`add_date_flags`)

Zusätzlich zu Feiertagen können eigene Perioden-Flags definiert werden:

```yaml
feature_engineering:
  params:
    date_flags:
      is_newyear:
        - {month: 12, day_start: 27, day_end: 31}
        - {month: 1, day_start: 1, day_end: 2}
```

Dies erzeugt eine Spalte `is_newyear` mit 1 für alle Tage in den definierten Perioden.

---

### 6. Gesamte Transformation (`transform`)

Die Methode `transform(df)` führt alle Schritte in definierter Reihenfolge durch:

1. **ID-Spalten zu String** (falls numerisch)
2. Kalendermerkmale hinzufügen  
3. Zeitindex erzeugen  
4. Feiertagsmerkmale ergänzen
5. Custom Date Flags hinzufügen

Das Ergebnis ist ein erweiterter DataFrame mit allen relevanten Zeitmerkmalen.

---

## Konfiguration über YAML

```yaml
preprocessing:
  - step: "feature_engineering"
    enabled: true
    params:
      country: "DE"              # DE, US, oder EU
      include_holiday_name: false
      date_flags:                # Optional
        is_newyear:
          - {month: 12, day_start: 27, day_end: 31}
          - {month: 1, day_start: 1, day_end: 2}
```

---

## Aufruf

```powershell
# Via Pipeline (empfohlen)
python -m src.pipeline --dataset configs/datasets/booksales.yaml --steps preprocessing

# Einzeln
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.feature_engineering
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.feature_engineering
```

---

## Ergebnis und Nutzen

- strukturierte zeitliche Beschreibung der Daten (täglich, wöchentlich, monatlich)
- explizite Kodierung von Kalender-, Wochenend- und Feiertagseffekten  
- frequenzunabhängiger Zeitindex für die spätere Sequenzmodellierung  
- automatische String-Konvertierung für PyTorch Forecasting-Kompatibilität
- konsistenter Input für das nachfolgende TFT- und Dataset-Setup