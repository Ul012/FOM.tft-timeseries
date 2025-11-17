# FeatureEngineer – Zweck und Funktionsweise

**Datum:** 2025-11-17  
**Script:** `src/data/feature_engineering.py`  
**Ziel & Inhalt:** Beschreibung der Erstellung von Kalender-, Feiertags- und Zeitindex-Features als Grundlage für die spätere TFT-Modellierung.

---

## Überblick
Der **FeatureEngineer** erweitert den bereinigten Datensatz um zusätzliche zeitliche Strukturmerkmale.  
Auf Basis der täglichen Verkaufsdaten werden unter anderem Kalendermerkmale, Wochenendkennzeichnung, ein fortlaufender Zeitindex sowie deutsche Feiertage erzeugt.

Eingabe: `data/interim/train_cleaned.parquet`  
Ausgabe: `data/processed/train_features.parquet`

---

## Ziel
Ziel ist die explizite und maschinenlesbare Darstellung zeitlicher Muster, damit nachfolgende Modelle wiederkehrende Zusammenhänge erkennen können:

- saisonale Effekte (z. B. höhere Verkäufe im Dezember)  
- Wochenmuster (z. B. Unterschiede zwischen Werktagen und Wochenende)  
- kalenderabhängige Effekte (z. B. gesetzliche Feiertage)  
- zeitliche Abfolge über einen numerischen Index (`time_idx`)

Damit wird die Basis für das spätere TimeSeriesDataSet und den Temporal Fusion Transformer geschaffen.

---

## Vorgehen

### 1. Kalendermerkmale (`add_calendar_features`)
Es werden klassische Kalender-Spalten ergänzt, zum Beispiel:

- `year`, `month`, `day`  
- `dayofweek` (0–6)  
- `weekofyear`  
- `is_weekend` (1 bei Samstag oder Sonntag, sonst 0)

Diese Merkmale beschreiben grobe saisonale und wöchentliche Strukturen.

---

### 2. Zeitindex (`add_time_index`)
Zusätzlich wird ein fortlaufender numerischer Index `time_idx` erstellt:

- Start bei 0 am frühesten Datum  
- aufsteigende Nummerierung je Zeitschritt  

Der Index dient als konsistente Zeitachse für das spätere TFT-Setup.

---

### 3. Deutsche Feiertage (`add_holiday_features_de`)
Mit Hilfe der Bibliothek `holidays` werden bundesweite deutsche Feiertage markiert:

- Flag-Spalte `is_holiday_de` (1/0)  
- optional: `holiday_name` (bei `include_holiday_name=True`)

Berücksichtigt werden gesetzliche Feiertage wie Neujahr, Karfreitag, Ostermontag, Tag der Arbeit, Tag der Deutschen Einheit sowie der 1. und 2. Weihnachtstag.

---

### 4. Gesamte Transformation (`transform`)
Die Methode `transform(df)` führt alle Schritte in definierter Reihenfolge durch:

1. Kalendermerkmale hinzufügen  
2. Zeitindex erzeugen  
3. Feiertagsmerkmale ergänzen  

Das Ergebnis ist ein erweiterter DataFrame mit allen relevanten Zeitmerkmalen.

---

## Beispielaufruf

```python
from src.data.feature_engineering import FeatureEngineer
import pandas as pd

df = pd.read_parquet("data/interim/train_cleaned.parquet")
fe = FeatureEngineer(date_col="date", include_holiday_name=False)
df_feats = fe.transform(df)
df_feats.to_parquet("data/processed/train_features.parquet")
```

---

## Ergebnis und Nutzen

- strukturierte zeitliche Beschreibung der Daily-Daten  
- explizite Kodierung von Kalender-, Wochenend- und Feiertagseffekten  
- einheitlicher Zeitindex für die spätere Sequenzmodellierung  
- konsistenter Input für das nachfolgende TFT- und Dataset-Setup  
