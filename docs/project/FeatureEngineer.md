# FeatureEngineer – Zweck und Funktionsweise

## Überblick
Die Klasse **FeatureEngineer** bereitet den Datensatz so vor, dass er vom **Temporal Fusion Transformer (TFT)** zeitlich verstanden werden kann. Sie erweitert den DataFrame um zusätzliche Spalten (Features), die zeitliche und strukturelle Informationen enthalten – z. B. Kalendermerkmale, Wochenendkennzeichnung, Feiertage und einen fortlaufenden Zeitindex. 

Das Modul arbeitet auf den bereinigten Daten (`train_cleaned.parquet`) und speichert das Ergebnis als `train_features.parquet` im Ordner `data/processed/`.

---

## Ziel
Ziel ist es, zeitliche Muster explizit und maschinenlesbar darzustellen, damit der TFT wiederkehrende Zusammenhänge erkennen kann, etwa:
- **Saisonale Muster** (z. B. höhere Verkäufe im Dezember),
- **Wochenmuster** (z. B. geringere Verkäufe am Wochenende),
- **Kalenderabhängige Trends** (z. B. Feiertagseffekte),
- **Zeitliche Abfolge** (über den numerischen Index `time_idx`).

Die Feature-Erweiterung stellt sicher, dass der TFT Zeitabhängigkeiten korrekt modelliert, auch wenn sie nicht direkt aus dem Datum ersichtlich sind.

---

## Hauptmethoden

### `add_calendar_features(df)`
- Fügt klassische Kalendermerkmale hinzu: `year`, `month`, `day`, `dayofweek`, `weekofyear`, `is_weekend`.
- `is_weekend` = 1, wenn Wochentag Samstag (5) oder Sonntag (6) ist.
- Grundlage für saisonale und wöchentliche Muster.

### `add_time_index(df)`
- Erstellt einen fortlaufenden numerischen Index (`time_idx`), beginnend mit 0 am frühesten Datum.
- Wird für das Sequenzverständnis des TFT benötigt.

### `add_holiday_features_de(df)`
- Markiert bundesweite deutsche Feiertage (keine länderspezifischen Varianten) mit einem Flag `is_holiday_de`.
- Verwendet die Bibliothek `holidays`, die die gesetzlichen Feiertage pro Jahr automatisch generiert.
- Optional kann zusätzlich der Name des Feiertags (`holiday_name`) gespeichert werden (Parameter `include_holiday_name=True`).
- Beispiele für berücksichtigte Feiertage:

| Feiertag | Typisches Datum | Bedeutung |
|-----------|----------------|------------|
| Neujahr | 1. Januar | Jahresbeginn |
| Karfreitag | variabel (März/April) | christlicher Feiertag |
| Ostermontag | variabel (März/April) | christlicher Feiertag |
| Tag der Arbeit | 1. Mai | gesetzlicher Feiertag |
| Christi Himmelfahrt | variabel (Mai) | kirchlicher Feiertag |
| Pfingstmontag | variabel (Mai/Juni) | christlicher Feiertag |
| Tag der Deutschen Einheit | 3. Oktober | nationaler Feiertag |
| 1. Weihnachtstag | 25. Dezember | Weihnachten |
| 2. Weihnachtstag | 26. Dezember | Weihnachten |

### `transform(df)`
- Führt alle Schritte in sinnvoller Reihenfolge aus:
  1. Kalendermerkmale hinzufügen,
  2. Zeitindex erstellen,
  3. Feiertagsmerkmale ergänzen.
- Gibt den erweiterten DataFrame zurück.

---

## Beispielnutzung
```python
from src.data.feature_engineering import FeatureEngineer
import pandas as pd

df = pd.read_parquet("data/interim/train_cleaned.parquet")
fe = FeatureEngineer(date_col="date", include_holiday_name=False)
df_feats = fe.transform(df)
```

Die erzeugte Datei wird typischerweise als `data/processed/train_features.parquet` gespeichert.

---

## Designprinzipien
- **Kleine, klar abgegrenzte Methoden** – jede erfüllt genau eine Aufgabe.
- **Keine Seiteneffekte** – Originaldaten bleiben unverändert.
- **Erweiterbarkeit** – zusätzliche Features (z. B. Lag- oder Rolling-Features, Promotions) können später ergänzt werden.
- **Kompatibilität mit TFT** – erzeugt genau die Struktur, die `TimeSeriesDataSet` benötigt.
- **Klarer Datenfluss:** `train_cleaned.parquet → FeatureEngineer → train_features.parquet`

---

## Beispielausgabe
Nach der Transformation enthält der Datensatz u. a. folgende Spalten:
```
['date', 'country', 'store', 'book', 'num_sold',
 'year', 'month', 'day', 'dayofweek', 'weekofyear', 'is_weekend',
 'time_idx', 'is_holiday_de']
```
Optional (bei Aktivierung): zusätzlich `holiday_name`.

Diese Struktur bildet die Grundlage für die Erstellung des `TimeSeriesDataSet` und ermöglicht dem TFT, saisonale und kalendarische Effekte korrekt zu modellieren.
