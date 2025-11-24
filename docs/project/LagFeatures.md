# LagFeatures – Zweck und Funktionsweise

**Datum:** 2025-11-24 (aktualisiert)  
**Script:** `src/data/lag_features.py`  
**Ziel & Inhalt:** Erstellung konfigurierbarer Lag- und Rolling-Features mit NaN-Handling und Gruppen-Filterung.

---

## Überblick

Das Modul erzeugt zeitliche Merkmale auf Basis der vorhandenen Daten:

- **Lag-Features** – Vergangene Werte der Zielvariable
- **Rolling-Features** – Gleitende Statistiken (z.B. Mittelwert)
- **NaN-Handling** – Median-Imputation + Missing-Indicator
- **Gruppen-Filterung** – Zu kurze Zeitreihen entfernen

**Input:** `train_features_cyc.parquet`  
**Output:** `train_features_cyc_lag.parquet`

---

## Konfiguration (YAML)

```yaml
preprocessing:
  - step: "lag_features"
    enabled: true
    params:
      min_group_length: <int>    # Minimale Gruppenlänge (optional)
      lags: [<int>, ...]         # Lag-Schritte
      roll_windows: [<int>, ...] # Rolling-Fenstergrößen (optional)
      roll_stats: ["mean", ...]  # Statistiken pro Fenster (optional)
      prefix: "lag_"             # Spaltenpräfix
```

### Parameter

| Parameter | Typ | Beschreibung |
|-----------|-----|--------------|
| `min_group_length` | Int | Gruppen mit weniger Zeitschritten werden entfernt |
| `lags` | Liste | Zeitliche Verzögerungen |
| `roll_windows` | Liste | Fenstergrößen für Rolling-Features |
| `roll_stats` | Liste | Statistiken (z.B. `["mean"]`) |
| `prefix` | String | Spaltenpräfix (default: `"lag_"`) |

---

## Vorgehen

### 1. Gruppen-Filterung

Entfernt Gruppen die kürzer als `min_group_length` sind. Dies ist wichtig, damit:
- Genug Historie für Encoder vorhanden ist
- Genug Daten für Val/Test-Split bleiben

```python
filter_short_groups(df, min_length)
```

### 2. Lag-Features

Für jedes konfigurierte Lag L wird eine neue Spalte erzeugt:

```
lag_L(g, t) = target(g, t - L)
```

Wobei `g` die Gruppe und `t` der Zeitpunkt ist.

### 3. Rolling-Features

Gleitende Kennzahlen auf Basis vergangener Werte:

```
lag_W_stat(g, t) = stat(target(g, t-1), ..., target(g, t-W))
```

Die Berechnung erfolgt mit `shift(1)`, damit keine Zukunftsinformation einfließt.

### 4. NaN-Handling

Best Practice laut PyTorch Forecasting FAQ:

1. **Missing-Indicator erstellen:** `<lag_col>_missing` (1 = war NaN, 0 = hatte Wert)
2. **Median-Imputation:** 
   - Erst Gruppen-Median
   - Dann globaler Median als Fallback
   - Zuletzt 0 falls noch NaN

---

## Beispielaufruf

```bash
$env:DATASET_CONFIG="configs/datasets/<dataset>.yaml"
python -m src.data.lag_features
```

---

## Frequenz-spezifische Empfehlungen

| Frequenz | Typische Lags | Roll-Windows | min_group_length |
|----------|---------------|--------------|------------------|
| Täglich | 1, 7, 14, 30 | 7, 14 | encoder + prediction |
| Wöchentlich | 1, 4, 8, 12 | 4, 8 | encoder + prediction |
| Monatlich | 1, 3, 6, 12 | 3, 6 | encoder + prediction |

**Hinweis:** Bei Jahres-Lag (z.B. `lag_365` für tägliche Daten) muss `min_group_length` entsprechend hoch sein (z.B. 400).

---

## Ergebnis und Nutzen

- Explizite Kodierung vergangener Werte
- Robuste Abbildung kurzfristiger Trends über Rolling-Statistiken
- Zu kurze Gruppen werden automatisch gefiltert
- NaN-freie Features durch Median-Imputation + Missing-Indicator
- Flexible, zentral konfigurierbare Feature-Erzeugung