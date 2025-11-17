# LagFeatures – Zweck und Funktionsweise

**Datum:** 2025-11-16  
**Script:** `src/data/lag_features.py`  
**Ziel & Inhalt:** Beschreibung der Erstellung konfigurierbarer Lag- und Rolling-Features zur Abbildung vergangener Werte und lokaler Trends.

---

## Überblick
Das Modul erzeugt zusätzliche zeitliche Merkmale auf Basis der vorhandenen Daily-Daten.  
Dazu gehören klassische Lag-Features sowie optionale Rolling-Statistiken.  
Die Eingabedatei ist `train_features_cyc.parquet`, die Ausgabe `train_features_cyc_lag.parquet`.

---

## Ziel
Die Lag-Features stellen dem Modell explizite Vergangenheitsinformationen bereit und unterstützen die Erfassung von:

- kurzfristigen Trends (z. B. über Rolling-Fenster)  
- wiederkehrenden Mustern wie Vorwoche oder Vortag  
- saisonalen Strukturen über definierte Lag-Schritte  

Die Feature-Erweiterung ist vollständig konfigurierbar und wird zentral über `LAG_CONF` gesteuert.

---

## Konfiguration (`LAG_CONF`)
In `src/config.py` definiert, typischer Aufbau:

```python
LAG_CONF = {
    "target_col": "num_sold",
    "lags": [1, 7, 14],
    "roll_windows": [7, 28],
    "roll_stats": ["mean"],
    "prefix": "lag_",
}
```

- **lags**: zeitliche Verzögerungen  
- **roll_windows**: Fenstergrößen für Rolling-Features  
- **roll_stats**: Kennzahlen pro Fenster (z. B. mean)  
- **prefix**: Spaltenpräfix

---

## Vorgehen

### 1. Sortierung nach Gruppen und Zeit
Vor der Berechnung werden die Daten nach den Gruppenspalten (`GROUP_COLS`) und dem Zeitfeld (`TIME_COL`) sortiert.  
Dies stellt sicher, dass Shifts und Rolling-Fenster korrekt angewendet werden.

---

### 2. Lag-Features
Für jedes in `lags` definierte Lag \(L\) wird eine neue Spalte erzeugt:

\[
	ext{lag}_L(g, t) = y(g, t - L)
\]

Beispiel: `lag_1`, `lag_7`, `lag_14`.

---

### 3. Rolling-Features (optional)
Falls Rolling-Fenster definiert sind, werden gleitende Kennzahlen auf Basis vergangener Werte berechnet.

Beispiel:

\[
	ext{lag\_7\_mean}(g, t) = 	ext{mean}(y(g, t-1), \dots, y(g, t-7))
\]

Die Berechnung erfolgt immer mit `shift(1)`, damit keine Zukunftsinformation einfließt.

---

### 4. Jahres-Lag (`lag_365`)
Der Jahres-Lag wird als reguläre Feature-Spalte erzeugt, nicht über den internen Lag-Mechanismus des `TimeSeriesDataSet`.  
Dies vermeidet Fälle, in denen Validierungsserien aufgrund fehlender Historie entfernt würden. Führt zu Fehlern wie:

```
filters should not remove all entries – check encoder/decoder lengths and lags
```

---

## Beispielaufruf

```python
import pandas as pd
from src.data.lag_features import add_lag_features
from src.config import PROCESSED_DIR

df = pd.read_parquet(PROCESSED_DIR / "train_features_cyc.parquet")
df_lag = add_lag_features(df)
```

---

## Ergebnis und Nutzen

- explizite Codierung vergangener Werte  
- robuste Abbildung kurzfristiger Trends über Rolling-Statistiken  
- flexible, zentral konfigurierbare Feature-Erzeugung  
- vollständiger Lag-Jahreswert (`lag_365`) ohne Einschränkungen beim Dataset  
