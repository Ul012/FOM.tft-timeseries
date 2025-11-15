# LagFeatures – Zweck und Funktionsweise

**Datum:** 2025-11-15  
**Script:** src/data/lag_features.py  
**Ziel & Inhalt:** Dokumentiert die Erstellung konfigurierbarer Lag- und Rolling-Features. Zeigt Vorgehen, Konfiguration (LAG_CONF), Implementierungsdetails und Pipeline-Position.


## Überblick

Das Modul `lag_features.py` erzeugt auf Basis der bestehenden Zeitreihen zusätzliche **Lag- und Rolling-Features**.  
Diese Merkmale sind insbesondere für den Temporal Fusion Transformer (TFT) wichtig, um Trägheit, kurzfristige Trends und verzögerte Effekte im Buchverkauf abzubilden.

Eingabe: `data/processed/train_features_cyc.parquet`  
Ausgabe: `data/processed/train_features_cyc_lag.parquet`

Der Schritt folgt auf:
- `feature_engineering.py`
- `cyclical_encoder.py`

und bildet damit den Abschluss der Feature-Erzeugung vor dem zeitbasierten Split (`model_dataset.py`).

---

## Ziel

Der Lag-Schritt verfolgt drei Hauptziele:

1. **Vergangene Werte explizit machen**  
   Das Modell soll nicht nur den aktuellen Zustand sehen, sondern auch explizit frühere Beobachtungen (z. B. Vortag, Vorwoche).

2. **Kurzfristige Trends und Glättung**  
   Durch Rolling-Features (z. B. gleitender Mittelwert) werden lokale Mittelwerte und Trends über definierte Fenster erfasst.

3. **Strukturierte, konfigurierbare Feature-Erzeugung**  
   Welche Lags und Rolling-Fenster erzeugt werden, wird zentral über `LAG_CONF` in `src/config.py` gesteuert – nicht direkt im Script.

---

## Konfiguration (`LAG_CONF`)

Die erzeugten Features werden über ein Konfigurationsdiktat gesteuert, das in `src/config.py` definiert ist.  
Typischer Aufbau:

```python
LAG_CONF = {
    "target_col": "num_sold",       # Zielspalte, auf der die Lags gebildet werden
    "lags": [1, 7, 14],             # explizite Lag-Schritte (z. B. 1 Tag, 1 Woche, 2 Wochen)
    "roll_windows": [7, 28],        # optionale Rolling-Fenstergrößen
    "roll_stats": ["mean"],         # Kennzahlen je Fenster (z. B. mean, std)
    "prefix": "lag_",               # Präfix für erzeugte Spalten
}
```

Damit ist klar nachvollziehbar, welche Verzögerungs- und Glättungsmerkmale in einer bestimmten Experimentkonfiguration verwendet wurden.

---

## Funktionsweise

### 1. Sortierung nach Gruppen und Zeit

Zunächst wird der DataFrame nach den in `GROUP_COLS` definierten Gruppenspalten und der Zeitspalte (`TIME_COL`) sortiert:

- `GROUP_COLS` – z. B. `["country", "store", "product"]`  
- `TIME_COL` – z. B. `"date"`

Diese Sortierung stellt sicher, dass Shifts und Rolling-Fenster in der korrekten zeitlichen Reihenfolge berechnet werden.

### 2. Lag-Features

Für jedes in `LAG_CONF["lags"]` angegebene Lag \(L\) wird eine neue Spalte erzeugt:

- Spaltenname: `"{prefix}{L}"`, z. B. `lag_1`, `lag_7`, `lag_14`
- Inhalt: Wert der Zielspalte `target_col` um \(L\) Zeitschritte nach hinten verschoben

Formal für eine Gruppe \(g\) und Zeitindex \(t\):

\[
\text{lag}_L(g, t) = y(g, t - L)
\]

### 3. Rolling-Features (optional)

Falls `roll_windows` und `roll_stats` gesetzt sind, werden gleitende Kennzahlen berechnet, z. B. der gleitende Mittelwert über die letzten 7 oder 28 Zeitschritte.

- Spaltenname: `"{prefix}{window}_{stat}"`, z. B. `lag_7_mean`
- Basis: `target_col` um 1 Schritt nach hinten verschoben (lookback-only, keine Zukunftsinformation)

Beispiel (gleitender Mittelwert über 7 Tage):

\[
\text{lag\_7\_mean}(g, t) = \text{mean}\big(y(g, t-1), y(g, t-2), \dots, y(g, t-7)\big)
\]

Fehlende Werte am Anfang eines Fensters werden durch `min_periods=1` abgefedert, so dass auch am Serienanfang aussagekräftige Werte entstehen.

---

## Implementierung (Kurzüberblick)

```python
from src.config import PROCESSED_DIR, LAG_CONF, GROUP_COLS, TIME_COL

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    target = LAG_CONF["target_col"]
    lags = LAG_CONF["lags"]
    roll_windows = LAG_CONF.get("roll_windows", [])
    roll_stats = LAG_CONF.get("roll_stats", [])
    prefix = LAG_CONF.get("prefix", "lag_")

    df = df.sort_values(GROUP_COLS + [TIME_COL]).copy()

    # Lags
    for lag in lags:
        df[f"{prefix}{lag}"] = df.groupby(GROUP_COLS)[target].shift(lag)

    # Rolling
    for window in roll_windows:
        for stat in roll_stats:
            colname = f"{prefix}{window}_{stat}"
            rolled = df.groupby(GROUP_COLS)[target].transform(
                lambda x: getattr(x.shift(1).rolling(window=window, min_periods=1), stat)()
            )
            df[colname] = rolled

    return df
```

Das Skript `main()` liest `train_features_cyc.parquet`, ruft `add_lag_features()` auf und schreibt `train_features_cyc_lag.parquet` zurück.

---

## Position in der Pipeline

1. `data_alignment.py` (optional)  
2. `data_cleaning.py` (optional)  
3. `feature_engineering.py`  
4. `cyclical_encoder.py`  
5. **`lag_features.py`**  ⟵ *dieses Modul*  
6. `model_dataset.py`  
7. `dataset_tft.py`  
8. `trainer_tft.py`

Damit ist `lag_features.py` der letzte Schritt der Feature-Erzeugung vor dem Split in Train/Val/Test.

---

## Beispielaufruf

### Aus der Shell

```bash
python -m src.data.lag_features
```

### Aus Python

```python
import pandas as pd
from src.data.lag_features import add_lag_features
from src.config import PROCESSED_DIR

df = pd.read_parquet(PROCESSED_DIR / "train_features_cyc.parquet")
df_lag = add_lag_features(df)
```

---

## Hinweise & Fallstricke

- Die Qualität der Lag-Features hängt direkt von der Wahl der Fenster und Lags in `LAG_CONF` ab.  
- Zu viele Lags können die Feature-Dimension stark erhöhen – wichtig für Modellkomplexität und Trainingszeit.  
- Rolling-Features werden standardmäßig so berechnet, dass keine Zukunftsinformation einfließt (Shift um 1 Schritt).  
- Die Gruppierung nach `GROUP_COLS` ist essenziell, damit Zeitreihen je Gruppe getrennt verarbeitet werden.

---

## Zusammenfassung

| Aspekt | Beschreibung |
|--------|-------------|
| Zweck | Erzeugung von Lag- und Rolling-Features |
| Steuerung | Über `LAG_CONF` in `src/config.py` |
| Input | `train_features_cyc.parquet` |
| Output | `train_features_cyc_lag.parquet` |
| Nutzen | Explizite Vergangenheitsinformationen und lokale Trends für das Modell |
