# DataCleaning – Bereinigung von Ausreißern und Lockdown-Zeiträumen

**Datum:** 2025-11-15  
**Script:** src/data/data_cleaning.py  
**Ziel & Inhalt:** Dokumentiert die Korrektur eines Ausreißers sowie die Glättung der Lockdown-Monate 2020. Erläutert das gruppenweise Vorgehen, die Imputation und typische Fehlerquellen bei der täglichen Buchverkaufsreihe.


## Überblick

Der Cleaning-Schritt bereinigt die durch das Alignment erzeugten Zeitreihen, indem er
- einen **Einzelausreißer** (01.01.2020) korrigiert und
- den **Lockdown-Zeitraum März–Mai 2020** glättet.

Die Bereinigung orientiert sich an der Methode aus dem Medium-Artikel “Forecasting Book Sales”, wurde jedoch für **tägliche Daten** angepasst.

---

## Zielsetzung des Cleanings

1. **Ausreißer glätten**  
2. **Lockdown-Zeitraum rekonstruieren**  
3. **Pro-Serie-Historie nutzen** (country, store, product)

---

## Vorgehen im Projekt (Daily-Daten)

Wir nutzen **365 Schritte** (1 Jahr), nicht 48×365 wie im Artikel.

```python
self._fill_with_shifted_mean(periods=365, repeats=3)
```

---

## Algorithmik

### 1) Single-Day-Outlier

```python
self.handle_single_day_outlier("2020-01-01")
self._fill_with_shifted_mean(periods=365, repeats=3)
```

### 2) Lockdown (März–Mai 2020)

```python
self.handle_lockdown_period(2020, (3,4,5))
self._fill_with_shifted_mean(periods=365, repeats=3)
```

---

## Gruppenweise Imputation (wichtig!)

```python
self.df.groupby(["country","store","product"])["num_sold"].shift(periods)
```

---

## Nutzung

```bash
cd C:\\Users\\ulrik\\Projekte\\TFT-TimeSeries
python -m src.data.data_cleaning
```

Output:

```
data/interim/train_cleaned.parquet
```

---

## Typische Fehlerquellen

- falsche period-Werte (z. B. 365*48)  
- fehlende gruppenweise Shifts  
- unsortierte Zeitreihen  

---

## Zusammenfassung

- Daily statt 30-Minuten-Daten  
- Jahresbasierte Imputation  
- Gruppenweise Zeitreihen  
- Keine Lockdown-Lücke mehr  
