# DataCleaning – Bereinigung von Ausreißern und Lockdown-Zeiträumen

**Datum:** 2025-11-17  
**Script:** `src/data/data_cleaning.py`  
**Ziel & Inhalt:** Bereinigung eines Einzelausreißers sowie Glättung der Lockdown-Monate 2020. Beschreibt Vorgehen, Algorithmik und Nutzen für tägliche Buchverkaufszeitreihen.

---

## Überblick
Das Modul **DataCleaning** korrigiert Auffälligkeiten in den täglichen Verkaufsdaten. Zwei Bereiche werden behandelt:

- ein **Einzelausreißer am 01.01.2020**  
- die **Lockdown-Monate März–Mai 2020**, deren Werte geglättet werden  

Die Methode orientiert sich am Medium-Artikel „Forecasting Book Sales“, wurde jedoch auf **Daily-Daten** angepasst.

---

## Ziel
Ziel des Bereinigungsschritts ist es:

- fehlerhafte Tageswerte zu korrigieren  
- außergewöhnlich verzerrte Zeiträume (Lockdown) zu glätten  
- die Imputation **pro Zeitreihe** durchzuführen (`country`, `store`, `product`)  

Damit entsteht eine konsistente Grundlage für Feature-Engineering und Modellierung.

---

## Vorgehen
Im Gegensatz zum Originalartikel (48×365 Minutenintervalle) werden in diesem Projekt **365 Tagesperioden** genutzt.

```python
self._fill_with_shifted_mean(periods=365, repeats=3)
```

---

## Algorithmik

### 1. Ausreißer (01.01.2020)
```python
self.handle_single_day_outlier("2020-01-01")
self._fill_with_shifted_mean(periods=365, repeats=3)
```

### 2. Lockdown-Glättung (März–Mai 2020)
```python
self.handle_lockdown_period(2020, (3, 4, 5))
self._fill_with_shifted_mean(periods=365, repeats=3)
```

Beide Schritte arbeiten **gruppenweise**, sodass jede Kombination aus Land, Store und Produkt separat behandelt wird.

---

## Gruppenweise Imputation
```python
self.df.groupby(["country", "store", "product"])["num_sold"].shift(periods)
```

Die Verschiebung erfolgt zeitlich sortiert und innerhalb jeder Gruppe.

---

## Nutzung
```bash
python -m src.data.data_cleaning
```

**Output:**  
`data/interim/train_cleaned.parquet`

---

## Ergebnis und Nutzen
- Korrektur des Einzelausreißers  
- Glättung des Lockdown-Zeitraums  
- konsistente Daily-Zeitreihen für das weitere Feature-Engineering  
