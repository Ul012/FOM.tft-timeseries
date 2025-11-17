# DataCleaning – Visuelle Überprüfung

**Datum:** 2025-11-17  
**Script:** –  
**Ziel & Inhalt:** Sehr kompakte Darstellung der drei optionalen Visualisierungsskripte zur Kontrolle der bereinigten Daily-Daten.

---

## Überblick
Die Visualisierungsskripte ermöglichen eine schnelle qualitative Prüfung der Ergebnisse aus `data_cleaning.py`. Sie zeigen den Gesamtverlauf der bereinigten Daten, vergleichen Werte vor und nach der Bereinigung und markieren Serien mit den größten Anpassungen. Die Skripte sind optional und nicht Teil der Pipeline.

---

## 1. `data_cleaning_overview.py`
Plausibilitätscheck der bereinigten Zeitreihen (`train_cleaned.parquet`).  
Aggregiert pro Land und visualisiert den Zeitverlauf.

**Aufruf:**
```bash
python -m src.visualization.data_cleaning_overview
```

---

## 2. `data_cleaning_compare.py`
Vorher–Nachher-Vergleich für 2020 auf Basis von  
`train_aligned.parquet` und `train_cleaned.parquet`.

**Aufruf:**
```bash
python -m src.visualization.data_cleaning_compare
```

---

## 3. `data_cleaning_diff.py`
Identifiziert die Serie mit den größten Änderungen (`cleaned – aligned`) und zeigt deren Verlauf.

**Aufruf:**
```bash
python -m src.visualization.data_cleaning_diff
```

---

## Eingabedateien
- `data/interim/train_aligned.parquet`  
- `data/interim/train_cleaned.parquet`

---

## Ergebnis und Nutzen
- schneller Überblick über bereinigte Daten  
- klare Sicht auf Unterschiede vor/nach Cleaning  
- Identifikation stärker angepasster Zeitreihen  
