# DataCleaning – Visuelle Überprüfung

## Zweck

Die Cleaning-Komponente (`data_cleaning.py`) korrigiert:
- einen Einzelausreißer am 01.01.2020 und
- den Lockdown-Zeitraum März–Mai 2020,

indem Werte auf *NaN* gesetzt und anschließend mithilfe historischer Jahreswerte je Zeitreihe imputiert werden.

Um die Bereinigung nachvollziehbar zu machen, gibt es **drei Visualisierungs-Skripte**, die unterschiedliche Fragen beantworten:

1. **Wie sehen die bereinigten Zeitreihen insgesamt aus?**  
2. **Wie unterscheiden sich die Verläufe vor und nach dem Cleaning?**  
3. **Wo genau hat sich eine Serie wie stark verändert?**

Diese Skripte sind **optional** und nicht Teil der produktiven Pipeline.

---

## 1. `data_cleaning_overview.py` – Überblick über bereinigte Zeitreihen

### Zweck  
Schneller Überblick, ob die bereinigten Zeitreihen plausibel aussehen und keine Ausreißer oder Lücken enthalten.

### Konzept

- lädt `train_cleaned.parquet`
- aggregiert tägliche Verkäufe pro Land
- plottet `num_sold` über die Zeit für alle Länder

### Aufruf

```bash
python -m src.visualization.data_cleaning_overview
```

---

## 2. `data_cleaning_compare.py` – Vorher/Nachher-Vergleich

### Zweck  
Vergleich der Verläufe **vor** und **nach** dem Cleaning für das Jahr 2020.

### Konzept

- lädt `train_aligned.parquet` (vorher) und `train_cleaned.parquet` (nachher)
- filtert auf 2020
- aggregiert und plottet zwei Diagramme übereinander:
  - oben: vor Cleaning
  - unten: nach Cleaning

### Aufruf

```bash
python -m src.visualization.data_cleaning_compare
```

---

## 3. `data_cleaning_diff.py` – Serie mit größter Änderung

### Zweck  
Identifiziert die Serie mit den stärksten Änderungen und zeigt die Differenz `cleaned - aligned`.

### Konzept

- merged aligned und cleaned (2020)
- berechnet Differenzen je Zeile
- summiert absolute Änderungen je Zeitreihe (`country`, `store`, `product`)
- wählt die Serie mit der größten Änderung
- plottet `diff` über die Zeit

### Aufruf

```bash
python -m src.visualization.data_cleaning_diff
```

---

## Eingabedateien

Alle Skripte nutzen die gleichen Datenquellen:

- `data/interim/train_aligned.parquet`
- `data/interim/train_cleaned.parquet`

---

## Pipeline-Einordnung

Die Visualisierung ist **nicht Teil der Pipeline**, kann aber nach Schritt 2 (`data_cleaning.py`) ausgeführt werden:

1. `data_alignment.py`  
2. `data_cleaning.py`  
3. **optionale Kontrolle:**  
   - `data_cleaning_overview.py`  
   - `data_cleaning_compare.py`  
   - `data_cleaning_diff.py`  
4. `feature_engineering.py`  
5. `cyclical_encoder.py`  
6. `lag_features.py`  
7. `model_dataset.py`  
8. `dataset_tft.py`  
9. `trainer_tft.py`

---

## Typische Einsatzszenarien

- Debugging von Null-Lücken oder NaN-Blöcken  
- Veranschaulichung für Präsentation oder Seminararbeit  
- Sanity Check vor Feature Engineering

