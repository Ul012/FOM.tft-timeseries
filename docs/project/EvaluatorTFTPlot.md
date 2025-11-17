# EvaluatorTFT – Plots und aggregierte Auswertung

**Datum:** 2025-11-17  
**Scripts:**  
- `src/evaluation/aggregate_tft_eval.py`  
- `src/visualization/plot_tft_eval_comparison.py`  
- `src/visualization/plot_tft_forecast_series.py`  

**Ziel & Inhalt:** Übersicht über Aggregation und Visualisierung der Evaluationsergebnisse mehrerer TFT-Runs.

---

## Überblick
Die Skripte aggregieren Evaluationsdaten, vergleichen Modelle anhand definierter Kennzahlen und visualisieren ausgewählte Zeitreihen. Diese Werkzeuge arbeiten ausschließlich lesend auf vorhandenen Artefakten.

---

## 1. Aggregation (`aggregate_tft_eval.py`)
- lädt alle `eval_summary.json`-Dateien  
- erzeugt `eval_overview.csv` und `eval_overview.json`  

---

## 2. Run-Vergleich (`plot_tft_eval_comparison.py`)
- Vergleich mehrerer Runs über eine ausgewählte Kennzahl und Split  
- erzeugt PNG-Plots in `results/tft/plots/eval/`

---

## 3. Visualisierung einer Zeitreihe (`plot_tft_forecast_series.py`)
- zeigt Historie, Forecast-Horizont und Modellvorhersage für eine Serie  
- wird typischerweise run- und split-spezifisch ausgeführt  

---

## Ergebnis und Nutzen
- tabellarische Übersicht über mehrere Runs  
- grafische Vergleiche von Leistungskennzahlen  
- qualitative Analyse einzelner Zeitreihen  