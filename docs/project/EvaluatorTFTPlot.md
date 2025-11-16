# Evaluator_TFT – Plots und Auswertung

**Datum:** 2025-11-16  
**Scripts:**  
- `src/evaluation/aggregate_tft_eval.py`  
- `src/visualization/plot_tft_eval_comparison.py`  
- `src/visualization/plot_tft_forecast_series.py`  

**Ziel & Inhalt:**  
Dieses Dokument beschreibt die Auswertungsschritte **nach** der numerischen Evaluation via `evaluate_tft.py`. Die Skripte fassen die Ergebnisse mehrerer Runs zusammen und visualisieren sie. Sie ändern keine Modelldateien, sondern arbeiten ausschließlich lesend auf den Artefakten in `results/tft/...`.

---

## 1. Aggregation der Evaluationsergebnisse

### Script: `aggregate_tft_eval.py`

**Rolle:**  
Erzeugt eine kompakte Übersicht aller vorhandenen TFT-Evaluationsläufe.

**Input:**

- `results/tft/eval/<run_id>/eval_summary.json`  

**Output:**

- `results/tft/eval/eval_overview.csv`  
- `results/tft/eval/eval_overview.json`

---

## 2. Run-Vergleich über eine Kennzahl

### Script: `plot_tft_eval_comparison.py`

**Rolle:**  
Vergleicht Runs anhand einer gewählten Metrik und eines Splits (Val/Test).

**Output:**

- `results/tft/plots/eval/compare_<split>_<metric>.png`

---

## 3. Einzelzeitreihe: Vorhersage vs. Istwerte

### Script: `plot_tft_forecast_series.py`

**Rolle:**  
Visualisiert für eine Beispiel-Zeitreihe:
- Ist-Historie  
- Ist im Forecast-Horizont  
- TFT-Prognose im Forecast-Horizont

**Output:**

- `results/tft/plots/eval/<run_id>_<split>_forecast_series.png`

---

## 4. Zusammenspiel mit `evaluate_tft.py`

- `evaluate_tft.py`: erzeugt `eval_summary.json` je Run  
- `aggregate_tft_eval.py`: fasst alle Runs tabellarisch zusammen  
- `plot_tft_eval_comparison.py`: modellübergreifender Vergleich  
- `plot_tft_forecast_series.py`: qualitative Analyse pro Run/Zeitreihe

Damit entsteht eine klare Kette:

**Evaluation → Aggregation → Visualisierung** 
