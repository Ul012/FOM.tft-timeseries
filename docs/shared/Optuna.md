# Optuna Hyperparameter-Tuning

Automatisierte Hyperparameter-Optimierung für TFT, ARIMA und Prophet mittels Bayesian Optimization.

---

## 📚 Verfügbare Guides

| Modell | Detaillierte Dokumentation |
|--------|---------------------------|
| **TFT** | [OptunaTFT.md](OptunaTFT.md) |
| **ARIMA** | [OptunaARIMA.md](OptunaARIMA.md) |
| **Prophet** | [OptunaProphet.md](OptunaProphet.md) |

---

## 🚀 Standard-Workflow

### 1. Hyperparameter-Optimierung durchführen

```bash
# TFT
python -m src.modeling.optuna_tft --study-name tft_booksales --n-trials 50

# ARIMA
python -m src.modeling.optuna_arima --study-name arima_booksales --n-trials 50

# Prophet
python -m src.modeling.optuna_prophet --study-name prophet_booksales --n-trials 50
```

**Output:**
- SQLite-Datenbank mit allen Trials: `results/<model>/optuna/<dataset>/<model>_studies.db`
- Pro Trial ein Checkpoint-Ordner: `results/<model>/optuna/<dataset>/trial_<n>/`

**Dauer:** Abhängig von Modell und Dataset
- TFT: ~30min pro Trial → 50 Trials ≈ 25 Stunden
- ARIMA (Booksales): ~3min pro Trial → 50 Trials ≈ 2.5 Stunden
- ARIMA (Walmart): ~2h pro Trial → 20 Trials ≈ 40 Stunden
- Prophet: ~10min pro Trial → 50 Trials ≈ 8 Stunden

---

### 2. Analyse und Visualisierung

```bash
# Statistische Analyse
python -m src.evaluation.analyze_optuna_<model>_trials \
  --study-name <name> \
  --top-n 10

# Visualisierung erstellen
python -m src.visualization.plot_<model>_optuna_study \
  --study-name <name>
```

**Output:**
- **Analyse:** CSV-Exports mit allen/top-n Trials, Terminal-Ausgabe mit Statistiken
- **Plots:** Optimization History, Parameter Importance, Parallel Coordinate, Slice Plot

---

### 3. Beste Config exportieren

```bash
python -m src.modeling.optuna_<model>_export_best --study-name <name>
```

**Output:** `configs/models/<model>/optuna_best.yaml`

**Alternative:** Spezifischen Trial exportieren
```bash
python -m src.modeling.optuna_<model>_export_trial \
  --study-name <name> \
  --trial-number <n>
```

---

### 4. Finales Training mit bester Config

```bash
python -m src.pipeline \
  --dataset configs/datasets/<dataset>.yaml \
  --model configs/models/<model>/optuna_best.yaml \
  --steps training
```

**Hinweis:** In der exportierten Config `max_epochs` eventuell erhöhen (z.B. von 30 auf 100).

---

## 🎯 Optimierungsziel

**Metrik:** Mean Absolute Error (MAE) auf Validation-Set  
**Richtung:** Minimierung

**Begründung:**
- Etabliert für Zeitreihen-Forecasting
- Weniger outlier-sensitiv als RMSE
- Direkt interpretierbar (gleiche Einheit wie Zielvariable)

---

## 🔧 Verfügbare Scripts pro Modell

Jedes Modell hat 5 Optuna-Scripts:

| Script | Funktion |
|--------|----------|
| `optuna_<model>.py` | HPO durchführen |
| `optuna_<model>_export_best.py` | Beste Config exportieren |
| `optuna_<model>_export_trial.py` | Spezifischen Trial exportieren |
| `plot_<model>_optuna_study.py` | Visualisierungen |
| `analyze_optuna_<model>_trials.py` | Statistische Analyse |

Siehe [SCRIPTS.md](SCRIPTS.md) für vollständige Übersicht.

---

## 📊 Visualisierungen

Nach dem Tuning stehen 4 interaktive Plots zur Verfügung:

### Optimization History
**Zeigt:** Verlauf der val_mae über alle Trials  
**Interpretation:** Niedrigster Punkt = beste Trial, abflachende Kurve = Konvergenz

### Parameter Importance
**Zeigt:** Einfluss jedes Parameters auf val_mae  
**Interpretation:** Höherer Balken = wichtigerer Parameter

### Parallel Coordinate
**Zeigt:** Parameter-Kombinationen aller Trials (Farbe = val_mae)  
**Interpretation:** Blaue Linien-Cluster zeigen erfolgreiche Kombinationen

### Slice Plot
**Zeigt:** Einzelner Parameter vs. val_mae  
**Interpretation:** Punkt-Cluster am unteren Rand = optimaler Wertebereich

---

## 📁 Ordnerstruktur

```
results/<model>/optuna/<dataset>/
├── <model>_studies.db              # SQLite: alle Studies und Trials
├── study_<name>_<timestamp>.csv    # CSV-Export aller Trials
├── study_<name>_<timestamp>.json   # JSON-Summary
├── trial_0000/
│   ├── checkpoints/                # Modell-Checkpoints
│   └── trial_summary.json          # Hyperparameter und Metriken
├── plots/
│   └── <study_name>/
│       ├── optimization_history.html/png
│       ├── param_importances.html/png
│       ├── parallel_coordinate.html/png
│       └── slice.html/png
└── analysis/
    ├── <study_name>_all_trials.csv
    └── <study_name>_top<n>.csv
```

---

## 🔍 Modell-Spezifische Parameter

### TFT
- `learning_rate`, `batch_size`, `hidden_size`, `dropout`
- `attention_head_size`, `gradient_clip_val`, `hidden_continuous_size`

### ARIMA
- `max_p`, `max_q`, `max_d` (non-seasonal)
- `max_P`, `max_Q`, `max_D` (seasonal)

### Prophet
- `changepoint_prior_scale`, `seasonality_prior_scale`
- `holidays_prior_scale`, `seasonality_mode`

Details siehe jeweilige Optuna-Dokumentation.

---

## 💡 Best Practices

### Anzahl Trials
- **Exploration:** 50-100 Trials für breite Suche
- **Refinement:** 20-30 Trials für engere Ranges
- **Quick Test:** 10 Trials für erste Orientierung

### Parallelisierung
- **Single GPU:** `--n-jobs 1` (Standard)
- **Multi GPU:** `--n-jobs <n_gpus>` (nur bei mehreren GPUs)
- **CPU:** Parallelisierung möglich, aber langsamer

### Pruning
- Automatisches Abbrechen unpromising Trials
- Typische Zeitersparnis: 15-30%
- Konfiguriert in jedem `optuna_<model>.py`

---

## 📖 Weiterführende Dokumentation

- **[OptunaTFT.md](OptunaTFT.md)** - Vollständige technische Dokumentation für TFT
- **[OptunaARIMA.md](OptunaARIMA.md)** - Vollständige technische Dokumentation für ARIMA
- **[OptunaProphet.md](OptunaProphet.md)** - Vollständige technische Dokumentation für Prophet
- **[SCRIPTS.md](SCRIPTS.md)** - Übersicht aller verfügbaren Scripts