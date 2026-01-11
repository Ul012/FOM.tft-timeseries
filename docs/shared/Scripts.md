# Script-Übersicht

Zentrale Referenz aller verfügbaren Scripts im Projekt.

---

## 🎯 Modeling

### TFT (Temporal Fusion Transformer)

| Script | Funktion | Beispiel-Aufruf |
|--------|----------|-----------------|
| `trainer_tft.py` | Training | `python -m src.modeling.trainer_tft --config configs/models/tft/booksales/baseline.yaml` |
| `optuna_tft.py` | Hyperparameter-Optimierung | `python -m src.modeling.optuna_tft --study-name tft_booksales --n-trials 50` |
| `optuna_tft_export_best.py` | Export beste Config | `python -m src.modeling.optuna_tft_export_best --study-name tft_booksales` |
| `optuna_tft_export_trial.py` | Export spezifischer Trial | `python -m src.modeling.optuna_tft_export_trial --study-name tft_booksales --trial-number 5` |

### ARIMA

| Script | Funktion | Beispiel-Aufruf |
|--------|----------|-----------------|
| `trainer_arima.py` | Training | `python -m src.modeling.trainer_arima --config configs/models/arima/walmart/baseline.yaml` |
| `dataset_arima.py` | Preprocessing | `$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.modeling.dataset_arima` |
| `resume_arima_training.py` | Training fortsetzen | `python -m src.modeling.resume_arima_training --run-dir results/arima/runs/run_20260106_180327_arima_baseline` |
| `optuna_arima.py` | Hyperparameter-Optimierung | `python -m src.modeling.optuna_arima --study-name arima_booksales --n-trials 50` |
| `optuna_arima_export_best.py` | Export beste Config | `python -m src.modeling.optuna_arima_export_best --study-name arima_booksales` |
| `optuna_arima_export_trial.py` | Export spezifischer Trial | `python -m src.modeling.optuna_arima_export_trial --study-name arima_booksales --trial-number 3` |

### Prophet

| Script | Funktion | Beispiel-Aufruf |
|--------|----------|-----------------|
| `trainer_prophet.py` | Training | `python -m src.modeling.trainer_prophet --config configs/models/prophet/booksales/baseline.yaml` |
| `dataset_prophet.py` | Preprocessing | `$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.modeling.dataset_prophet` |
| `optuna_prophet.py` | Hyperparameter-Optimierung | `python -m src.modeling.optuna_prophet --study-name prophet_booksales --n-trials 50` |
| `optuna_prophet_export_best.py` | Export beste Config | `python -m src.modeling.optuna_prophet_export_best --study-name prophet_booksales` |
| `optuna_prophet_export_trial.py` | Export spezifischer Trial | `python -m src.modeling.optuna_prophet_export_trial --study-name prophet_booksales --trial-number 7` |

---

## 📊 Evaluation

### Einzelmodell-Evaluation

| Script | Funktion | Beispiel-Aufruf |
|--------|----------|-----------------|
| `evaluate_tft.py` | TFT Evaluation | `python -m src.evaluation.evaluate_tft --run-id <run_id> --split test` |
| `evaluate_arima.py` | ARIMA Evaluation | `python -m src.evaluation.evaluate_arima --run-id <run_id> --split val` |
| `evaluate_prophet.py` | Prophet Evaluation | `python -m src.evaluation.evaluate_prophet --run-id <run_id> --split test` |

### Optuna-Analyse

| Script | Funktion | Beispiel-Aufruf |
|--------|----------|-----------------|
| `analyze_optuna_tft_trials.py` | TFT Optuna Analyse | `python -m src.evaluation.analyze_optuna_tft_trials --study-name tft_booksales --top-n 10` |
| `analyze_optuna_arima_trials.py` | ARIMA Optuna Analyse | `python -m src.evaluation.analyze_optuna_arima_trials --study-name arima_booksales --top-n 10` |
| `analyze_optuna_prophet_trials.py` | Prophet Optuna Analyse | `python -m src.evaluation.analyze_optuna_prophet_trials --study-name prophet_booksales --top-n 10` |

### Aggregation

| Script | Funktion | Beispiel-Aufruf |
|--------|----------|-----------------|
| `aggregate_tft_eval.py` | TFT Aggregation | `python -m src.evaluation.aggregate_tft_eval` |
| `aggregate_prophet_eval.py` | Prophet Aggregation | `python -m src.evaluation.aggregate_prophet_eval` |
| `aggregate_arima_eval.py` | ARIMA Aggregation | `python -m src.evaluation.aggregate_arima_eval` |
| `aggregate_all_models_eval.py` | Cross-Model Aggregation | `python -m src.evaluation.aggregate_all_models_eval` |

---

## 📈 Visualization

### Cross-Model Plots (Übergreifend)

| Script | Funktion | Output | Beispiel-Aufruf |
|--------|----------|--------|-----------------|
| `plot_cross_model_comparison.py` | Best Run Vergleich | `results/plots/` | `python -m src.visualization.plot_cross_model_comparison` |
| `plot_multi_model_forecast.py` | Multi-Model Forecasts | `results/plots/` | `python -m src.visualization.plot_multi_model_forecast` |
| `plot_baseline_vs_optuna.py` | Optuna-Verbesserung | `results/plots/` | `python -m src.visualization.plot_baseline_vs_optuna` |

### Modell-spezifische Plots

| Script | Funktion | Output | Beispiel-Aufruf |
|--------|----------|--------|-----------------|
| `plot_tft_optuna_study.py` | TFT Optuna Plots | `results/tft/plots/` | `python -m src.visualization.plot_tft_optuna_study --study-name tft_booksales` |
| `plot_tft_forecast_series.py` | TFT Forecast Visualisierung | `results/tft/plots/` | `python -m src.visualization.plot_tft_forecast_series --run-id <run_id>` |
| `plot_tft_eval_comparison.py` | TFT Run-Vergleich | `results/tft/plots/` | `python -m src.visualization.plot_tft_eval_comparison` |
| `plot_arima_optuna_study.py` | ARIMA Optuna Plots | `results/arima/plots/` | `python -m src.visualization.plot_arima_optuna_study --study-name arima_booksales` |
| `plot_prophet_optuna_study.py` | Prophet Optuna Plots | `results/prophet/plots/` | `python -m src.visualization.plot_prophet_optuna_study --study-name prophet_booksales` |

### Data Exploration Plots

| Script | Funktion | Output | Beispiel-Aufruf |
|--------|----------|--------|-----------------|
| `plot_data_alignment.py` | Daten-Harmonisierung | `results/plots/` | `python -m src.visualization.plot_data_alignment` |
| `plot_data_cleaning.py` | Daten-Bereinigung | `results/plots/` | `python -m src.visualization.plot_data_cleaning` |

---

## 🔧 Data Processing

| Script | Funktion | Beispiel-Aufruf |
|--------|----------|-----------------|
| `load_raw.py` | Rohdaten laden | `$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.load_raw` |
| `alignment.py` | Daten normalisieren | `$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.alignment` |
| `cleaning.py` | Daten bereinigen | `$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.cleaning` |
| `feature_engineering.py` | Features erstellen | `$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.feature_engineering` |
| `cyclical_encoder.py` | Zyklische Encoding | `$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.cyclical_encoder` |
| `lag_features.py` | Lag Features | `$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.lag_features` |

---

## 🚀 Pipeline & Orchestration

| Script | Funktion | Beispiel-Aufruf |
|--------|----------|-----------------|
| `pipeline.py` | Komplette Pipeline | `python -m src.pipeline --dataset configs/datasets/booksales.yaml --model configs/models/tft/booksales/baseline.yaml` |

---

## 📝 Hinweise

### Dataset Config erforderlich
Viele Scripts benötigen die Umgebungsvariable `DATASET_CONFIG`:
```powershell
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"
```

### Visualization-Ordnerstruktur
**Übergreifende Plots** (Cross-Model Vergleiche):
- Output: `results/plots/`
- Zeigen TFT vs Prophet vs ARIMA

**Modell-spezifische Plots**:
- TFT: `results/tft/plots/`
- Prophet: `results/prophet/plots/`
- ARIMA: `results/arima/plots/`

### Optuna Workflow
Der typische Optuna-Workflow besteht aus 4 Schritten:
1. `optuna_<model>.py` - Hyperparameter-Suche durchführen
2. `analyze_optuna_<model>_trials.py` - Statistische Analyse
3. `plot_<model>_optuna_study.py` - Visualisierung
4. `optuna_<model>_export_best.py` - Beste Config exportieren

### Evaluation & Aggregation Workflow
1. **Einzelmodell evaluieren:**
   ```bash
   python -m src.evaluation.evaluate_tft --run-id <run_id> --split test
   python -m src.evaluation.evaluate_prophet --run-id <run_id> --split test
   python -m src.evaluation.evaluate_arima --run-id <run_id> --split test
   ```

2. **Pro Modell aggregieren:**
   ```bash
   python -m src.evaluation.aggregate_tft_eval
   python -m src.evaluation.aggregate_prophet_eval
   python -m src.evaluation.aggregate_arima_eval
   ```

3. **Alle Modelle kombinieren:**
   ```bash
   python -m src.evaluation.aggregate_all_models_eval
   ```

4. **Visualisieren:**
   ```bash
   python -m src.visualization.plot_cross_model_comparison
   python -m src.visualization.plot_multi_model_forecast
   ```

### Run-IDs
Run-IDs haben das Format: `run_YYYYMMDD_HHMMSS_<model>_<config>`

Beispiel: `run_20260106_174337_arima_baseline`