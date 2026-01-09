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

| Script | Funktion | Beispiel-Aufruf |
|--------|----------|-----------------|
| `evaluate_tft.py` | TFT Evaluation | `python -m src.evaluation.evaluate_tft --run-id run_20260106_174337_tft_baseline --split test` |
| `evaluate_arima.py` | ARIMA Evaluation | `python -m src.evaluation.evaluate_arima --run-id run_20260106_180327_arima_baseline --split val` |
| `evaluate_prophet.py` | Prophet Evaluation | `python -m src.evaluation.evaluate_prophet --run-id run_20260106_182500_prophet_baseline --split test` |
| `analyze_optuna_tft_trials.py` | TFT Optuna Analyse | `python -m src.evaluation.analyze_optuna_tft_trials --study-name tft_booksales --top-n 10` |
| `analyze_optuna_arima_trials.py` | ARIMA Optuna Analyse | `python -m src.evaluation.analyze_optuna_arima_trials --study-name arima_booksales --top-n 10` |
| `analyze_optuna_prophet_trials.py` | Prophet Optuna Analyse | `python -m src.evaluation.analyze_optuna_prophet_trials --study-name prophet_booksales --top-n 10` |
| `compare_models.py` | Modellvergleich | `python -m src.evaluation.compare_models --runs run1,run2,run3` |
| `aggregate_tft_eval.py` | TFT Aggregation | `python -m src.evaluation.aggregate_tft_eval` |

---

## 📈 Visualization

| Script | Funktion | Beispiel-Aufruf |
|--------|----------|-----------------|
| `plot_tft_optuna_study.py` | TFT Optuna Plots | `python -m src.visualization.plot_tft_optuna_study --study-name tft_booksales` |
| `plot_arima_optuna_study.py` | ARIMA Optuna Plots | `python -m src.visualization.plot_arima_optuna_study --study-name arima_booksales` |
| `plot_prophet_optuna_study.py` | Prophet Optuna Plots | `python -m src.visualization.plot_prophet_optuna_study --study-name prophet_booksales` |
| `plot_forecasts.py` | Forecast Visualisierung | `python -m src.visualization.plot_forecasts --run-id <run_id>` |
| `plot_training.py` | Training Metrics | `python -m src.visualization.plot_training --run-id <run_id>` |

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

### Optuna Workflow
Der typische Optuna-Workflow besteht aus 4 Schritten:
1. `optuna_<model>.py` - Hyperparameter-Suche durchführen
2. `analyze_optuna_<model>_trials.py` - Statistische Analyse
3. `plot_<model>_optuna_study.py` - Visualisierung
4. `optuna_<model>_export_best.py` - Beste Config exportieren

Siehe [OPTUNA.md](OPTUNA.md) für Details.

### Run-IDs
Run-IDs haben das Format: `run_YYYYMMDD_HHMMSS_<model>_<config>`

Beispiel: `run_20260106_174337_arima_baseline`