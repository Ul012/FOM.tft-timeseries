# Optuna – Vorbereitungsschritte für die Hyperparameter-Optimierung (TFT)
Datum: 2025-11-16  
Ziel & Inhalt:  
Diese Datei listet alle Vorbereitungen auf, die notwendig sind, um Optuna für die Hyperparameter-Optimierung des Temporal Fusion Transformer (TFT) im TFT-TimeSeries-Projekt einzusetzen. Die Schritte basieren auf Best Practices aus Forschung (Lim et al.), dem Optuna-Paper, Energy-Forecasting-Studien sowie einer empfohlenen „Lean & Powerful“-Vorgehensweise.

## 1. Technische Voraussetzungen

### 1.1 Lokale Umgebung
- Optuna installieren:
  ```
  pip install optuna
  ```
- `.venv` verwenden  
- Logging-Ordner:
  ```
  logs/optuna/
  ```
- Output-Verzeichnisse:
  ```
  results/tft/runs/
  results/tft/optuna/
  ```

### 1.2 Baseline-Setup
- `trainer_tft.py` funktioniert stabil  
- `dataset_tft.py` erzeugt valide Train-/Val-Datasets  
- `config.py` enthält überschreibbare Parameter  
- Seeds gesetzt  

## 2. Vorbereitungen

### 2.1 Study-Struktur
```python
study = optuna.create_study(
    study_name="tft_hyperparam_search",
    direction="minimize",
    sampler=TPESampler(),
    pruner=MedianPruner(),
    load_if_exists=True
)
```

### 2.2 Parameter-Suchräume
- hidden_size: 16–128  
- hidden_continuous_size: 8–64  
- dropout: 0.05–0.3  
- learning_rate: 1e-4 – 5e-3  
- attention_head_size: 1–4  
- gradient_clip_val: 0.01–0.2  
- reduce_on_plateau_patience: 2–4  
- max_encoder_length: 21–60 (optional)  

### 2.3 Ergebnisse speichern
- `study.pkl`  
- `best_params.json`  
- `best_trial_summary.json`  
- Visualisierungen (opt.)  

## 3. Empfohlenes Setup („Lean & Powerful“)

### Framework
- Optuna

### Strategie
1. Random Search (20–40 Trials)  
2. TPE Bayesian Optimization (50–100 Trials)  
3. Pruning: MedianPruner oder SuccessiveHalvingPruner  

### Speicherung
- `.pkl` für Fortsetzung  
- `.json` für Best Params  

### Enger Parameterraum
- basierend auf Literatur (Lim et al., Optuna etc.)  

## 4. Warum dieses Setup ideal ist
- State-of-the-Art für Deep Learning  
- Wiederverwendbar  
- Reproduzierbar  
- Kein Overengineering  
- Literaturgestützt  

## 5. Checkliste
- [ ] Baseline stabil  
- [ ] hidden_size ≥ 32  
- [ ] EarlyStopping korrekt  
- [ ] Config überschreibbar  
- [ ] Optuna-Ordner erstellt  
- [ ] objective(trial) vorhanden  
- [ ] Seeds gesetzt  
- [ ] Logging aktiv  
- [ ] Warmup-Läufe getestet  
