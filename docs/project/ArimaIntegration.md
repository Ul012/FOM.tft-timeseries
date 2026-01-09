# ARIMA-Integration für Time Series Forecasting – Technische Dokumentation

**Datum:** 2026-01-09  
**Status:** Production-Ready  
**Ziel & Inhalt:** Technische Dokumentation der ARIMA-Integration für automatisierte Time Series Forecasting. Beschreibt Dataset-Spec-Erstellung, Training, Evaluation, Resume-Funktion und Optuna-Integration.

---

## Überblick

Die ARIMA-Integration ermöglicht robustes Time Series Forecasting mit ARIMA/SARIMA-Modellen. Die Implementation unterstützt:

- Gruppenweises Training (ein Modell pro Gruppe)
- Automatische Parametersuche via pmdarima's `auto_arima`
- Seasonal und Non-seasonal ARIMA (dataset-spezifisch konfigurierbar)
- Externe Regressoren (exogene Variablen)
- Resume-Funktion bei Unterbrechungen
- Hyperparameter-Optimierung via Optuna

**Modell-Charakteristik:**
- Local Models: Ein ARIMA-Modell pro Gruppe
- Stationarität durch Differencing (d, D)
- Autoregression: AR(p), SAR(P)
- Moving Average: MA(q), SMA(Q)

---

## Technische Voraussetzungen

### Software-Abhängigkeiten
```bash
pip install pmdarima statsmodels --break-system-packages
```

### Projekt-Voraussetzungen
- Abgeschlossenes Preprocessing (train/val/test.parquet)
- Dataset-Config mit ARIMA-Parametern
- ARIMA-Config mit seasonal-Parameter
- Umgebungsvariable DATASET_CONFIG gesetzt

---

## Implementierte Komponenten

### Script-Übersicht

| Script | Modul | Funktion | CLI-Beispiel |
|--------|-------|----------|--------------|
| `dataset_arima.py` | `src/modeling/` | Erstellt ARIMA-Spezifikation | `python -m src.modeling.dataset_arima` |
| `trainer_arima.py` | `src/modeling/` | Trainiert ARIMA-Modelle | `python -m src.modeling.trainer_arima --config <path>` |
| `evaluate_arima.py` | `src/evaluation/` | Evaluiert auf Val/Test | `python -m src.evaluation.evaluate_arima --run-id <id> --split <val\|test>` |
| `resume_arima_training.py` | `src/modeling/` | Setzt unterbrochenes Training fort | `python -m src.modeling.resume_arima_training --run-id <id>` |
| `optuna_arima.py` | `src/modeling/` | Hyperparameter-Optimierung | `python -m src.modeling.optuna_arima --study-name <n> --n-trials 20` |

### Ordnerstruktur

```
data/processed/<dataset>/
├── train/val/test.parquet
└── arima_spec.json

results/arima/runs/<run_id>/
├── models/arima_<group>.pkl
├── forecasts/train_<group>.parquet
├── summary.json
├── eval_val.json
└── eval_test.json

results/arima/optuna/<dataset>/
├── arima_studies.db
└── trial_XXXX/
```

---

## Komponente 1: Dataset-Spezifikation (`dataset_arima.py`)

### Zweck
Erstellt ARIMA-spezifische Spezifikation aus preprocessed Daten.

### ARIMA-Anforderungen
- Zeit-sortierte Daten
- Endogene Variable (Target)
- Optional: Exogene Variablen (Regressoren)
- Gruppen-Identifikatoren

### Automatische Regressor-Erkennung

Exogene Variablen werden automatisch erkannt:
- Numerische Features (Temperature, Fuel_Price)
- Binäre Flags (IsHoliday, Lockdown)
- Kategorische One-Hot-Encodings

Ausgeschlossen:
- TIME_COL, TARGET_COL, GROUP_COLS
- time_idx (TFT-spezifisch)

### Seasonal Period Bestimmung

| Frequenz | m | Interpretation |
|----------|---|----------------|
| D (täglich) | 7 | Wochen-Saisonalität |
| W (wöchentlich) | 52 | Jahres-Saisonalität |
| M (monatlich) | 12 | Jahres-Saisonalität |

**Wichtig:** Bei m=52 ist seasonal ARIMA sehr rechenintensiv → Nutze `seasonal: false` in Config.

### Output: arima_spec.json

```json
{
  "time_col": "Date",
  "target_col": "Weekly_Sales",
  "group_cols": ["Store", "Dept"],
  "exog_vars": ["IsHoliday", "Temperature"],
  "frequency": "W",
  "seasonal_period": 52,
  "n_groups": 3050,
  "prediction_length": 4,
  "auto_arima": true
}
```

### Beispielaufruf

```bash
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"
python -m src.modeling.dataset_arima
```

---

## Komponente 2: Training (`trainer_arima.py`)

### Zweck
Trainiert ARIMA/SARIMA-Modelle für alle Gruppen.

### ARIMA-Notation

**ARIMA(p,d,q):** Non-seasonal
- p: Autoregressive Order
- d: Differencing Order
- q: Moving Average Order

**SARIMA(p,d,q)(P,D,Q,m):** Seasonal
- P, D, Q: Seasonal components
- m: Seasonal Period

**Beispiel:** ARIMA(2,1,1)(1,1,1,7)

### Config-Struktur

```yaml
type: "arima"
name: "baseline"

model:
  # Seasonal aktivieren/deaktivieren
  seasonal: true  # oder false

  # Auto-ARIMA
  auto_arima: true

  # Such-Parameter
  max_p: 3
  max_q: 3
  max_d: 2
  max_P: 2  # ignored wenn seasonal=false
  max_Q: 2  # ignored wenn seasonal=false
  max_D: 1  # ignored wenn seasonal=false

training:
  prediction_length: 7
```

### Hyperparameter-Details

**seasonal (true/false)**

true:
- Verwendet SARIMA mit (P,D,Q,m)
- Erfasst saisonale Muster explizit
- Geeignet für: Kleine Datensätze, niedrige m

false:
- Verwendet nur non-seasonal ARIMA
- Speedup: 10-15×
- Saisonale Muster über höhere p-Werte
- Geeignet für: Große Datensätze, hohe m

**auto_arima (true/false)**

true: Automatische Parametersuche  
false: Manuelle Order

**max_p, max_q:** AR und MA Orders (0-5 typisch)

**max_d:** Differencing (typisch 0-2, empfohlen: fixiere auf 1)

**max_P, max_Q:** Seasonal AR und MA (0-2 typisch)

**max_D:** Seasonal Differencing (typisch 0-1, empfohlen: fixiere auf 1)

### Run-ID-Format

```
run_YYYYMMDD_HHMMSS_arima_<config_name>
```

### Outputs

**Pro Run:**
- `models/arima_<group_id>.pkl`
- `forecasts/train_<group_id>.parquet` (optional)
- `summary.json`

**summary.json:**
```json
{
  "run_id": "run_20260109_143000_arima_baseline",
  "dataset": "booksales",
  "n_models": 48,
  "n_successful": 48,
  "n_failed": 0,
  "training_duration_seconds": 12450,
  "metrics": {
    "by_group": { ... },
    "overall": {
      "mae": 14.29,
      "rmse": 19.87
    }
  }
}
```

### Beispielaufruf

```bash
# Booksales (seasonal=true)
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"
python -m src.modeling.trainer_arima --config configs/models/arima/booksales/baseline.yaml

# Walmart (seasonal=false)
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"
python -m src.modeling.trainer_arima --config configs/models/arima/walmart/baseline.yaml
```

**Laufzeit:**
- Booksales (48 Gruppen, seasonal): 3-4 Stunden
- Walmart (3050 Gruppen, non-seasonal): 12-24 Stunden
- Walmart (3050 Gruppen, seasonal): Zu lang (nicht empfohlen)

---

## Komponente 3: Evaluation (`evaluate_arima.py`)

### Zweck
Evaluiert trainierte ARIMA-Modelle auf Val/Test-Daten.

### Workflow

```
1. Lade ARIMA-Modelle
2. Lade arima_spec.json
3. Lade Val/Test-Daten
4. Pro Gruppe:
   - Erstelle Forecast
   - Berechne Metriken
5. Aggregiere über alle Gruppen
6. Speichere Evaluation-Summary
```

### Metriken

| Metrik | Formel | Interpretation |
|--------|--------|----------------|
| MAE | `mean(\|y_true - y_pred\|)` | Durchschnittlicher Fehler |
| RMSE | `sqrt(mean((y_true - y_pred)²))` | Bestraft große Fehler |
| MAPE | `mean(\|y_true - y_pred\| / \|y_true\|) * 100` | Prozentuale Abweichung |
| SMAPE | Symmetrische prozentuale Abweichung | Robust bei Nullwerten |

**Wichtig:** MAPE/SMAPE können bei Daten mit vielen Nullwerten extrem hohe Werte annehmen → Für solche Datensätze nur MAE und RMSE verwenden.

### Output

```json
{
  "run_id": "run_20260109_143000_arima_baseline",
  "dataset": "booksales",
  "split": "val",
  "n_groups": 48,
  "metrics": {
    "overall": {
      "mae": 35.43,
      "rmse": 48.67
    }
  }
}
```

### Beispielaufruf

```bash
# Validation
python -m src.evaluation.evaluate_arima --run-id <id> --split val

# Test
python -m src.evaluation.evaluate_arima --run-id <id> --split test
```

---

## Komponente 4: Resume-Training (`resume_arima_training.py`)

### Zweck
Setzt unterbrochenes Training fort.

### Anwendungsfälle
- System-Absturz während Training
- Manuelle Unterbrechung (Ctrl+C)
- Out-of-Memory bei einzelnen Gruppen
- Zeitüberschreitung bei großen Datensätzen

### Resume-Logik

```python
# Bereits trainierte Gruppen werden übersprungen
trained_groups = [f.stem.replace("arima_", "") 
                  for f in models_dir.glob("arima_*.pkl")]
remaining_groups = [g for g in all_groups 
                   if g not in trained_groups]

# Nur remaining_groups werden trainiert
```

### Beispielaufruf

```bash
python -m src.modeling.resume_arima_training --run-id <run_id>
```

---

## Komponente 5: Hyperparameter-Optimierung (Optuna)

Siehe separate Dokumentation: **OptunaArima.md**

### Schnell-Workflow

```bash
# 1. DB löschen
Remove-Item results/arima/optuna/<dataset>/arima_studies.db -Force

# 2. HPO (20 Trials)
python -m src.modeling.optuna_arima --study-name arima_booksales --n-trials 20

# 3. Visualisieren
python -m src.visualization.plot_arima_optuna_study --study-name arima_booksales

# 4. Beste Config exportieren
python -m src.modeling.optuna_arima_export_best --study-name arima_booksales

# 5. Finales Training
python -m src.modeling.trainer_arima --config optuna_best.yaml
```

---

## Dataset-spezifische Konfigurationen

### Booksales (seasonal=true)

**Charakteristik:**
- 48 Gruppen
- Wöchentliche Frequenz (m=7 bei täglichen Daten)
- Starke Wochenmuster

**Config:**
```yaml
model:
  seasonal: true
  max_p: 3
  max_q: 3
  max_d: 2
  max_P: 2
  max_Q: 2
  max_D: 1
```

**Performance:**
- Training: 3-4 Stunden
- Optuna (20 Trials): 1-2 Stunden

---

### Walmart (seasonal=false)

**Charakteristik:**
- 3050 Gruppen
- Wöchentliche Frequenz mit m=52 (Jahres-Saisonalität)
- m=52 ist fachlich KORREKT aber zu rechenintensiv

**Config:**
```yaml
model:
  seasonal: false  # m=52 zu langsam!
  max_p: 5         # Erhöht für implizite Muster
  max_q: 3
  max_d: 2
  # max_P, max_Q werden ignoriert
```

**Performance:**
- Training: 12-24 Stunden (non-seasonal)
- Optuna (20 Trials): 12-20 Stunden (non-seasonal)
- Speedup: ~12-15× vs. seasonal

**Begründung non-seasonal:**
- m=52 bedeutet: Lookback über 52 Wochen (1 Jahr)
- Fachlich korrekt für Jahres-Saisonalität
- Aber: Matrix-Operationen explodieren exponentiell
- Lösung: Non-seasonal mit höherem max_p erfasst Muster implizit

---

## Besonderheiten ARIMA

### Local vs. Global Models

**ARIMA:** Ein Modell pro Gruppe
- Vorteil: Sehr flexibel
- Nachteil: Training dauert bei vielen Gruppen länger

**TFT:** Ein globales Modell
- Vorteil: Schneller
- Nachteil: Weniger Flexibilität

### Stationarität

ARIMA benötigt (schwach) stationäre Zeitreihen.

**Transformation:**
- d (Differencing): Entfernt Trends
- D (Seasonal Differencing): Entfernt saisonale Nicht-Stationarität
- auto_arima führt automatisch Tests durch (ADF, KPSS)

**Over-Differencing:** d>1 oder D>1 können künstliche Autokorrelationen erzeugen → Empfehlung: max_d=1, max_D=1

### Seasonal Period Impact

**Kritischer Faktor:** m hat exponentiellen Einfluss!

| m | SARIMA Komplexität | Empfehlung |
|---|-------------------|------------|
| 7 | Moderat | seasonal=true |
| 12 | Hoch | seasonal=true (mit Vorsicht) |
| 52 | Extrem hoch | seasonal=false! |

**Beispiel Walmart (m=52, 3050 Gruppen):**
- Pro Gruppe: 10-30 Min (seasonal) vs. 30-60 Sek (non-seasonal)
- Gesamt: Zu lang (seasonal) vs. 12-24h (non-seasonal)

### Resume-Kompatibilität

Bei großen Datensätzen (Walmart: 3050 Gruppen) kann Training unterbrochen werden:

```bash
# Training starten
python -m src.modeling.trainer_arima --config <config>

# Bei Unterbrechung: Resume
python -m src.modeling.resume_arima_training --run-id <id>
```

---

## Workflow-Zusammenfassung

### Standard-Workflow

```
1. Dataset-Spec → python -m src.modeling.dataset_arima
2. Training     → python -m src.modeling.trainer_arima
3. Evaluation   → python -m src.evaluation.evaluate_arima
```

### Workflow mit Optuna

```
1. Dataset-Spec → dataset_arima.py
2. HPO          → optuna_arima.py (20 Trials)
3. Visualisieren → plot_arima_optuna_study.py
4. Export Best   → optuna_arima_export_best.py
5. Training      → trainer_arima.py (mit optuna_best.yaml)
6. Evaluation    → evaluate_arima.py
```

---

## Performance-Vergleich

| Aspekt | ARIMA (Booksales) | ARIMA (Walmart) | Prophet | TFT |
|--------|-------------------|-----------------|---------|-----|
| Training (Baseline) | 3-4h | 12-24h | 30-60 Min | 2-4h |
| Optuna (20 Trials) | 1-2h | 12-20h | 30-60 Min | 10-20h |
| Modelle pro Run | 48 | 3050 | 48/3050 | 1 |
| Seasonal | true (m=7) | false (m=52) | true | true |
| Resume-Funktion | Ja | Ja | Nein | Nein |

---

## Best Practices

### Seasonal-Entscheidung

**seasonal=true wenn:**
- Wenige Gruppen (<100)
- Niedrige m (<12)
- Rechenzeit akzeptabel

**seasonal=false wenn:**
- Viele Gruppen (>1000)
- Hohe m (>12)
- Rechenzeit kritisch

### Hyperparameter-Fixierung

**Immer fixieren:**
- max_d = 1
- max_D = 1

**Optimieren:**
- max_p, max_q
- max_P, max_Q (wenn seasonal=true)

### Optuna-Nutzung

Empfehlung: Nutze Optuna für finale Modelle!
- Findet bessere Hyperparameter
- Konsistent mit Prophet/TFT (20 Trials)
- Zeitinvestition lohnt sich

### Resume bei großen Datensätzen

Bei Walmart oder ähnlich großen Datensätzen:
- Nutze resume_arima_training.py
- Plane mit Unterbrechungen
- Prüfe regelmäßig Fortschritt

---

## Zusammenfassung

Die ARIMA-Integration bietet:

- Robuste Time Series Forecasting mit ARIMA/SARIMA
- Flexible gruppenweise Modellierung
- Automatische Parametersuche via auto_arima
- Config-basierte seasonal/non-seasonal Steuerung
- Hyperparameter-Optimierung via Optuna
- Resume-Funktion für große Datensätze
- Comprehensive Evaluation
- Interpretierbare Modelle