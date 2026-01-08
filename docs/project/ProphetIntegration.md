# Prophet-Integration für Time Series Forecasting – Technische Dokumentation

**Datum:** 2026-01-08 (erstellt)  
**Status:** Production-Ready  
**Ziel & Inhalt:** Technische Dokumentation der Prophet-Integration für automatisierte Time Series Forecasting. Beschreibt Dataset-Spec-Erstellung, Training, Evaluation, Outputs und Best Practices.

---

## Überblick

Die Prophet-Integration ermöglicht schnelles, robustes Time Series Forecasting mit Facebook Prophet. Prophet ist besonders geeignet für Daten mit starker Saisonalität, Feiertags-Effekten und Trend-Changes. Die Implementation unterstützt:

- Gruppenweises Training (z.B. pro Country/Store/Product)
- Automatische Regressor-Erkennung
- Externe Features (Kalender, Feiertage, Flags)
- Multiplicative/Additive Seasonality
- Linear/Logistic Growth Models

---

## Technische Voraussetzungen

### Software-Abhängigkeiten
```bash
pip install prophet --break-system-packages
```

**Hinweis:** Prophet benötigt C++ Compiler (Windows: Visual Studio Build Tools)

### Projekt-Voraussetzungen
- Abgeschlossenes Preprocessing (train.parquet, val.parquet, test.parquet)
- Dataset-Config: `configs/datasets/<dataset>.yaml`
- Prophet-Config: `configs/models/prophet/<dataset>/baseline.yaml`
- Umgebungsvariable: `DATASET_CONFIG` gesetzt

---

## Implementierte Komponenten

### Script-Übersicht

| Script | Modul | Funktion | CLI-Beispiel |
|--------|-------|----------|--------------|
| `dataset_prophet.py` | `src/modeling/` | Erstellt Prophet-Spezifikation | `python -m src.modeling.dataset_prophet` |
| `trainer_prophet.py` | `src/modeling/` | Trainiert Prophet-Modelle | `python -m src.modeling.trainer_prophet --config <path>` |
| `evaluate_prophet.py` | `src/evaluation/` | Evaluiert auf Val/Test | `python -m src.evaluation.evaluate_prophet --run-id <id> --split <val|test>` |

### Ordnerstruktur

```
configs/
├── datasets/
│   └── booksales.yaml                 # Dataset-Config mit Prophet-Parametern
└── models/
    └── prophet/
        └── booksales/
            └── baseline.yaml          # Prophet-Hyperparameter

data/processed/<dataset>/
├── train.parquet                      # Training-Daten
├── val.parquet                        # Validation-Daten
├── test.parquet                       # Test-Daten
└── prophet_spec.json                  # Prophet-Spezifikation (generiert)

results/prophet/runs/<run_id>/
├── models/                            # Trainierte Modelle
│   ├── prophet_DE_Book1.pkl
│   ├── prophet_DE_Book2.pkl
│   └── ... (N Gruppen)
├── forecasts/                         # Training-Forecasts (optional)
│   └── train_DE_Book1.parquet
├── summary.json                       # Training-Summary
├── eval_val.json                      # Validation-Metriken
└── eval_test.json                     # Test-Metriken

logs/                                  # Keine spezifischen Logs (Prophet verwendet print)
```

---

## Komponente 1: Dataset-Spezifikation (`dataset_prophet.py`)

### Zweck
Konvertiert TFT-formatierte Daten zu Prophet-Format und erstellt eine Spezifikation für das Training.

### Prophet-Format-Anforderungen
Prophet benötigt:
- `ds` Spalte (Datetime)
- `y` Spalte (Target)
- Optional: Regressoren (zusätzliche Features)
- Optional: cap/floor (für logistic growth)

### Automatische Regressor-Erkennung

Das Script identifiziert automatisch Regressoren aus dem DataFrame:

```python
# Ausgeschlossen werden:
- TIME_COL, TARGET_COL, time_idx (TFT-spezifisch)
- ID_COLS (Gruppen-Identifikatoren)
- cyc_* (zyklische Encodings - Prophet hat eigene Seasonality)
- lag_* Features (Prophet erstellt eigene Lags)

# Automatisch erkannt werden:
- Kalenderfeatures: year, month, day, dayofweek, weekofyear, is_weekend
- Feiertags-Features: is_holiday_*
- Flaggen: is_* (außer is_weekend)
- Externe Features: Alle verbleibenden numerischen Spalten
```

### Konfiguration (Dataset-YAML)

```yaml
prophet:
  regressors: []                        # Leer = automatische Erkennung
  country_holidays: "DE"                # Länderkürzel für Feiertage
  growth: "linear"                      # "linear" oder "logistic"
  seasonality_mode: "multiplicative"    # "multiplicative" oder "additive"
  prediction_length: 7                  # Forecast-Horizont
```

### Output: prophet_spec.json

```json
{
  "time_col": "Date",
  "target_col": "num_sold",
  "group_cols": ["country", "store", "book"],
  "regressors": [
    "year", "month", "day", "dayofweek",
    "is_holiday_de", "is_lockdown_period"
  ],
  "country_holidays": "DE",
  "growth": "linear",
  "seasonality_mode": "multiplicative",
  "n_groups": 27,
  "prediction_length": 7
}
```

### Validierung

Die Spezifikation wird gegen den DataFrame validiert:
- Alle Regressoren müssen vorhanden sein
- time_col und target_col müssen existieren
- group_cols müssen im DataFrame sein

### Beispielaufruf

```bash
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"
python -m src.modeling.dataset_prophet
```

**Output:** `data/processed/booksales/prophet_spec.json`

---

## Komponente 2: Training (`trainer_prophet.py`)

### Zweck
Trainiert Prophet-Modelle für jede Gruppe (z.B. pro Country/Store/Product) und speichert Modelle sowie Forecasts.

### Workflow

```
1. Lade prophet_spec.json und Train-Daten
2. Iteriere über Gruppen (falls vorhanden)
3. Pro Gruppe:
   - Konvertiere zu Prophet-Format (ds/y)
   - Füge Regressoren hinzu (automatisch float64)
   - Trainiere Prophet-Modell
   - Erstelle Forecast
   - Speichere Modell (.pkl)
4. Aggregiere Metriken
5. Speichere Summary
```

### Konfiguration (Model-YAML)

```yaml
type: "prophet"
name: "baseline"

model:
  # Growth Model
  growth: "linear"                      # "linear" oder "logistic"
  
  # Seasonality
  seasonality_mode: "multiplicative"    # "additive" oder "multiplicative"
  yearly_seasonality: true              # Jährliche Saisonalität
  weekly_seasonality: true              # Wöchentliche Saisonalität
  daily_seasonality: false              # Tägliche Saisonalität (nur für stündliche Daten)
  
  # Tuning-Parameter
  changepoint_prior_scale: 0.05         # Flexibilität bei Trend-Änderungen (0.001-0.5)
  seasonality_prior_scale: 10.0         # Stärke der Saisonalität (0.01-10)
  holidays_prior_scale: 10.0            # Stärke von Feiertags-Effekten (0.01-10)
  
  # Uncertainty Intervals
  interval_width: 0.95                  # Breite der Konfidenzintervalle (0.95 = 95%)
  
  # MCMC Sampling (optional)
  mcmc_samples: 0                       # 0 = MAP estimation (schnell), >0 = MCMC (genau aber langsam)

training:
  prediction_length: 7                  # Anzahl Schritte für Forecast
```

### Hyperparameter-Details

#### Growth Model
- **linear**: Standard, für nicht-saturierende Zeitreihen
- **logistic**: Für Zeitreihen mit Sättigungsgrenze (benötigt 'cap' Spalte)

#### Seasonality Mode
- **additive**: Saisonale Effekte sind konstant über Zeit
- **multiplicative**: Saisonale Effekte skalieren mit Trend (häufiger für Verkaufsdaten)

#### changepoint_prior_scale (default: 0.05)
- **Höher (0.1, 0.5)**: Modell passt sich schneller an Trend-Änderungen an
- **Niedriger (0.01, 0.001)**: Konservativerer Trend

#### seasonality_prior_scale (default: 10.0)
- **Höher (15, 20)**: Stärkere Saisonalität
- **Niedriger (5, 1)**: Schwächere Saisonalität

#### holidays_prior_scale (default: 10.0)
- **Höher (15, 20)**: Stärkere Feiertags-Effekte
- **Niedriger (5, 1)**: Schwächere Feiertags-Effekte

### Training-Details

**Pro Gruppe:**
1. **Datenkonvertierung**: Alle Features → float64 (Prophet-Anforderung)
2. **NaN-Handling**: Fehlende Werte werden mit 0 oder Median gefüllt
3. **Regressor-Hinzufügen**: `model.add_regressor(name)` für jeden Regressor
4. **Country Holidays**: Automatisch aus `country_holidays` geladen
5. **Training**: `model.fit(prophet_df)`
6. **Forecast**: In-Sample Predictions für Metrik-Berechnung

**Metriken (Training-Set):**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)

### Run-ID-Format

```
run_YYYYMMDD_HHMMSS_prophet_<config_name>
```

Beispiel: `run_20260108_143000_prophet_baseline`

### Outputs

**Pro Run:**
- `models/prophet_<group_id>.pkl` - Trainiertes Prophet-Modell (Pickle)
- `forecasts/train_<group_id>.parquet` - Training-Forecasts (optional)
- `summary.json` - Aggregierte Training-Summary

**summary.json Struktur:**
```json
{
  "run_id": "run_20260108_143000_prophet_baseline",
  "timestamp": "2026-01-08T14:30:00",
  "dataset": "booksales",
  "model_name": "prophet_baseline",
  "config": { /* vollständige Model-Config */ },
  "n_models": 27,
  "training_duration_seconds": 324.5,
  "metrics": {
    "by_group": {
      "DE_Book1": {"mae": 12.34, "rmse": 16.78, "mape": 23.45}
    },
    "overall": {
      "mae": 12.56,
      "rmse": 17.23,
      "mape": 24.12
    }
  }
}
```

### Beispielaufruf

```bash
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"
python -m src.modeling.trainer_prophet --config configs/models/prophet/booksales/baseline.yaml
```

**Dauer:** ~5-10 Minuten für 27 Gruppen (erste Ausführung länger wegen Stan-Kompilierung)

---

## Komponente 3: Evaluation (`evaluate_prophet.py`)

### Zweck
Evaluiert trainierte Prophet-Modelle auf Validation/Test-Daten und berechnet Standard-Metriken.

### Workflow

```
1. Lade gespeicherte Prophet-Modelle
2. Lade prophet_spec.json
3. Lade Val/Test-Daten
4. Pro Gruppe:
   - Konvertiere zu Prophet-Format
   - Erstelle Forecast mit Modell
   - Berechne Metriken (MAE, RMSE, MAPE, SMAPE)
5. Aggregiere über alle Gruppen
6. Speichere Evaluation-Summary
```

### Metriken

| Metrik | Formel | Interpretation |
|--------|--------|----------------|
| **MAE** | `mean(|y_true - y_pred|)` | Durchschnittlicher absoluter Fehler |
| **RMSE** | `sqrt(mean((y_true - y_pred)²))` | Root Mean Squared Error, bestraft große Fehler |
| **MAPE** | `mean(|y_true - y_pred| / |y_true|) * 100` | Prozentuale Abweichung |
| **SMAPE** | `mean(|y_true - y_pred| / (|y_true| + |y_pred|) / 2) * 100` | Symmetrische prozentuale Abweichung |

**⚠️ WICHTIG - MAPE/SMAPE bei Daten mit Null-/Kleinwerten:**

MAPE und SMAPE können bei Datasets mit vielen Null- oder sehr kleinen Werten (z.B. Walmart: 1-5 Sales/Woche) **extrem hohe** und **nicht aussagekräftige** Werte annehmen:

- **Problem:** Division durch kleine Zahlen führt zu Ausreißern (MAPE bis zu 100+ Millionen %)
- **Beispiel Walmart:** 
  - Einzelne Gruppen: MAPE 3-50%
  - Overall MAPE: 159,000% (durch extreme Ausreißer bei Kleinwerten)
- **Empfehlung:** Bei Datasets mit vielen Null-/Kleinwerten **nur MAE und RMSE verwenden**
- **Booksales:** MAPE/SMAPE sind aussagekräftig (keine extremen Kleinwerte)

**Für Modellvergleich:**
- ✅ **MAE, RMSE**: Immer zuverlässig
- ⚠️ **MAPE, SMAPE**: Nur bei Datasets ohne Null-/Kleinwerte verwenden

### Output: eval_<split>.json

```json
{
  "run_id": "run_20260108_143000_prophet_baseline",
  "dataset": "booksales",
  "split": "test",
  "n_groups": 27,
  "metrics": {
    "by_group": {
      "DE_Book1": {
        "mae": 11.23,
        "rmse": 15.67,
        "mape": 22.34,
        "smape": 21.12
      }
    },
    "overall": {
      "mae": 12.45,
      "rmse": 16.89,
      "mape": 23.56,
      "smape": 22.34
    }
  }
}
```

### Beispielaufruf

```bash
# Validation
python -m src.evaluation.evaluate_prophet --run-id run_20260108_143000_prophet_baseline --split val

# Test
python -m src.evaluation.evaluate_prophet --run-id run_20260108_143000_prophet_baseline --split test
```

---

## Dataset-spezifische Anpassungen

### Booksales (Täglich, 7-Tage Forecast)

```yaml
# configs/datasets/booksales.yaml
prophet:
  country_holidays: "DE"
  seasonality_mode: "multiplicative"
  prediction_length: 7

# configs/models/prophet/booksales/baseline.yaml
model:
  growth: "linear"
  yearly_seasonality: true
  weekly_seasonality: true
  daily_seasonality: false
```

**Besonderheiten:**
- Starke Wochensaisonalität (Wochenend-Effekt)
- Feiertags-Effekte (Deutschland)
- Lockdown-Flag als Regressor

### Walmart (Wöchentlich, 4-Wochen Forecast)

```yaml
# configs/datasets/walmart.yaml
prophet:
  country_holidays: "US"
  seasonality_mode: "multiplicative"
  prediction_length: 4

# configs/models/prophet/walmart/baseline.yaml
model:
  growth: "linear"
  yearly_seasonality: true
  weekly_seasonality: false  # Wöchentliche Daten → keine weekly seasonality
  daily_seasonality: false
```

**Besonderheiten:**
- Externe Features (Temperature, Fuel_Price, CPI, Unemployment)
- MarkDown-Features als Regressoren
- Store-spezifische Unterschiede

---

## Vergleich mit anderen Modellen

### Prophet vs. TFT vs. ARIMA

| Aspekt | Prophet | TFT | ARIMA |
|--------|---------|-----|-------|
| **Training Zeit** | 5-10 Min | 2h | 2 Min |
| **Komplexität** | Mittel | Hoch | Niedrig |
| **Seasonality** | Exzellent | Gut | Begrenzt |
| **Trend-Changes** | Automatisch | Lernt | Fest |
| **Externe Features** | Ja | Ja | Nein |
| **Interpretierbarkeit** | Hoch | Mittel | Hoch |
| **Gruppenweise** | Separate Modelle | Ein Modell | Separate Modelle |

### Erwartete Performance (Booksales)

```
TFT:    MAE 9.64   (Beste Performance, 2h Training)
Prophet: MAE ~12-13 (Gute Performance, 5 Min Training)
ARIMA:  MAE 14.29  (Baseline, 2 Min Training)
```

**Trade-off:** Prophet bietet ~20% schlechtere Performance als TFT bei 24x schnellerem Training.

---

## Best Practices

### Hyperparameter-Tuning

**Start mit Baseline:**
```yaml
changepoint_prior_scale: 0.05
seasonality_prior_scale: 10.0
holidays_prior_scale: 10.0
```

**Wenn Underfitting (hohe MAE):**
```yaml
changepoint_prior_scale: 0.1    # Mehr Trend-Flexibilität
seasonality_prior_scale: 15.0   # Stärkere Saisonalität
holidays_prior_scale: 15.0      # Stärkere Feiertags-Effekte
```

**Wenn Overfitting (große Validierungs-Test-Differenz):**
```yaml
changepoint_prior_scale: 0.01   # Konservativerer Trend
seasonality_prior_scale: 5.0    # Schwächere Saisonalität
holidays_prior_scale: 5.0       # Schwächere Feiertags-Effekte
```

### Feature Engineering

**Gute Regressoren für Prophet:**
- ✅ Kalenderfeatures (year, month, day, dayofweek)
- ✅ Feiertags-Flags (is_holiday_*)
- ✅ Externe Events (is_lockdown_period)
- ✅ Externe Variablen (Temperature, Fuel_Price)

**Schlechte Regressoren:**
- ❌ Zyklische Encodings (cyc_*) - Prophet hat eigene Seasonality
- ❌ Lag-Features (lag_*) - Prophet erstellt eigene Lags

### Metrik-Auswahl

**Dataset-abhängige Metriken:**

| Dataset-Typ | Empfohlene Metriken | Ungeeignete Metriken | Begründung |
|-------------|---------------------|----------------------|------------|
| **Sparse Data** (viele Nullen/Kleinwerte, z.B. Walmart) | MAE, RMSE | MAPE, SMAPE | Division durch Kleinwerte → extreme Ausreißer |
| **Dense Data** (keine Nullen, z.B. Booksales) | MAE, RMSE, MAPE, SMAPE | - | Alle Metriken aussagekräftig |

**Praktische Regel:**
- Wenn Dataset Werte < 10 enthält → Nur MAE/RMSE verwenden
- Wenn alle Werte > 10 → Alle Metriken möglich
- Bei Unsicherheit → Prüfe Max MAPE in `by_group` Metriken (>10,000% = Problem!)

**Beispiel Walmart:**
```json
"overall": {
  "mae": 1360.44,     // ✅ Aussagekräftig
  "rmse": 2818.73,    // ✅ Aussagekräftig
  "mape": 159331.07,  // ❌ Unbrauchbar (extreme Ausreißer)
  "smape": 16.64      // ⚠️  Vorsicht (kann verzerrt sein)
}
```

- ❌ target_encoded Features - Risiko von Data Leakage

### Datenqualität

**Anforderungen:**
- Mindestens 2 Jahre Daten für gute Saisonalität
- Regelmäßige Zeitintervalle (keine Lücken)
- Keine NaN im Target
- Regressoren als float64 (automatisch konvertiert)

**Umgang mit Missing Data:**
- Prophet interpoliert automatisch fehlende Zeitpunkte
- Regressoren werden mit 0 oder Median gefüllt
- Warnung bei zu vielen NaN

---

## Troubleshooting

### Häufige Fehler

**1. "Prophet not installed"**
```bash
pip install prophet --break-system-packages
```

**2. "prophet_spec.json not found"**
```bash
python -m src.modeling.dataset_prophet
```

**3. "ValueError: could not convert string to float"**
→ Automatische Konvertierung in trainer_prophet.py (Zeile 95-100)
→ Falls Problem bleibt: Spalte aus Regressoren entfernen

**4. Training sehr langsam (>30 Min)**
→ Erste Ausführung kompiliert Stan-Model (15-20 Min normal)
→ Nachfolgende Runs schneller (~5 Min)

**5. Schlechte Performance (MAE >20)**
→ Hyperparameter anpassen (siehe Best Practices)
→ Regressoren prüfen (prophet_spec.json)
→ Datenqualität prüfen (Ausreißer, Lücken)

**6. Extreme MAPE/SMAPE-Werte (>10,000%)**
→ **Normal bei Datasets mit Null-/Kleinwerten!**
→ Problem: Division durch sehr kleine Zahlen (z.B. y_true = 1, y_pred = 100 → MAPE = 9,900%)
→ Lösung: **Ignoriere MAPE/SMAPE**, verwende nur **MAE und RMSE**
→ Beispiel: Walmart hat overall MAPE von 159,000% (Ausreißer bei Gruppen mit 1-5 Sales/Woche)
→ Für Modellvergleich: Nutze MAE/RMSE, nicht MAPE/SMAPE

### Performance-Optimierung

**Parallelisierung (TODO):**
```python
# Zukünftig: Parallel Training über Gruppen
from multiprocessing import Pool
pool.map(train_single_group, groups)
```

**Memory-Optimierung:**
- Große Gruppen separat trainieren
- Forecasts nicht speichern (`save_forecasts=False`)

---

## Erweiterbarkeit

### Multi-Dataset-Support

Aktuelle Datasets:
- ✅ Booksales (täglich, 27 Gruppen)
- ✅ Walmart (wöchentlich, 285 Gruppen)

Neue Datasets hinzufügen:
1. Prophet-Parameter in Dataset-YAML definieren
2. Prophet-Config erstellen (von Vorlage kopieren)
3. Dataset-Spec erstellen: `python -m src.modeling.dataset_prophet`
4. Training starten

### Feature-Erweiterungen

**Geplant:**
- Prophet-Forecast-Visualisierungen (`plot_prophet_forecast.py`)
- Hyperparameter-Tuning mit Optuna (`optuna_prophet.py`)
- Ensemble-Methoden (Prophet + TFT)
- Pipeline-Integration (`--model-type prophet`)

**Möglich:**
- Custom Seasonality (z.B. Quartalssaisonalität)
- Logistic Growth mit Capacity-Schätzung
- Uncertainty Quantification via MCMC

---

## Workflow-Zusammenfassung

```
1. Dataset-Spec erstellen
   └─> dataset_prophet.py
       └─> prophet_spec.json

2. Training durchführen
   └─> trainer_prophet.py
       └─> results/prophet/runs/<run_id>/
           ├── models/*.pkl
           └── summary.json

3. Evaluation
   ├─> evaluate_prophet.py (Val)
   │   └─> eval_val.json
   └─> evaluate_prophet.py (Test)
       └─> eval_test.json

4. Analyse
   └─> Ergebnisse vergleichen mit TFT/ARIMA
```

**Automatisiert:** Siehe `run_prophet_booksales.ps1` für vollständigen Workflow

---

## Ergebnis und Nutzen

Die Prophet-Integration bietet:

- **Schnelles Training:** 5-10 Min vs. 2h (TFT)
- **Robuste Saisonalität:** Automatische Erkennung von yearly/weekly Patterns
- **Feiertags-Integration:** Eingebaute Country-Specific Holidays
- **Trend-Flexibilität:** Automatische Changepoint-Detection
- **Externe Features:** Einfache Integration von Regressoren
- **Interpretierbarkeit:** Klare Komponenten (Trend, Seasonality, Holidays)
- **Produktionsreife:** Bewährtes Modell von Facebook
- **Gute Baseline:** Performance zwischen ARIMA und TFT

**Empfohlener Einsatz:**
- Als schnelle Baseline vor aufwändigem TFT-Training
- Für Datasets mit starker Saisonalität und Feiertags-Effekten
- In Produktionsumgebungen mit Geschwindigkeitsanforderungen
- Für explorative Analysen und schnelle Prototypen