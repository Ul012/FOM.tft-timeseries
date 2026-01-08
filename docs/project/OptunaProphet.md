# Optuna-Integration für Prophet – Technische Dokumentation

**Datum:** 2026-01-08 (erstellt)  
**Status:** Production-Ready  
**Ziel & Inhalt:** Technische Dokumentation der Optuna-Integration für automatisierte Hyperparameter-Optimierung von Facebook Prophet. Beschreibt Implementierung, Workflow, Outputs und Visualisierungsinterpretation.

---

## Überblick

Die Optuna-Integration ermöglicht automatisierte Hyperparameter-Optimierung für Prophet mittels Bayesian Optimization (TPE Sampler) und intelligentem Pruning (MedianPruner). Alle Trials werden persistent in SQLite gespeichert und können nachträglich analysiert werden. Die Integration ist dataset-spezifisch organisiert (Booksales/Walmart getrennt).

---

## Technische Voraussetzungen

### Software-Abhängigkeiten
```bash
pip install prophet optuna plotly kaleido --break-system-packages
```

**Hinweis:** 
- `kaleido` ist optional und nur für PNG-Export der Visualisierungen erforderlich
- Prophet benötigt C++ Compiler (Windows: Visual Studio Build Tools)

### Projekt-Voraussetzungen
- Abgeschlossenes Preprocessing (train.parquet, val.parquet, test.parquet)
- Vorhandene prophet_spec.json im processed-Verzeichnis
- Funktionierender trainer_prophet.py
- Dataset-Config gesetzt via `$env:DATASET_CONFIG`

---

## Implementierte Komponenten

### Script-Übersicht

| Script | Modul | Funktion | CLI-Beispiel |
|--------|-------|----------|--------------|
| `optuna_prophet.py` | `src/modeling/` | Führt HPO-Study aus | `python -m src.modeling.optuna_prophet --study-name <name> --n-trials <n>` |
| `optuna_prophet_export_best.py` | `src/modeling/` | Exportiert beste Trial als YAML | `python -m src.modeling.optuna_prophet_export_best --study-name <name>` |
| `optuna_prophet_export_trial.py` | `src/modeling/` | Exportiert spezifische Trial als YAML | `python -m src.modeling.optuna_prophet_export_trial --study-name <name> --trial-number <n>` |
| `plot_prophet_optuna_study.py` | `src/visualization/` | Erstellt Visualisierungen | `python -m src.visualization.plot_prophet_optuna_study --study-name <name>` |

### Ordnerstruktur

```
results/prophet/optuna/
├── booksales/                         # Dataset-spezifisch
│   ├── prophet_studies.db             # SQLite: alle Studies und Trials
│   ├── study_<name>_<timestamp>.csv   # CSV-Export aller Trials einer Study
│   ├── study_<name>_<timestamp>.json  # JSON-Summary einer Study
│   ├── trial_0000/
│   │   └── trial_summary.json         # Hyperparameter und val_mae
│   ├── trial_0001/
│   │   └── ...
│   └── plots/
│       └── <study_name>/
│           ├── optimization_history.html/png
│           ├── param_importances.html/png
│           ├── parallel_coordinate.html/png
│           └── slice.html/png
└── walmart/                           # Dataset-spezifisch
    ├── prophet_studies.db
    └── ...

configs/models/prophet/
├── booksales/
│   ├── baseline.yaml
│   └── optuna_prophet_booksales_best.yaml   # Automatisch generiert
└── walmart/
    ├── baseline.yaml
    └── optuna_prophet_walmart_best.yaml     # Automatisch generiert
```

---

## Optimierungsziel

**Metrik:** Mean Absolute Error (MAE) auf Validation-Set  
**Richtung:** Minimierung

**Begründung:**
- Etablierte Metrik für Time Series Forecasting
- Weniger outlier-sensitiv als RMSE
- Direkt interpretierbar (gleiche Einheit wie Zielvariable)
- MAPE/SMAPE ungeeignet bei Null-/Kleinwerten (z.B. Walmart)

**Berechnung:** Durchschnitt über alle Gruppen (z.B. 48 Gruppen bei Booksales, 3031 bei Walmart)

---

## Hyperparameter-Suchräume

### Konfigurierte Ranges

Die Suchräume sind in `optuna_prophet.py` unter `SEARCH_SPACE` definiert:

```python
changepoint_prior_scale:    [0.001, 0.5]      # log-scale, float
seasonality_prior_scale:    [0.01, 10.0]      # log-scale, float
holidays_prior_scale:       [0.01, 10.0]      # log-scale, float
seasonality_mode:           ["multiplicative", "additive"]  # categorical
growth:                     ["linear"]         # categorical (logistic benötigt cap/floor)
```

**Parameter-Bedeutung:**

| Hyperparameter | Beschreibung | Niedrig → Hoch |
|----------------|--------------|----------------|
| `changepoint_prior_scale` | Trend-Flexibilität | Steifer Trend → Flexibler Trend |
| `seasonality_prior_scale` | Saisonalitäts-Stärke | Schwache Seasonality → Starke Seasonality |
| `holidays_prior_scale` | Feiertags-Effekt-Stärke | Schwacher Effekt → Starker Effekt |
| `seasonality_mode` | Saisonalitäts-Typ | additive vs. multiplicative |
| `growth` | Trend-Typ | linear (nicht-saturierend) |

---

### Fixe Parameter

Die folgenden Parameter sind für alle Trials identisch (konfiguriert in `FIXED_CONFIG`):

- `yearly_seasonality`: True
- `weekly_seasonality`: True
- `daily_seasonality`: False (nur für stündliche Daten)
- `interval_width`: 0.95 (Konfidenzintervalle)
- `mcmc_samples`: 0 (MAP-Estimation, schneller als MCMC)

---

## Pruning-Strategie

### MedianPruner-Konfiguration

```python
pruner = MedianPruner(
    n_startup_trials=5,    # Erste 5 Trials vollständig durchlaufen
    n_warmup_steps=0       # Kein Warmup (Prophet hat keine Epochen)
)
```

### Funktionsweise

- Prophet hat keine Epochen → Pruning auf Trial-Level, nicht innerhalb eines Trainings
- Vergleich mit Median bisheriger Trials nach Abschluss
- Automatische Abbruchkriterien bei deutlicher Unterperformance
- Zeitersparnis durch frühzeitigen Abbruch unpromiser Parameter-Kombinationen

**Typische Ergebnisse:** ~10-15% der Trials werden gepruned

---

## Workflow

### Phase 1: Hyperparameter-Optimierung

```bash
# Booksales (schnell, ~7-10 Min für 20 Trials)
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"
python -m src.modeling.optuna_prophet --study-name prophet_booksales --n-trials 20

# Walmart (langsam, ~10-15h für 20 Trials)
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"
python -m src.modeling.optuna_prophet --study-name prophet_walmart --n-trials 20
```

**Parameter:**
- `--study-name`: Eindeutiger Name der Study (empfohlen: `prophet_<dataset>`)
- `--n-trials`: Anzahl durchzuführender Trials (empfohlen: 20-30)

**Output:**
- `results/prophet/optuna/<dataset>/prophet_studies.db` (erstellt/erweitert)
- `results/prophet/optuna/<dataset>/trial_<n>/` für jede Trial
- `results/prophet/optuna/<dataset>/study_<name>_<timestamp>.csv`
- `results/prophet/optuna/<dataset>/study_<name>_<timestamp>.json`

**Dauer:**
- **Booksales:** ~20 Sekunden pro Trial (48 Gruppen) → ~7 Minuten für 20 Trials
- **Walmart:** ~30-45 Minuten pro Trial (3031 Gruppen) → ~10-15 Stunden für 20 Trials

**Hinweis für Walmart:** cmdstanpy Error-Messages sind normal und nicht kritisch. Prophet fällt automatisch auf Newton-Optimizer zurück.

---

### Phase 2: Visualisierung und Analyse

#### Visualisierungen erstellen

```bash
# Booksales
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"
python -m src.visualization.plot_prophet_optuna_study --study-name prophet_booksales

# Walmart (nur wichtigste Plots)
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"
python -m src.visualization.plot_prophet_optuna_study --study-name prophet_walmart --plots history importance
```

**Output:** 4 Plot-Typen als HTML und PNG (falls kaleido installiert)

**Plot-Typen:**

| Plot | Interpretation |
|------|----------------|
| **Optimization History** | Verlauf val_mae über alle Trials. Niedrigster Punkt = beste Trial. Abflachende Kurve = Konvergenz. |
| **Parameter Importance** | Einfluss jedes Parameters auf val_mae (höherer Wert = wichtiger). Zeigt kritische Hyperparameter. |
| **Parallel Coordinate** | Parameter-Kombinationen der besten Trials (blaue Linien). Erkennung erfolgreicher Konfigurationen. |
| **Slice Plot** | Einfluss einzelner Parameter isoliert. Zeigt optimale Wertebereiche je Parameter. |

---

### Phase 3: Export und Finales Training

#### Beste Konfiguration exportieren

```bash
# Booksales
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"
python -m src.modeling.optuna_prophet_export_best --study-name prophet_booksales

# Walmart
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"
python -m src.modeling.optuna_prophet_export_best --study-name prophet_walmart
```

**Output:** `configs/models/prophet/<dataset>/optuna_prophet_<study_name>_best.yaml`

**Inhalt:**
- Alle Hyperparameter der besten Trial
- Metadaten: Study-Name, Trial-Nummer, val_mae

#### Spezifische Trial exportieren (optional)

```bash
python -m src.modeling.optuna_prophet_export_trial --study-name prophet_booksales --trial-number 5
```

**Anwendungsfälle:**
- Export einer fast-optimalen aber konservativeren Trial
- Vergleich verschiedener Parameter-Kombinationen
- Ensemble-Methoden

**Output:** `configs/models/prophet/<dataset>/optuna_prophet_<study_name>_trial_<n>.yaml`

#### Finales Training

```bash
# Booksales
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"
python -m src.modeling.trainer_prophet --config configs/models/prophet/booksales/optuna_prophet_booksales_best.yaml

# Walmart
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"
python -m src.modeling.trainer_prophet --config configs/models/prophet/walmart/optuna_prophet_walmart_best.yaml
```

---

## Interpretation der Visualisierungen

### Optimization History

**Achsen:**
- X-Achse: Trial-Nummer (chronologisch)
- Y-Achse: Objective Value (val_mae)

**Interpretation:**
- Niedrigster Punkt = beste Trial
- Abwärtstrend = erfolgreiche Optimierung
- Flache Kurve am Ende = Konvergenz (weitere Trials bringen wenig Verbesserung)
- Starke Schwankungen = Search Space zu groß oder mehr Trials nötig

**Typisch für Prophet:** Erste 3-5 Trials explorieren weit, dann Konvergenz zu optimalen Bereichen.

---

### Parameter Importance

**Darstellung:** Balkendiagramm, sortiert nach Einfluss

**Interpretation:**
- Höherer Wert = Parameter hat größeren Einfluss auf val_mae
- **changepoint_prior_scale** oft dominant (70-90% Importance)
- **seasonality_mode** moderat wichtig (5-15%)
- **seasonality_prior_scale, holidays_prior_scale** oft gering (<5%)

**Praktische Konsequenz:**
- Fokus auf wichtigste Parameter in Folge-HPOs
- Unwichtige Parameter können bei Baseline-Werten belassen werden

**Beispiel Booksales:**
```
changepoint_prior_scale:  ████████████████ 80%  ← Kritisch!
seasonality_mode:         ██               9%   ← Moderat
holidays_prior_scale:     █                5%   ← Gering
seasonality_prior_scale:  █                5%   ← Gering
growth:                                    <1%  ← Egal
```

**Interpretation:** Trend-Flexibilität (changepoint) ist der Haupttreiber der Forecast-Qualität bei Booksales.

---

### Parallel Coordinate Plot

**Darstellung:** Jede Linie = eine Trial, Farbe = val_mae (blau=gut, rot=schlecht)

**Interpretation:**
- Cluster von blauen Linien zeigen erfolgreiche Parameter-Kombinationen
- Bei Prophet: Oft ein dominanter Cluster (ein lokales Optimum)
- Getrennte Cluster deuten auf Trade-offs hin (z.B. additive vs. multiplicative)

**Beispiel-Muster:**
- Blaue Linien (beste Trials) bei `changepoint_prior_scale` = 0.01-0.05
- Blaue Linien bei `seasonality_mode` = additive
→ Konservative Trend + additive Seasonality funktioniert gut

**Nutzung:** Interaktive HTML-Version ermöglicht Filterung nach val_mae-Bereichen.

---

### Slice Plot

**Darstellung:** Grid aus Scatter-Plots, jeweils ein Parameter vs. val_mae

**Interpretation:**
- Vertikale Achse = val_mae (niedriger = besser)
- Punkt-Cluster am unteren Rand = optimaler Wertebereich
- Breite Streuung = Parameter hat wenig Einfluss
- Klares Optimum = Parameter ist wichtig

**Beispiel `changepoint_prior_scale`:**
- U-förmige Verteilung: Extreme Werte (sehr niedrig/hoch) schlecht
- Optimum bei 0.03-0.08 (für Booksales)
- → Sweet Spot zwischen zu steif und zu flexibel

**Beispiel `seasonality_mode`:**
- Zwei Cluster: additive vs. multiplicative
- Wenn additive deutlich tiefer → Klarer Gewinner

**Nutzung:** Identifikation optimaler Einzelwerte ohne Interaktionseffekte.

---

## Outputs und Artefakte

### Persistente Datenbank

**Datei:** `results/prophet/optuna/<dataset>/prophet_studies.db` (SQLite)

**Inhalt:**
- Alle Studies mit Namen und Konfiguration
- Alle Trials mit Parametern, Metriken, State (COMPLETE/PRUNED/FAIL)
- Timestamps und Laufzeiten

**Zugriff:**
- Via Optuna-API: `optuna.load_study(study_name=<name>, storage=<path>)`
- Via SQLite-Browser (optional)
- Via Dashboard: `optuna-dashboard sqlite:///<path>`

---

### CSV-Exports

**Automatisch:** `results/prophet/optuna/<dataset>/study_<name>_<timestamp>.csv`

**Spalten:**
- `number`: Trial-Nummer
- `value`: Finale val_mae
- `state`: COMPLETE, PRUNED, FAIL
- `params_*`: Hyperparameter-Werte
- `duration`: Laufzeit
- `datetime_start`, `datetime_complete`: Zeitstempel

**Nutzung:** Externe Analyse in Excel, Python Pandas, R

---

### Trial-Artefakte

**Pro Trial:** `results/prophet/optuna/<dataset>/trial_<n>/`

**Inhalt:**
- `trial_summary.json`: Hyperparameter, val_mae, Anzahl Gruppen

**Nutzung:** Nachvollziehen welche Hyperparameter welche Performance erreichten

**Hinweis:** Prophet-Modelle selbst werden nicht gespeichert (zu groß, regenerierbar).

---

## Dataset-Spezifische Besonderheiten

### Booksales

**Charakteristika:**
- 48 Gruppen (3 Countries × 16 Books)
- Täglich, 7-Tage-Forecast
- Keine Null-Werte
- Starke Trend-Änderungen (neue Bücher)

**Optimierung:**
- Schnell (~7 Min für 20 Trials)
- `changepoint_prior_scale` dominiert Importance
- MAPE/SMAPE sind aussagekräftig

**Empfehlung:**
- 20 Trials ausreichend
- Fokus auf changepoint_prior_scale

---

### Walmart

**Charakteristika:**
- 3031 Gruppen (45 Stores × ~67 Departments)
- Wöchentlich, 4-Wochen-Forecast
- Viele Null-Werte (Sparse Data)
- Stabile Trends, starke Saisonalität

**Optimierung:**
- Langsam (~10-15h für 20 Trials)
- cmdstanpy Errors normal (automatischer Fallback auf Newton)
- MAPE/SMAPE unbrauchbar (Null-Werte)
- Einige Gruppen haben unzureichende Daten → werden übersprungen

**Empfehlung:**
- 20-30 Trials, über Nacht laufen lassen
- Nur MAE/RMSE betrachten, MAPE/SMAPE ignorieren
- Fokus auf changepoint_prior_scale und seasonality_mode

**Normale Log-Messages:**
```
cmdstanpy - ERROR - Chain [1] error: code '1' Operation not permitted
Optimization terminated abnormally. Falling back to Newton.
```
→ **Nicht kritisch!** Prophet switched automatisch Optimizer.

---

## Troubleshooting

### "Optuna-Datenbank nicht gefunden"

**Ursache:** Optuna wurde noch nicht durchgeführt

**Lösung:**
```bash
python -m src.modeling.optuna_prophet --study-name prophet_booksales --n-trials 20
```

---

### "Study hat keine abgeschlossenen Trials"

**Ursache:** Alle Trials sind PRUNED oder FAILED

**Lösung:**
- Prüfe Logs auf Fehler
- Mehr Trials durchführen
- Search Space anpassen (zu eng/zu weit?)

---

### "KeyError: prediction_length"

**Ursache:** Dataset-Config hat `prediction_length` in `forecasting`-Section

**Lösung:** Bereits gefixt in Export-Scripts (`.get("forecasting", {}).get("prediction_length", 7)`)

---

### "cmdstanpy ERROR" Messages bei Walmart

**Ursache:** Einige Gruppen numerisch instabil (wenige Datenpunkte, Nullen)

**Ist das kritisch?** Nein! Prophet fällt automatisch auf Newton-Optimizer zurück.

**Lösung:** Keine Aktion nötig, Training läuft weiter.

---

### "Modell nicht gefunden, überspringe Gruppe"

**Ursache:** Einige Store/Dept-Kombinationen haben zu wenige Datenpunkte

**Ist das kritisch?** Nein! Durchschnitt wird über valide Gruppen berechnet.

**Beispiel:** 3031 Gruppen evaluiert, 5 übersprungen → val_mae über 3026 Gruppen

---

## Erweiterbarkeit

### Multi-Study-Management

Verschiedene Studies für unterschiedliche Experimente:

```bash
# Verschiedene Search Spaces
python -m src.modeling.optuna_prophet --study-name prophet_booksales_v1 --n-trials 20
python -m src.modeling.optuna_prophet --study-name prophet_booksales_v2 --n-trials 20

# Verschiedene Datasets
python -m src.modeling.optuna_prophet --study-name prophet_booksales --n-trials 20
python -m src.modeling.optuna_prophet --study-name prophet_walmart --n-trials 30
```

**Vorteil:** Isolierte Exploration verschiedener Hypothesen

---

### Custom Search Spaces

Anpassung in `optuna_prophet.py`:

```python
# Engerer Range basierend auf Erkenntnissen
SEARCH_SPACE = {
    "changepoint_prior_scale": {"min": 0.01, "max": 0.1, "log": True},  # ← Fokussiert
    "seasonality_mode": {"choices": ["additive"]},  # ← Fixiert auf beste
    # Rest weglassen
}
```

**Anwendungsfall:** Feintuning nach initialer breiter Suche

---

## Vergleich: Prophet vs. TFT Optuna

| Aspekt | Prophet | TFT |
|--------|---------|-----|
| **Hyperparameter** | 5 (changepoint, seasonality, ...) | 7 (learning_rate, batch_size, ...) |
| **Training Zeit/Trial** | 20 Sek - 45 Min (je nach Gruppen) | 20-60 Min (je nach Epochen) |
| **Wichtigster Parameter** | changepoint_prior_scale (oft 70-90%) | learning_rate, hidden_size |
| **Pruning** | Trial-Level (kein In-Training) | Epoch-Level (innerhalb Training) |
| **GPU erforderlich** | Nein (CPU ausreichend) | Ja (empfohlen) |
| **DB-Organisation** | Dataset-spezifisch (booksales/walmart) | Nicht dataset-spezifisch |
| **Typische Trials** | 20-30 | 30-50 |

---

## Workflow-Zusammenfassung

```
1. HPO durchführen (Dataset-Config setzen!)
   └─> optuna_prophet.py
       └─> results/prophet/optuna/<dataset>/prophet_studies.db

2. Visualisieren
   └─> plot_prophet_optuna_study.py
       └─> results/prophet/optuna/<dataset>/plots/

3. Exportieren
   ├─> optuna_prophet_export_best.py
   │   └─> configs/models/prophet/<dataset>/optuna_*_best.yaml
   └─> (optional) optuna_prophet_export_trial.py
       └─> configs/models/prophet/<dataset>/optuna_*_trial_<n>.yaml

4. Finales Training
   └─> trainer_prophet.py mit optuna_best.yaml
       └─> results/prophet/runs/

5. Evaluation
   └─> evaluate_prophet.py
       └─> results/prophet/runs/<run_id>/eval_test.json
```

---

## Best Practices

### Anzahl Trials

**Empfehlung:**
- **Explorativ:** 10-20 Trials (erste Einschätzung)
- **Produktiv:** 20-30 Trials (solide Hyperparameter)
- **Sehr wichtig:** 30-50 Trials (diminishing returns nach ~30)

**Trade-off:** Mehr Trials = bessere Chance auf Optimum, aber auch mehr Zeit

---

### Search Space Anpassung

**Initial:** Breiter Range (aktuelle Konfiguration)

**Nach erster HPO:** 
- Parameter Importance Plot anschauen
- Unwichtige Parameter fixieren
- Wichtige Parameter enger fassen

**Beispiel nach Booksales HPO:**
```python
# Wenn changepoint_prior_scale dominiert:
SEARCH_SPACE = {
    "changepoint_prior_scale": {"min": 0.02, "max": 0.08},  # ← Enger!
    "seasonality_mode": {"choices": ["additive"]},  # ← Fixiert auf beste
    # Rest bei Baseline-Werten lassen
}
```

---

### Interpretation der Ergebnisse

**Val MAE betrachten:**
- Absolute Verbesserung zu Baseline
- Relative Verbesserung (%)

**Parameter Importance nutzen:**
- Zeigt wo weitere Optimierung lohnt
- Validiert/widerlegt Hypothesen über Dataset

**Für Seminararbeit:**
- Optimization History Plot → Zeigt Konvergenz
- Parameter Importance → Zeigt kritische Hyperparameter
- Vergleich Baseline vs. Optuna-Best

---

## Ergebnis und Nutzen

Die Optuna-Integration für Prophet bietet:

- Vollautomatische Hyperparameter-Suche für Prophet-Modelle
- Dataset-spezifische Organisation (Booksales/Walmart getrennt)
- Persistente Speicherung aller Trials für spätere Analyse
- Intelligentes Pruning zur Zeitersparnis
- Umfangreiche Visualisierungs-Tools
- Export beliebiger Trial-Konfigurationen als YAML
- Reproduzierbare Ergebnisse durch SQLite-Storage
- Typische Verbesserung: 10-20% bessere val_mae vs. Baseline
- Erkenntnisse über kritische Hyperparameter (z.B. changepoint_prior_scale)

---

**Erstellt:** 2026-01-08  
**Letzte Aktualisierung:** 2026-01-08  
**Python-Version:** 3.10+  
**Abhängigkeiten:** prophet, optuna, plotly, pandas, numpy