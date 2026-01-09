# Optuna-Integration für ARIMA — Technische Dokumentation

**Datum:** 2026-01-09  
**Status:** Production-Ready  
**Ziel & Inhalt:** Technische Dokumentation der Optuna-Integration für automatisierte Hyperparameter-Optimierung von ARIMA/SARIMA. Beschreibt Implementierung, Workflow, Outputs und Visualisierungsinterpretation.

---

## Überblick

Die Optuna-Integration ermöglicht automatisierte Hyperparameter-Optimierung für ARIMA mittels Bayesian Optimization (TPE Sampler) und intelligentem Pruning (MedianPruner). Alle Trials werden persistent in SQLite gespeichert und können nachträglich analysiert werden.

**Besonderheit ARIMA:** Im Gegensatz zu TFT sind ARIMA-Trials deutlich schneller (Minuten statt Stunden), ermöglichen aber dafür umfangreichere Hyperparameter-Grids.

---

## Technische Voraussetzungen

### Software-Abhängigkeiten
```bash
pip install optuna pmdarima plotly kaleido
```

**Hinweis:** `kaleido` ist optional und nur für PNG-Export der Visualisierungen erforderlich. HTML-Plots funktionieren ohne.

### Projekt-Voraussetzungen
- Abgeschlossenes Preprocessing (train.parquet, val.parquet, test.parquet)
- ARIMA-spezifische Datenaufbereitung durchgeführt (`dataset_arima.py`)
- Vorhandene arima_spec.json im processed-Verzeichnis
- Funktionierender trainer_arima.py

---

## Implementierte Komponenten

### Script-Übersicht

| Script | Modul | Funktion | CLI-Beispiel |
|--------|-------|----------|--------------|
| `optuna_arima.py` | `src/modeling/` | Führt HPO-Study aus | `python -m src.modeling.optuna_arima --study-name <name> --n-trials <n>` |
| `optuna_arima_export_best.py` | `src/modeling/` | Exportiert beste Trial als YAML | `python -m src.modeling.optuna_arima_export_best --study-name <name>` |
| `optuna_arima_export_trial.py` | `src/modeling/` | Exportiert spezifische Trial als YAML | `python -m src.modeling.optuna_arima_export_trial --study-name <name> --trial-number <n>` |
| `plot_arima_optuna_study.py` | `src/visualization/` | Erstellt Visualisierungen | `python -m src.visualization.plot_arima_optuna_study --study-name <name>` |
| `analyze_optuna_arima_trials.py` | `src/evaluation/` | Detaillierte Statistik-Analyse | `python -m src.evaluation.analyze_optuna_arima_trials --study-name <name>` |

### Ordnerstruktur

```
results/arima/optuna/<dataset>/
├── arima_studies.db                    # SQLite: alle Studies und Trials
├── study_<name>_<timestamp>.csv        # CSV-Export aller Trials einer Study
├── study_<name>_<timestamp>.json       # JSON-Summary einer Study
├── trial_0000/
│   ├── models/
│   │   └── arima_<group>.pkl           # Trainierte Modelle pro Gruppe
│   └── trial_summary.json              # Metadaten und Hyperparameter
├── trial_0001/
│   └── ...
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

## Optimierungsziel

**Metrik:** Mean Absolute Error (MAE) auf Validation-Set  
**Richtung:** Minimierung

**Begründung:**
- Etablierte Metrik für Time Series Forecasting
- Weniger outlier-sensitiv als RMSE
- Direkt interpretierbar (gleiche Einheit wie Zielvariable)
- Geeignet für Modellvergleiche mit TFT und Prophet

---

## Hyperparameter-Suchräume

### Konfigurierte Ranges

Die Suchräume sind in `optuna_arima.py` unter `SEARCH_SPACE` definiert:

```python
# Non-seasonal ARIMA
max_p:         [0, 5]           # integer, autoregressive order
max_d:         [0, 2]           # integer, differencing order
max_q:         [0, 5]           # integer, moving average order

# Seasonal ARIMA (falls seasonal_period > 1)
max_P:         [0, 2]           # integer, seasonal AR
max_Q:         [0, 2]           # integer, seasonal MA
max_D:         [0, 1]           # integer, seasonal differencing
```

### Fixe Parameter

Die folgenden Parameter sind für alle Trials identisch (konfiguriert in `FIXED_CONFIG`):

- `seasonal_period`: m (aus arima_spec.json, z.B. 7 für Booksales, 52 für Walmart)
- `auto_arima`: True (automatische Modellselektion innerhalb der Ranges)
- `stepwise`: True (schnellere Suche)
- `suppress_warnings`: True
- `error_action`: 'ignore' (fehlerhafte Modelle überspringen)

### Datensatz-Spezifische Anpassungen

**Booksales (täglich, m=7):**
- Alle Parameter aktiv (seasonal und non-seasonal)
- Schnelles Training (~2-3 Minuten pro Trial)
- Empfohlen: 50-100 Trials

**Walmart (wöchentlich, m=52):**
- Problem: m=52 macht SARIMA extrem langsam
- Lösung 1: Nur non-seasonal ARIMA (max_P=0, max_Q=0, max_D=0)
- Lösung 2: Reduzierte seasonal ranges (max_P=1, max_Q=1)
- Empfohlen: 20-30 Trials wegen langer Laufzeit

---

## Pruning-Strategie

### MedianPruner-Konfiguration

```python
pruner = MedianPruner(
    n_startup_trials=3,    # Erste n Trials vollständig durchlaufen
    n_warmup_steps=2,      # Erste n Gruppen nicht prunen
    interval_steps=1       # Prüfung nach jeder Gruppe
)
```

### Funktionsweise

- ARIMA trainiert pro Trial alle Gruppen sequenziell
- Nach jeder Gruppe wird val_mae an Optuna gemeldet (intermediate value)
- Vergleich mit Median bisheriger Trials am gleichen Gruppen-Step
- Trial-Abbruch bei deutlicher Unterperformance
- Zeitersparnis besonders bei vielen Gruppen (z.B. Walmart: 3331 Gruppen)

**Typische Ergebnisse:** 5-15% der Trials werden gepruned, Zeitersparnis ~10-20%.

**Unterschied zu TFT:** Bei TFT wird nach Epochen gepruned, bei ARIMA nach Gruppen.

---

## Workflow

### Phase 1: Hyperparameter-Optimierung

```bash
python -m src.modeling.optuna_arima --study-name <name> --n-trials <n>
```

**Parameter:**
- `--study-name`: Eindeutiger Name der Study (default: `arima_hpo`)
- `--n-trials`: Anzahl durchzuführender Trials (empfohlen: 20-50)
- `--timeout`: Maximale Laufzeit in Sekunden (optional)

**Output:**
- `results/arima/optuna/<dataset>/arima_studies.db` (erstellt/erweitert)
- `results/arima/optuna/<dataset>/trial_<n>/` für jeden Trial
- `results/arima/optuna/<dataset>/study_<name>_<timestamp>.csv`
- `results/arima/optuna/<dataset>/study_<name>_<timestamp>.json`

**Dauer:** Stark dataset-abhängig
- **Booksales (48 Gruppen, m=7):** ~3 Min/Trial → 50 Trials ≈ 2.5 Stunden
- **Walmart (3331 Gruppen, m=52, full SARIMA):** ~2h/Trial → 20 Trials ≈ 40 Stunden
- **Walmart (3331 Gruppen, non-seasonal):** ~5 Min/Trial → 50 Trials ≈ 4 Stunden

---

### Phase 2: Visualisierung und Analyse

#### Visualisierungen erstellen

```bash
python -m src.visualization.plot_arima_optuna_study --study-name <name>
```

**Output:** 4 Plot-Typen als HTML und PNG (falls kaleido installiert)

**Plot-Typen:**

| Plot | Interpretation |
|------|----------------|
| **Optimization History** | Verlauf val_mae über alle Trials. Niedrigster Punkt = beste Trial. Abflachende Kurve = Konvergenz. |
| **Parameter Importance** | Einfluss jedes Parameters auf val_mae (höherer Wert = wichtiger). Zeigt, welche Parameter priorisiert werden sollten. |
| **Parallel Coordinate** | Parameter-Kombinationen der besten Trials (blaue Linien). Erkennung von Mustern in erfolgreichen Konfigurationen. |
| **Slice Plot** | Einfluss einzelner Parameter isoliert. Zeigt optimale Wertebereiche je Parameter. |

#### Statistik-Analyse

```bash
python -m src.evaluation.analyze_optuna_arima_trials --study-name <name> --top-n 10
```

**Output:**
- Terminal-Ausgabe: Top-n Trials, Parameter-Statistiken, Korrelationen
- `results/arima/optuna/<dataset>/analysis/<name>_all_trials.csv`
- `results/arima/optuna/<dataset>/analysis/<name>_top<n>.csv`

**Metriken:**
- Übersicht: Anzahl abgeschlossener/geprunter/fehlgeschlagener Trials
- Top-n: Beste Trials nach val_mae sortiert
- Parameter-Statistiken: Min/Max/Mean/Median je Parameter
- Korrelationen: Zusammenhang Parameter ↔ val_mae
- Pruning-Analyse: Zeitersparnis durch abgebrochene Trials

---

### Phase 3: Export und Finales Training

#### Beste Konfiguration exportieren

```bash
python -m src.modeling.optuna_arima_export_best --study-name <name>
```

**Output:** `configs/models/arima/optuna_best.yaml`

**Inhalt:**
- Alle Hyperparameter der besten Trial
- Metadaten: Study-Name, Trial-Nummer, val_mae

#### Spezifische Trial exportieren (optional)

```bash
python -m src.modeling.optuna_arima_export_trial --study-name <name> --trial-number <n>
```

**Anwendungsfälle:**
- Export einer fast-optimalen aber schnelleren Konfiguration
- Export einer Trial mit weniger Parametern (einfacheres Modell)
- Ensemble-Methoden mit mehreren Trials

**Output:** `configs/models/arima/trial_<n>.yaml`

#### Finales Training

```bash
# Dataset vorbereiten
$env:DATASET_CONFIG="configs/datasets/<dataset>.yaml"
python -m src.modeling.dataset_arima

# Training mit bester Config
python -m src.modeling.trainer_arima --config configs/models/arima/optuna_best.yaml
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
- Starke Schwankungen bis zum Ende = mehr Trials benötigt

**ARIMA-Spezifisch:** Erste Trials können sehr stark variieren, da kleine Änderungen in p,d,q große Auswirkungen haben können.

---

### Parameter Importance

**Darstellung:** Balkendiagramm, sortiert nach Einfluss

**Interpretation:**
- Höherer Wert = Parameter hat größeren Einfluss auf val_mae
- Unwichtige Parameter (kleine Balken) können in zukünftigen Studies fixiert werden
- Wichtige Parameter sollten engere Ranges in Folge-HPOs erhalten

**ARIMA-Spezifisch:** 
- Oft ist `max_p` (AR-Order) am wichtigsten
- Bei starker Saisonalität sind `max_P`, `max_Q` wichtig
- `max_d`, `max_D` meist weniger wichtig (oft optimal bei 0 oder 1)

---

### Parallel Coordinate Plot

**Darstellung:** Jede Linie = eine Trial, Farbe = val_mae (blau=gut, rot=schlecht)

**Interpretation:**
- Cluster von blauen Linien zeigen erfolgreiche Parameter-Kombinationen
- Getrennte Cluster deuten auf mehrere lokale Optima hin
- Parallele blaue Linien über einen Parameter = konsistent guter Wert

**ARIMA-Spezifisch:**
- Oft sieht man, dass niedrige p,q Werte (z.B. 1-2) am besten funktionieren
- Seasonal Parameter zeigen oft klare Muster (z.B. P=1, Q=0 oder P=0, Q=1)

**Nutzung:** Interaktive HTML-Version ermöglicht Filterung nach val_mae-Bereichen.

---

### Slice Plot

**Darstellung:** Grid aus Scatter-Plots, jeweils ein Parameter vs. val_mae

**Interpretation:**
- Vertikale Achse = val_mae (niedriger = besser)
- Punkt-Cluster am unteren Rand = optimaler Wertebereich
- Breite Streuung = Parameter hat wenig Einfluss
- Klares Optimum = Parameter ist wichtig

**ARIMA-Spezifisch:**
- Diskrete Integer-Parameter erzeugen vertikale Linien
- Oft sieht man U-förmige Kurven (zu niedrig und zu hoch sind schlecht)

**Nutzung:** Identifikation optimaler Einzelwerte ohne Interaktionseffekte.

---

## Outputs und Artefakte

### Persistente Datenbank

**Datei:** `results/arima/optuna/<dataset>/arima_studies.db` (SQLite)

**Inhalt:**
- Alle Studies mit Namen und Konfiguration
- Alle Trials mit Parametern, Metriken, State (COMPLETE/PRUNED/FAIL)
- Intermediate Values (val_mae nach jeder Gruppe)
- System-Attribute (datetime, duration)

**Zugriff:**
- Via Optuna-API: `optuna.load_study(study_name=<name>, storage=<path>)`
- Via SQLite-Browser (optional)
- Via Dashboard: `optuna-dashboard sqlite:///<path>`

---

### CSV-Exports

**Automatisch:** `results/arima/optuna/<dataset>/study_<name>_<timestamp>.csv`

**Spalten:**
- `number`: Trial-Nummer
- `value`: Finale val_mae
- `state`: COMPLETE, PRUNED, FAIL
- `params_*`: Hyperparameter-Werte
- `duration`: Laufzeit als Timedelta
- `datetime_start`, `datetime_complete`: Zeitstempel

**Nutzung:** Externe Analyse in Excel, R, Jupyter

---

### Trial-Artefakte

**Pro Trial:** `results/arima/optuna/<dataset>/trial_<n>/`

**Inhalt:**
- `models/`: Trainierte ARIMA-Modelle für alle Gruppen (.pkl files)
- `trial_summary.json`: Hyperparameter, val_mae, Anzahl Gruppen, Duration

**Nutzung:** Nachladen und Re-Evaluation einzelner Trials

**Unterschied zu TFT:** ARIMA speichert viele einzelne Modelle (pro Gruppe eins), TFT nur ein großes Modell.

---

### Analyse-Exports

**Manuelle Erstellung:** Via `analyze_optuna_arima_trials.py`

**Dateien:**
- `<name>_all_trials.csv`: Kopie aller Trials (identisch zu auto-export)
- `<name>_top<n>.csv`: Nur beste n Trials

---

## Besonderheiten ARIMA

### Multi-Group Training

**Problem:** ARIMA trainiert pro Gruppe ein separates Modell (im Gegensatz zu TFT's globalem Modell).

**Lösung:**
- Optuna meldet intermediate values nach jeder Gruppe
- Pruning kann früh abbrechen (nach z.B. 100 von 3331 Gruppen)
- Finale val_mae ist Durchschnitt über alle Gruppen

**Vorteil:** Sehr flexible Modellierung (jede Gruppe kann eigene Order haben)

**Nachteil:** Training dauert bei vielen Gruppen länger

---

### Seasonal Period Limitation

**Problem:** Hohe seasonal periods (m=52) machen auto_arima extrem langsam.

**Walmart-Beispiel:**
- Full SARIMA (max_P=2, max_Q=2): ~2h pro Trial
- Non-seasonal (max_P=0, max_Q=0): ~5 Min pro Trial

**Empfehlung für Walmart:**
```yaml
# Option 1: Nur non-seasonal (schnell)
max_P: 0
max_Q: 0
max_D: 0

# Option 2: Minimale seasonal (Kompromiss)
max_P: 1
max_Q: 1
max_D: 1
```

**Wissenschaftliche Begründung:** Extended lookback (z.B. 100 Wochen) kann langfristige Patterns implizit erfassen.

---

### Resume-Funktion

**Problem:** ARIMA-Training kann durch Systemabstürze unterbrochen werden.

**Lösung:** Separates Script `resume_arima_training.py` (nicht Teil von Optuna, aber kompatibel)

**Workflow:**
1. Optuna-Trial wird gestartet
2. System-Crash bei Gruppe 1000/3331
3. Resume-Script erkennt bereits trainierte Modelle
4. Training wird ab Gruppe 1001 fortgesetzt
5. Trial wird in Optuna als COMPLETE markiert

---

## Erweiterbarkeit

### Multi-Study-Management

Verschiedene Studies für unterschiedliche Fragestellungen:

```python
# Beispiel: Separate Studies für seasonal vs. non-seasonal
study_seasonal = optuna.create_study(study_name="arima_seasonal", ...)
study_nonseasonal = optuna.create_study(study_name="arima_nonseasonal", ...)
```

**Vorteil:** Gezielte Exploration unterschiedlicher Modellklassen.

---

### Dataset-Spezifische Studies

```python
# Booksales: Kurze Trials, viele Trials
study_books = optuna.create_study(study_name="arima_booksales", ...)
study_books.optimize(objective, n_trials=100)

# Walmart: Lange Trials, wenige Trials
study_walmart = optuna.create_study(study_name="arima_walmart", ...)
study_walmart.optimize(objective, n_trials=20)
```

---

## Workflow-Zusammenfassung

```
1. HPO durchführen
   └─> optuna_arima.py
       └─> results/arima/optuna/<dataset>/arima_studies.db

2. Analysieren
   ├─> plot_arima_optuna_study.py
   │   └─> results/arima/optuna/<dataset>/plots/
   └─> analyze_optuna_arima_trials.py
       └─> results/arima/optuna/<dataset>/analysis/

3. Exportieren
   ├─> optuna_arima_export_best.py
   │   └─> configs/models/arima/optuna_best.yaml
   └─> (optional) optuna_arima_export_trial.py
       └─> configs/models/arima/trial_<n>.yaml

4. Finales Training
   └─> trainer_arima.py mit optuna_best.yaml
       └─> results/arima/runs/
```

---

## Performance-Vergleich mit TFT

| Aspekt | ARIMA Optuna | TFT Optuna |
|--------|--------------|------------|
| **Trial-Dauer** | 3min - 2h (dataset-abhängig) | 20-60min (konstant) |
| **Anzahl Trials** | 20-100 (je nach Dataset) | 30-50 (typisch) |
| **Pruning-Effekt** | 10-20% Zeitersparnis | 15-30% Zeitersparnis |
| **Parallelisierung** | Schwierig (Multi-Group) | Einfach (GPU) |
| **Gesamt-Dauer** | 2-40h (stark variabel) | 15-30h (relativ stabil) |

---

## Ergebnis und Nutzen

Die Optuna-Integration bietet:

- Vollautomatische Hyperparameter-Suche für ARIMA/SARIMA
- Persistente Speicherung aller Trials für spätere Analyse
- Intelligentes Pruning zur Zeitersparnis (besonders bei vielen Gruppen)
- Umfangreiche Visualisierungs- und Analyse-Tools
- Export beliebiger Trial-Konfigurationen als YAML
- Reproduzierbare Ergebnisse durch SQLite-Storage
- Flexibilität für seasonal vs. non-seasonal Modelle
- Kompatibilität mit Resume-Training bei Unterbrechungen