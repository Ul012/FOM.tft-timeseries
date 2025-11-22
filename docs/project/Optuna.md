# Optuna-Integration für TFT – Technische Dokumentation

**Datum:** 2025-11-22 (aktualisiert)  
**Status:** Production-Ready  
**Ziel & Inhalt:** Technische Dokumentation der Optuna-Integration für automatisierte Hyperparameter-Optimierung des Temporal Fusion Transformer. Beschreibt Implementierung, Workflow, Outputs und Visualisierungsinterpretation.

---

## Überblick

Die Optuna-Integration ermöglicht automatisierte Hyperparameter-Optimierung für den TFT mittels Bayesian Optimization (TPE Sampler) und intelligentem Pruning (MedianPruner). Alle Trials werden persistent in SQLite gespeichert und können nachträglich analysiert werden.

---

## Technische Voraussetzungen

### Software-Abhängigkeiten
```bash
pip install optuna plotly kaleido
```

**Hinweis:** `kaleido` ist optional und nur für PNG-Export der Visualisierungen erforderlich. HTML-Plots funktionieren ohne.

### Projekt-Voraussetzungen
- Abgeschlossenes Preprocessing (train.parquet, val.parquet, test.parquet)
- Vorhandene dataset_spec.json im processed-Verzeichnis
- Funktionierender trainer_tft.py
- GPU empfohlen (CPU möglich, aber ~10x langsamer)

---

## Implementierte Komponenten

### Script-Übersicht

| Script | Modul | Funktion | CLI-Beispiel |
|--------|-------|----------|--------------|
| `optuna_tft.py` | `src/modeling/` | Führt HPO-Study aus | `python -m src.modeling.optuna_tft --study-name <name> --n-trials <n>` |
| `optuna_export_best.py` | `src/modeling/` | Exportiert beste Trial als YAML | `python -m src.modeling.optuna_export_best --study-name <name>` |
| `optuna_export_trial.py` | `src/modeling/` | Exportiert spezifische Trial als YAML | `python -m src.modeling.optuna_export_trial --study-name <name> --trial-number <n>` |
| `plot_optuna_study.py` | `src/visualization/` | Erstellt Visualisierungen | `python -m src.visualization.plot_optuna_study --study-name <name>` |
| `analyze_optuna_tft_trials.py` | `src/evaluation/` | Detaillierte Statistik-Analyse | `python -m src.evaluation.analyze_optuna_tft_trials --study-name <name>` |

### Ordnerstruktur

```
results/tft/optuna/
├── tft_studies.db                    # SQLite: alle Studies und Trials
├── study_<name>_<timestamp>.csv      # CSV-Export aller Trials einer Study
├── study_<name>_<timestamp>.json     # JSON-Summary einer Study
├── trial_0000/
│   ├── checkpoints/
│   │   └── tft-epoch=XX-val_loss=Y.ckpt
│   └── trial_summary.json            # Metadaten und Hyperparameter
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

logs/tft/
└── optuna_<trial_number>/
    └── metrics.csv                   # PyTorch Lightning Logs

configs/models/tft/
├── optuna_best.yaml                  # Automatisch generiert durch export_best
└── trial_<n>.yaml                    # Manuell generiert durch export_trial
```

---

## Optimierungsziel

**Metrik:** Mean Absolute Error (MAE) auf Validation-Set  
**Richtung:** Minimierung

**Begründung:**
- Etablierte Metrik für Time Series Forecasting
- Weniger outlier-sensitiv als RMSE
- Direkt interpretierbar (gleiche Einheit wie Zielvariable)
- Geeignet für Modellvergleiche

---

## Hyperparameter-Suchräume

### Konfigurierte Ranges

Die Suchräume sind in `optuna_tft.py` unter `SEARCH_SPACE` definiert:

```python
learning_rate:         [min, max]          # log-scale, float
batch_size:            [choices]            # categorical
hidden_size:           [choices]            # categorical
dropout:               [min, max]           # float
attention_head_size:   [min, max]           # integer
gradient_clip_val:     [min, max]           # float
hidden_continuous_size:[choices]            # categorical
```

### Fixe Parameter

Die folgenden Parameter sind für alle Trials identisch (konfiguriert in `TRAINING_CONFIG`):

- `max_epochs`: Anzahl Epochen pro Trial (typisch 20-30 für schnelle Evaluation)
- `seed`: Reproduzierbarkeit
- `num_workers`: DataLoader-Parallelisierung
- `accelerator`: Hardware-Typ (gpu/cpu)
- `devices`: Anzahl Geräte
- `early_stopping_patience`: Abbruchkriterium bei Stagnation

---

## Pruning-Strategie

### MedianPruner-Konfiguration

```python
pruner = MedianPruner(
    n_startup_trials=3,    # Erste n Trials vollständig durchlaufen
    n_warmup_steps=5,      # Erste n Epochen nicht prunen
    interval_steps=1       # Prüfung nach jeder Epoche
)
```

### Funktionsweise

- Nach jeder Validation-Epoche wird val_mae an Optuna gemeldet
- Vergleich mit Median bisheriger Trials am gleichen Epochen-Step
- Trial-Abbruch bei deutlicher Unterperformance
- Automatische Zeitersparnis durch frühzeitigen Abbruch unpromiser Trials

**Typische Ergebnisse:** 10-20% der Trials werden gepruned, Zeitersparnis ~15-30%.

---

## Workflow

### Phase 1: Hyperparameter-Optimierung

```bash
python -m src.modeling.optuna_tft --study-name <name> --n-trials <n>
```

**Parameter:**
- `--study-name`: Eindeutiger Name der Study (default: `tft_hpo`)
- `--n-trials`: Anzahl durchzuführender Trials (empfohlen: 20-50)
- `--n-jobs`: Anzahl paralleler Trials (nur bei mehreren GPUs sinnvoll)
- `--timeout`: Maximale Laufzeit in Sekunden (optional)

**Output:**
- `results/tft/optuna/tft_studies.db` (erstellt/erweitert)
- `results/tft/optuna/trial_<n>/` für jede Trial
- `results/tft/optuna/study_<name>_<timestamp>.csv`
- `results/tft/optuna/study_<name>_<timestamp>.json`

**Dauer:** Abhängig von Trials und Epochen. Beispiel: 20 Trials × 30min = ~10h.

---

### Phase 2: Visualisierung und Analyse

#### Visualisierungen erstellen

```bash
python -m src.visualization.plot_optuna_study --study-name <name>
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
python -m src.evaluation.analyze_optuna_tft_trials --study-name <name> --top-n 10
```

**Output:**
- Terminal-Ausgabe: Top-n Trials, Parameter-Statistiken, Korrelationen
- `results/tft/optuna/analysis/<name>_all_trials.csv`
- `results/tft/optuna/analysis/<name>_top<n>.csv`

**Metriken:**
- Übersicht: Anzahl abgeschlossener/geprunter/fehlgeschlagener Trials
- Top-n: Beste Trials nach val_mae sortiert
- Parameter-Statistiken: Min/Max/Mean/Median je Parameter
- Korrelationen: Zusammenhang Parameter ↔ val_mae (negative Korrelation = höherer Wert führt zu niedrigerem mae)
- Pruning-Analyse: Zeitersparnis durch abgebrochene Trials

---

### Phase 3: Export und Finales Training

#### Beste Konfiguration exportieren

```bash
python -m src.modeling.optuna_export_best --study-name <name>
```

**Output:** `configs/models/tft/optuna_best.yaml`

**Inhalt:**
- Alle Hyperparameter der besten Trial
- `max_epochs` automatisch auf höheren Wert gesetzt (z.B. 50 statt 30)
- Metadaten: Study-Name, Trial-Nummer, val_mae

#### Spezifische Trial exportieren (optional)

```bash
python -m src.modeling.optuna_export_trial --study-name <name> --trial-number <n>
```

**Anwendungsfälle:**
- Export einer fast-optimalen aber schnelleren Trial
- Export einer Trial mit kleinerem Modell (weniger Speicher)
- Ensemble-Methoden mit mehreren Trials

**Output:** `configs/models/tft/trial_<n>.yaml`

#### Finales Training

```bash
python -m src.pipeline \
    --dataset configs/datasets/<dataset>.yaml \
    --model configs/models/tft/optuna_best.yaml \
    --steps training
```

**Hinweis:** Finales Training nutzt typischerweise mehr Epochen als HPO-Trials (Wert in YAML anpassen).

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

**Hinweis:** Erste Trials haben oft höhere val_mae (Exploration-Phase).

---

### Parameter Importance

**Darstellung:** Balkendiagramm, sortiert nach Einfluss

**Interpretation:**
- Höherer Wert = Parameter hat größeren Einfluss auf val_mae
- Unwichtige Parameter (kleine Balken) können in zukünftigen Studies fixiert werden
- Wichtige Parameter sollten engere Ranges in Folge-HPOs erhalten

**Hinweis:** Basiert auf Feature-Importance-Algorithmus, nicht auf einfacher Korrelation.

---

### Parallel Coordinate Plot

**Darstellung:** Jede Linie = eine Trial, Farbe = val_mae (blau=gut, rot=schlecht)

**Interpretation:**
- Cluster von blauen Linien zeigen erfolgreiche Parameter-Kombinationen
- Getrennte Cluster deuten auf mehrere lokale Optima hin
- Parallele blaue Linien über einen Parameter = konsistent guter Wert

**Nutzung:** Interaktive HTML-Version ermöglicht Filterung nach val_mae-Bereichen.

---

### Slice Plot

**Darstellung:** Grid aus Scatter-Plots, jeweils ein Parameter vs. val_mae

**Interpretation:**
- Vertikale Achse = val_mae (niedriger = besser)
- Punkt-Cluster am unteren Rand = optimaler Wertebereich
- Breite Streuung = Parameter hat wenig Einfluss
- Klares Optimum = Parameter ist wichtig

**Nutzung:** Identifikation optimaler Einzelwerte ohne Interaktionseffekte.

---

## Outputs und Artefakte

### Persistente Datenbank

**Datei:** `results/tft/optuna/tft_studies.db` (SQLite)

**Inhalt:**
- Alle Studies mit Namen und Konfiguration
- Alle Trials mit Parametern, Metriken, State (COMPLETE/PRUNED/FAIL)
- Intermediate Values (val_mae nach jeder Epoche)
- System-Attribute (datetime, duration)

**Zugriff:**
- Via Optuna-API: `optuna.load_study(study_name=<name>, storage=<path>)`
- Via SQLite-Browser (optional)
- Via Dashboard: `optuna-dashboard sqlite:///<path>`

---

### CSV-Exports

**Automatisch:** `results/tft/optuna/study_<name>_<timestamp>.csv`

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

**Pro Trial:** `results/tft/optuna/trial_<n>/`

**Inhalt:**
- `checkpoints/`: Bestes Modell-Checkpoint (PyTorch Lightning)
- `trial_summary.json`: Hyperparameter, val_mae, val_loss, Epochenzahl

**Nutzung:** Nachladen und Re-Evaluation einzelner Trials

---

### Analyse-Exports

**Manuelle Erstellung:** Via `analyze_optuna_tft_trials.py`

**Dateien:**
- `<name>_all_trials.csv`: Kopie aller Trials (identisch zu auto-export)
- `<name>_top<n>.csv`: Nur beste n Trials

---

## Parallelisierung

### Single-GPU / CPU

```bash
python -m src.modeling.optuna_tft --study-name <name> --n-trials 20 --n-jobs 1
```

**Empfohlen für:** Standard-Setups, Single-GPU-Systeme

---

### Multi-GPU

```bash
python -m src.modeling.optuna_tft --study-name <name> --n-trials 20 --n-jobs <n_gpus>
```

**Voraussetzung:** Separate GPUs verfügbar (`CUDA_VISIBLE_DEVICES` manuell setzen)

**Hinweis:** SQLite-Locking kann bei hoher Parallelität auftreten.

---

## Erweiterbarkeit

### Multi-Study-Management

Verschiedene Studies für unterschiedliche Fragestellungen:

```python
# Beispiel: Separate Studies für verschiedene Parameter-Bereiche
study_lr_focus = optuna.create_study(study_name="tft_lr_search", ...)
study_size_focus = optuna.create_study(study_name="tft_size_search", ...)
```

**Vorteil:** Gezielte Exploration einzelner Parameter-Dimensionen.

---

### Multi-Objective Optimization

Optimierung mehrerer Metriken gleichzeitig:

```python
def objective(trial):
    # ...
    return val_mae, inference_time  # Zwei Ziele

study = optuna.create_study(
    directions=["minimize", "minimize"]  # Beide minimieren
)
```

**Anwendungsfall:** Trade-off zwischen Genauigkeit und Geschwindigkeit.

---

## Workflow-Zusammenfassung

```
1. HPO durchführen
   └─> optuna_tft.py
       └─> results/tft/optuna/tft_studies.db

2. Analysieren
   ├─> plot_optuna_study.py
   │   └─> results/tft/optuna/plots/
   └─> analyze_optuna_tft_trials.py
       └─> results/tft/optuna/analysis/

3. Exportieren
   ├─> optuna_export_best.py
   │   └─> configs/models/tft/optuna_best.yaml
   └─> (optional) optuna_export_trial.py
       └─> configs/models/tft/trial_<n>.yaml

4. Finales Training
   └─> pipeline.py mit optuna_best.yaml
       └─> results/tft/runs/
```

---

## Ergebnis und Nutzen

Die Optuna-Integration bietet:

- Vollautomatische Hyperparameter-Suche ohne manuelle Intervention
- Persistente Speicherung aller Trials für spätere Analyse
- Intelligentes Pruning zur Zeitersparnis
- Umfangreiche Visualisierungs- und Analyse-Tools
- Export beliebiger Trial-Konfigurationen als YAML
- Reproduzierbare Ergebnisse durch deterministische Seeds und SQLite-Storage
- Erweiterbarkeit für Multi-Study und Multi-Objective Szenarien