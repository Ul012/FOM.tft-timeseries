# Optuna-Integration für ARIMA — Technische Dokumentation

**Datum:** 2026-01-09  
**Status:** Production-Ready  
**Ziel & Inhalt:** Technische Dokumentation der Optuna-Integration für automatisierte Hyperparameter-Optimierung von ARIMA/SARIMA. Beschreibt Implementierung, Workflow, Outputs und Visualisierung.

---

## Überblick

Die Optuna-Integration ermöglicht automatisierte Hyperparameter-Optimierung für ARIMA mittels Bayesian Optimization (TPE Sampler). Alle Trials werden persistent in SQLite gespeichert und können nachträglich analysiert werden.

**Besonderheit ARIMA:** Trials sind dataset-abhängig sehr unterschiedlich in der Laufzeit (Minuten bis Stunden). Die Integration unterstützt sowohl seasonal als auch non-seasonal ARIMA mit dataset-spezifischen Optimierungen.

---

## Technische Voraussetzungen

### Software-Abhängigkeiten
```bash
pip install optuna pmdarima plotly kaleido --break-system-packages
```

### Projekt-Voraussetzungen
- Abgeschlossenes Preprocessing (train/val/test.parquet)
- ARIMA-Spezifikation (arima_spec.json)
- Funktionierender trainer_arima.py mit seasonal-Config-Support
- Dataset-Configs mit `seasonal` Parameter

---

## Implementierte Komponenten

### Script-Übersicht

| Script | Modul | Funktion | CLI-Beispiel |
|--------|-------|----------|--------------|
| `optuna_arima.py` | `src/modeling/` | Führt HPO-Study aus | `python -m src.modeling.optuna_arima --study-name <n> --n-trials 20` |
| `optuna_arima_export_best.py` | `src/modeling/` | Exportiert beste Trial als YAML | `python -m src.modeling.optuna_arima_export_best --study-name <n>` |
| `optuna_arima_export_trial.py` | `src/modeling/` | Exportiert spezifische Trial als YAML | `python -m src.modeling.optuna_arima_export_trial --study-name <n> --trial-number <n>` |
| `plot_arima_optuna_study.py` | `src/visualization/` | Erstellt Visualisierungen | `python -m src.visualization.plot_arima_optuna_study --study-name <n>` |
| `analyze_optuna_arima_trials.py` | `src/evaluation/` | Detaillierte Statistik-Analyse | `python -m src.evaluation.analyze_optuna_arima_trials --study-name <n>` |

### Ordnerstruktur

```
results/arima/optuna/<dataset>/
├── arima_studies.db                    # SQLite: alle Studies und Trials
├── study_<n>_<timestamp>.csv           # CSV-Export aller Trials
├── study_<n>_<timestamp>.json          # JSON-Summary
├── trial_0000/ ... trial_XXXX/
│   ├── models/arima_<group>.pkl        # Trainierte Modelle pro Gruppe
│   └── trial_summary.json              # Metadaten und Hyperparameter
└── plots/<study_name>/
    ├── optimization_history.html/png
    ├── param_importances.html/png
    ├── parallel_coordinate.html/png
    └── slice.html/png
```

---

## Optimierungsziel

**Metrik:** Mean Absolute Error (MAE) auf Validation-Set  
**Richtung:** Minimierung

---

## Hyperparameter-Strategie

### Fixe Parameter

Diese Parameter sind theoretisch begründet und werden NICHT von Optuna optimiert:

**max_d = 1** (Differencing Order)
- Stationarität wird typischerweise nach einer Differenzierung erreicht
- d=2 führt oft zu Over-Differencing

**max_D = 1** (Seasonal Differencing Order)
- Standard für seasonal ARIMA
- D>1 sehr selten in der Praxis benötigt

**seasonal** (true/false)
- Dataset-spezifisch in Config festgelegt
- Booksales: true (m=7 ist rechenbar)
- Walmart: false (m=52 ist zu rechenintensiv)

### Variable Parameter

Diese Parameter werden von Optuna optimiert:

```python
SEARCH_SPACE = {
    "max_p": {"min": 0, "max": 5},  # Non-seasonal AR order
    "max_q": {"min": 0, "max": 5},  # Non-seasonal MA order
    "max_P": {"min": 0, "max": 2},  # Seasonal AR order
    "max_Q": {"min": 0, "max": 2},  # Seasonal MA order
}
```

**Hinweis:** Bei seasonal=false werden max_P und max_Q automatisch auf 0 gesetzt.

**Search Space Größe:** 6 × 6 × 3 × 3 = 324 Kombinationen

---

## Trial-Anzahl

**Standard: 20 Trials** (konsistent mit Prophet und TFT)

**Begründung:**
- TPE Sampler konvergiert typischerweise in 20-30 Trials
- Faire Vergleichbarkeit über alle Modelle
- Praktikable Balance zwischen Exploration und Rechenzeit

---

## Pruning

**Status: Deaktiviert**

**Grund:**
- ARIMA trainiert gruppenweise (ein Modell pro Gruppe)
- Gruppen sind heterogen → unterschiedliche Schwierigkeit
- MedianPruner vergleicht Gruppen falsch
- Bei TFT funktioniert Pruning (global model, Epochen-basiert)

**Konfiguration:** `pruner = None`

---

## Workflow

### Phase 1: Hyperparameter-Optimierung

```bash
# Datenbank löschen bei Neustart
Remove-Item results/arima/optuna/<dataset>/arima_studies.db -Force

# Booksales (seasonal=true)
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"
python -m src.modeling.optuna_arima --study-name arima_booksales --n-trials 20

# Walmart (seasonal=false)
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"
python -m src.modeling.optuna_arima --study-name arima_walmart --n-trials 20
```

**Output:**
- SQLite DB mit allen Trials
- CSV/JSON-Exports
- Trial-Ordner mit Modellen

**Erwartete Laufzeit:**

| Dataset | Gruppen | Seasonal | m | Zeit/Trial | 20 Trials |
|---------|---------|----------|---|------------|-----------|
| Booksales | 48 | true | 7 | 3-5 Min | 1-2 Stunden |
| Walmart | 3050 | false | - | 40-60 Min | 12-20 Stunden |

---

### Phase 2: Visualisierung und Analyse

```bash
# Visualisierungen erstellen
python -m src.visualization.plot_arima_optuna_study --study-name arima_booksales

# Statistik-Analyse
python -m src.evaluation.analyze_optuna_arima_trials --study-name arima_booksales --top-n 10
```

**Plot-Typen:**
- **Optimization History:** Verlauf val_mae über Trials
- **Parameter Importance:** Einfluss jedes Parameters
- **Parallel Coordinate:** Parameter-Kombinationen der besten Trials
- **Slice Plot:** Einfluss einzelner Parameter isoliert

---

### Phase 3: Export und Finales Training

```bash
# Beste Config exportieren
python -m src.modeling.optuna_arima_export_best --study-name arima_booksales

# Finales Training mit bester Config
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"
python -m src.modeling.trainer_arima --config configs/models/arima/booksales/optuna_best.yaml
```

---

## Outputs

### SQLite-Datenbank
**Datei:** `arima_studies.db`  
**Inhalt:** Alle Studies, Trials, Parameter, Metriken

**Zugriff:**
- Optuna-API: `optuna.load_study()`
- SQLite-Browser
- Dashboard: `optuna-dashboard sqlite:///<path>`

### CSV/JSON-Exports
**Automatisch nach jedem Study:**
- `study_<n>_<timestamp>.csv` - Alle Trials tabellarisch
- `study_<n>_<timestamp>.json` - Summary mit Metadaten

### Trial-Artefakte
**Pro Trial:** `trial_<n>/`
- `models/` - Trainierte ARIMA-Modelle für alle Gruppen
- `trial_summary.json` - Hyperparameter, Metrik, Duration

---

## Besonderheiten ARIMA

### Multi-Group Training
- Ein Modell pro Gruppe (im Gegensatz zu TFT's globalem Modell)
- Finale val_mae ist Durchschnitt über alle Gruppen
- Bei Fehlern einzelner Gruppen: Diese werden übersprungen

### Seasonal Period Impact
**Kritischer Performance-Faktor:** m hat exponentiellen Einfluss auf Rechenzeit!

| m | Beispiel | Empfehlung |
|---|----------|------------|
| 7 | Booksales (wöchentlich bei täglichen Daten) | seasonal=true ✅ |
| 52 | Walmart (jährlich bei wöchentlichen Daten) | seasonal=false (zu langsam) |

**Walmart-Beispiel:**
- m=52 bedeutet: SARIMA schaut ein ganzes Jahr (52 Wochen) zurück
- Das ist fachlich KORREKT für Jahres-Saisonalität
- ABER: Extrem rechenintensiv (Matrix-Operationen explodieren)
- Lösung: Non-seasonal ARIMA mit höherem max_p

**Speedup non-seasonal vs. seasonal:** ~12-15×

### Config-basierte Seasonal-Steuerung

**Implementation in trainer_arima.py:**
```python
seasonal = model_params.get("seasonal", True)
model = auto_arima(
    seasonal=seasonal,
    m=seasonal_period if seasonal else 1,
    max_P=max_P if seasonal else 0,
    max_Q=max_Q if seasonal else 0,
    ...
)
```

**Config-Beispiele:**

Booksales (seasonal=true):
```yaml
model:
  seasonal: true
  max_p: 3
  max_q: 3
  max_P: 2
  max_Q: 2
```

Walmart (seasonal=false):
```yaml
model:
  seasonal: false  # m=52 zu langsam!
  max_p: 5         # Erhöht für implizite Muster
  max_q: 3
  # max_P, max_Q werden ignoriert
```

---

## Workflow-Zusammenfassung

```
1. HPO durchführen (20 Trials)
   └─> optuna_arima.py
       └─> arima_studies.db

2. Analysieren
   ├─> plot_arima_optuna_study.py → Visualisierungen
   └─> analyze_optuna_arima_trials.py → Statistiken

3. Exportieren
   └─> optuna_arima_export_best.py
       └─> optuna_best.yaml

4. Finales Training
   └─> trainer_arima.py mit optuna_best.yaml
```

---

## Performance-Vergleich

| Aspekt | ARIMA (Booksales) | ARIMA (Walmart) | TFT |
|--------|-------------------|-----------------|-----|
| Trial-Dauer | 3-5 Min | 40-60 Min | 20-60 Min |
| Anzahl Trials | 20 | 20 | 20 |
| Pruning | Nein | Nein | Ja |
| Seasonal | true (m=7) | false (m=52 zu langsam) | N/A |
| Gesamt-Dauer | 1-2h | 12-20h | 10-20h |
| Modelle/Trial | 48 | 3050 | 1 (global) |

---

## Best Practices

### Database Management
```powershell
# Immer vor Neustart löschen
Remove-Item results/arima/optuna/<dataset>/arima_studies.db -Force
```

### Study Names
- Eindeutig pro Dataset: `arima_booksales`, `arima_walmart`
- Bei Änderungen am Search Space: Neue Study oder DB löschen

### Seasonal-Entscheidung
- Wenige Gruppen + niedrige m → seasonal=true
- Viele Gruppen + hohe m → seasonal=false
- Trade-off: Rechenzeit vs. Modellqualität

---

## Ergebnis und Nutzen

- Vollautomatische Hyperparameter-Suche für ARIMA/SARIMA
- Konsistente Methodik über alle Modelle (20 Trials)
- Config-basierte seasonal-Steuerung
- Persistente Speicherung aller Trials
- Umfangreiche Visualisierungs- und Analyse-Tools
- Export beliebiger Konfigurationen als YAML
- Reproduzierbare Ergebnisse durch SQLite-Storage