# Projektstruktur – FOM.tft-timeseries

**Datum:** 2025-11-21 (aktualisiert)  
**Script:** –  
**Ziel & Inhalt:** Vollständige Übersicht über die Projektstruktur. Erklärt Ordner-Rollen, Datenfluss, Zuständigkeiten und Erweiterbarkeit.

---

## 🗂️ 1. `src/` – Hauptverzeichnis

```text
src/
├── data/           # Preprocessing
├── modeling/       # Training
├── evaluation/     # Bewertung
├── utils/          # Hilfsfunktionen
├── visualization/  # Plots
├── config.py       # Globale Konstanten
└── pipeline.py     # Orchestrierung (NEU)
```

---

## 📊 2. `data/` – Datenaufbereitung

| Datei | Aufgabe |
|-------|---------|
| `data_alignment.py` | Harmonisierung und Normalisierung |
| `data_cleaning.py` | Bereinigung, Imputation |
| `feature_engineering.py` | Kalender- und Feiertags-Features |
| `cyclical_encoder.py` | Zyklische Kodierung (sin/cos) |
| `lag_features.py` | Lag- und Rolling-Features |

**Output:** `data/processed/train_features_cyc_lag.parquet`

---

## 🤖 3. `modeling/` – Training

| Datei | Aufgabe |
|-------|---------|
| `model_dataset.py` | Split in Train/Val/Test |
| `dataset_tft.py` | TFT-Datensatz-Spezifikation |
| `trainer_tft.py` | TFT-Training |
| *(geplant)* `trainer_arima.py` | ARIMA-Training |
| *(geplant)* `trainer_prophet.py` | Prophet-Training |

### Datenfluss:

```
model_dataset.py
  Eingabe: data/processed/train_features_cyc_lag.parquet
  Ausgabe: data/processed/{train,val,test}.parquet + meta.json

dataset_tft.py
  Eingabe: data/processed/{train,val,test}.parquet
  Ausgabe: data/processed/dataset_spec.json

trainer_tft.py
  Eingabe: dataset_spec.json + configs/models/tft/*.yaml
  Ausgabe:
    - Logs: logs/tft/run_YYYYMMDD_HHMMSS_<config>/
    - Checkpoints: results/tft/runs/run_YYYYMMDD_HHMMSS_<config>/checkpoints/
    - JSONs: results/tft/runs/run_YYYYMMDD_HHMMSS_<config>/{results,summary}.json
```

---

## 4. `evaluation/` – Bewertung

| Datei | Aufgabe |
|-------|---------|
| `evaluate_tft.py` | Berechnet Fehlermaße für einen Run |
| `aggregate_tft_eval.py` | Aggregiert alle Evaluierungen |

**Ausgabe:**
```
results/tft/eval/
├── <run_id>/
│   └── eval_summary.json
├── eval_overview.csv
└── eval_overview.json
```

---

## 📈 5. `visualization/` – Plots

| Datei | Aufgabe |
|-------|---------|
| `data_alignment_plot.py` | Visualisierung der Harmonisierung |
| `data_cleaning_plot.py` | Vorher/Nachher-Vergleich |
| `plot_learning_rate.py` | Lernkurven |
| `plot_tft_eval_comparison.py` | Run-Vergleiche |
| `plot_tft_forecast_series.py` | Forecast-Beispiele |

**Ausgabe:** `results/tft/plots/`

---

## 🧰 6. `utils/` – Hilfsfunktionen

| Datei | Aufgabe |
|-------|---------|
| `config_loader.py` | YAML-Validierung |
| `json_results.py` | Metriken-Export |
| `load_trained_tft.py` | Checkpoint-Loader |

---

## 🔄 7. `pipeline.py` – Orchestrierung (NEU)

**Hauptfunktion:** Orchestriert alle Schritte von Preprocessing bis Training.

**Aufruf:**
```bash
python -m src.pipeline \
    --dataset configs/datasets/booksales.yaml \
    --model configs/models/tft/baseline.yaml \
    --steps preprocessing,model_dataset,dataset_tft,training
```

**Ausgabe:** `results/pipeline_runs/pipeline_YYYYMMDD_HHMMSS_manifest.json`

---

## ⚙️ 8. `config.py` – Zentrale Steuerung

**Enthält:**
- Pfade: `RAW_DIR`, `INTERIM_DIR`, `PROCESSED_DIR`
- Schema: `DATETIME_COLUMN`, `GROUP_COLS`, `TARGET_COL`
- Feature-Configs: `LAG_CONF`, `TFT_DATASET`
- Split-Parameter: `SPLIT_RATIOS`

**NICHT enthalten:**
- Trainings-Hyperparameter → `configs/models/tft/*.yaml`
- Dataset-Definition → `configs/datasets/*.yaml`

---

## 📁 9. `configs/` – Konfigurationen

```
configs/
├── datasets/
│   ├── booksales.yaml
│   ├── retail_sales.yaml (geplant)
│   └── ecommerce.yaml (geplant)
└── models/
    ├── tft/
    │   ├── baseline.yaml
    │   └── experiments/
    │       ├── lr_high.yaml
    │       ├── bs_small.yaml
    │       └── model_large.yaml
    ├── arima/ (geplant)
    └── prophet/ (geplant)
```

---

## 📂 10. `results/` – Outputs

```
results/
├── pipeline_runs/           # Pipeline-Manifests (modellübergreifend)
└── tft/
    ├── runs/
    │   └── run_YYYYMMDD_HHMMSS_<config>/
    │       ├── checkpoints/
    │       │   └── tft-epoch=XX-val_loss=Y.YYYY.ckpt
    │       ├── results.json
    │       └── summary.json
    ├── eval/
    │   ├── <run_id>/
    │   │   └── eval_summary.json
    │   ├── eval_overview.csv
    │   └── eval_overview.json
    └── plots/
        └── eval/
            └── compare_test_smape.png
```

---

## ✅ 11. Erweiterbarkeit

### Neues Modell hinzufügen (z.B. ARIMA):

1. **Config erstellen:** `configs/models/arima/baseline.yaml`
2. **Trainer erstellen:** `src/modeling/trainer_arima.py`
3. **Pipeline erweitern:** Modelltyp-Erkennung in `src/pipeline.py`
4. **Output-Struktur:** `results/arima/runs/...`

### Neuer Datensatz:

1. **Config erstellen:** `configs/datasets/neuer_datensatz.yaml`
2. **Preprocessing anpassen:** Steps aktivieren/deaktivieren
3. **Pipeline ausführen:**
   ```bash
   python -m src.pipeline --dataset configs/datasets/neuer_datensatz.yaml \
                          --model configs/models/tft/baseline.yaml
   ```

---

## 🎯 12. Workflow-Übersicht

```
Preprocessing → model_dataset → dataset_tft → trainer_tft → evaluate_tft
     ↓              ↓               ↓              ↓             ↓
   interim/      processed/      spec.json    checkpoints/  eval_summary.json
```

**Parallel möglich:**
- Alte Arbeitsweise (einzelne Scripte) ✅
- Neue Pipeline-Orchestrierung ✅

---

Diese Struktur bleibt **übersichtlich, modular und erweiterbar** – auch bei mehreren Modellen und Datensätzen.