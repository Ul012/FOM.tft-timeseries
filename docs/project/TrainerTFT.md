# TrainerTFT – Training des Temporal Fusion Transformer

**Datum:** 2025-11-24 (aktualisiert)  
**Script:** `src/modeling/trainer_tft.py`  
**Ziel & Inhalt:** Beschreibt den Ablauf des TFT-Trainings auf Basis der vorbereiteten Datensätze und der YAML-Konfiguration.

---

## Überblick

`trainer_tft.py` führt das Training des **Temporal Fusion Transformer (TFT)** aus.  
Ausgehend von der Datensatzspezifikation (`dataset_spec.json`) und den Train-/Val-Daten werden:

- ein `TimeSeriesDataSet` für Training und Validation aufgebaut,
- ein TFT-Modell mit den in der YAML-Datei definierten Hyperparametern erzeugt,
- das Modell mit PyTorch Lightning trainiert (inkl. Early Stopping und Checkpointing),
- Trainings- und Validierungsmetriken protokolliert und als Run-Artefakte abgelegt.

Das Script enthält **keine Feature-Erzeugung oder Datenbereinigung** – es arbeitet ausschließlich auf den modellfertigen Eingaben.

---

## Voraussetzungen (Preprocessing)

Folgende Schritte müssen **vor** dem Training abgeschlossen sein:

1. **`data_cleaning.py`** – Target bereinigt (float32, keine NaN, optional geclippt)
2. **`lag_features.py`** – Lag-Features erstellt, NaN imputiert, kurze Gruppen gefiltert
3. **`model_dataset.py`** – Train/Val/Test Split
4. **`dataset_tft.py`** – Feature-Listen in `dataset_spec.json`

---

## Eingaben und Ausgaben

### Eingaben

1. **Modellfertige Datensätze**:
   - `data/processed/<dataset>/train.parquet`
   - `data/processed/<dataset>/val.parquet`
   - `data/processed/<dataset>/dataset_spec.json`

2. **YAML-Konfiguration** (z.B. `configs/models/tft/<dataset>/baseline.yaml`)

3. **Target Normalizer Transformation** (optional, dataset-spezifisch):
   - `"softplus"` (Default) – Für strikt positive Werte
   - `null` – Standard z-score Normalisierung
   - `"relu"` – Clippt negative auf 0
   - `"log"` – Für log-normalverteilte Daten

### Ausgaben

```
results/tft/runs/run_<timestamp>_<dataset>_<config>/
├── checkpoints/
│   └── tft-epoch=XX-val_loss=Y.YYYY.ckpt
├── results.json
└── summary.json

logs/tft/run_<timestamp>_<dataset>_<config>/
├── metrics.csv
└── hparams.yaml
```

---

## Verarbeitungsschritte im Trainer

### 1. ID-Spalten zu String konvertieren
TFT benötigt kategorische Variablen als String:
```python
for col in static_categoricals:
    df[col] = df[col].astype(str)
```

### 2. TimeSeriesDataSet erstellen
Feature-Listen werden aus `dataset_spec.json` geladen und explizit übergeben.

### 3. Training mit PyTorch Lightning
- Early Stopping auf `val_loss`
- ModelCheckpoint (bestes Modell)
- LearningRateMonitor

---

## Konfiguration (YAML-Struktur)

```yaml
type: "tft"
name: "<config_name>"

training:
  seed: <int>
  max_epochs: <int>
  batch_size: <int>
  learning_rate: <float>
  gradient_clip_val: <float>
  early_stopping_patience: <int>
  accelerator: "gpu" | "cpu"
  devices: <int>
  num_workers: <int>

model:
  target_normalizer_transformation: "softplus" | null | "relu" | "log"
  loss: "quantile" | "mse"
  output_size: <int>           # Anzahl Quantile (bei quantile loss)
  hidden_size: <int>
  attention_head_size: <int>
  hidden_continuous_size: <int>
  dropout: <float>
  reduce_on_plateau_patience: <int>
```

---

## Beispielaufruf

```bash
# Via Pipeline (empfohlen)
python -m src.pipeline \
    --dataset configs/datasets/<dataset>.yaml \
    --model configs/models/tft/<dataset>/<config>.yaml \
    --steps training

# Einzeln
$env:DATASET_CONFIG="configs/datasets/<dataset>.yaml"
python -m src.modeling.trainer_tft --config configs/models/tft/<dataset>/<config>.yaml
```

---

## Aufgabenverteilung

### Im Preprocessing (vor Trainer)

| Aufgabe | Modul |
|---------|-------|
| Target clippen | `data_cleaning.py` |
| Target auf float32 | `data_cleaning.py` |
| Target-NaN entfernen | `data_cleaning.py` |
| Lag-NaN Imputation | `lag_features.py` |
| Kurze Gruppen filtern | `lag_features.py` |

### Im Trainer

| Aufgabe | Grund |
|---------|-------|
| ID-Spalten zu String | TFT-spezifische Anforderung |
| TimeSeriesDataSet erstellen | TFT-spezifisch |
| Training durchführen | Kernaufgabe |

---

## Ergebnis und Nutzen

Nach einem erfolgreichen Lauf liegen vor:

- Trainiertes TFT-Modell (Checkpoint)
- Trainings- und Validierungsmetriken je Epoche
- Run-Zusammenfassung in JSON-Form
- Logger-Artefakte für weitergehende Analysen