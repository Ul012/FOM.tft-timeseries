# Trainer_TFT – Lauf erklärt (aktuelle Version)

**Script:** `src/modeling/trainer_tft.py`  
**Kontext:** Trainingslauf eines Temporal Fusion Transformer (TFT) mit CSV-Logging und JSON-Export.

Dieses Dokument beschreibt, was beim Start von `trainer_tft.py` passiert, welche Artefakte entstehen und wie die wichtigsten Logzeilen zu lesen sind.

---

## 1. Aufruf und Konfiguration

Typischer Aufruf:

```bash
python -m src.modeling.trainer_tft --config configs/trainer_tft_baseline.yaml
```

- Die Datei unter `configs/*.yaml` enthält **alle Trainings- und Modell-Hyperparameter** (Learning Rate, Batch Size, Epochen, Modellgrößen, Device etc.).
- `src/config.py` liefert nur **statische Projektkonstanten** wie Pfade (`PROCESSED_DIR`) und Spaltennamen (`TIME_COL`, `ID_COLS`, `TARGET_COL`).
- Die YAML wird mit einem strikten Loader (`load_trainer_cfg`) eingelesen – **keine Fallbacks im Code**, fehlende Felder führen zu Fehlern.

Wichtige Felder aus der YAML (Auszug):

- `seed`
- `batch_size`
- `learning_rate`
- `max_epochs`
- `gradient_clip_val`
- `early_stopping_patience`
- `accelerator` (z. B. `"cpu"` oder `"gpu"`)
- `devices` (z. B. `1`)
- `limit_train_batches`, `limit_val_batches`
- `num_workers`
- `model.*` (z. B. `hidden_size`, `dropout`, `output_size`, `attention_head_size`)

---

## 2. Datasets und DataLoader

Der Trainer nutzt die zuvor erzeugten Dateien aus `data/processed/`:

1. `model_dataset.py` hat `train.parquet`, `val.parquet`, `test.parquet` sowie `meta.json` erzeugt.
2. `dataset_tft.py` hat daraus `dataset_spec.json` geschrieben.

`trainer_tft.py` liest:

- `data/processed/dataset_spec.json`
- die darin referenzierten Pfade zu `train.parquet` und `val.parquet`

Daraus werden zwei `TimeSeriesDataSet`-Objekte erstellt:

- `train_ds` für das Training
- `val_ds` für die Validierung

Wichtige Punkte:

- Die Zielspalte (`TARGET_COL`, z. B. `num_sold`) wird als `float32` gecastet.
- Als Zeitindex wird `time_idx` genutzt, falls vorhanden, sonst `TIME_COL` (z. B. `date`).
- Für die Zielnormalisierung wird `GroupNormalizer` mit `groups=ID_COLS` verwendet.

Die DataLoader:

- `train_loader = train_ds.to_dataloader(train=True, batch_size=..., num_workers=...)`
- `val_loader = val_ds.to_dataloader(train=False, ...)`

**`num_workers`** steuert, wie viele Hintergrundprozesse Daten vorbereiten. Unter Windows sind typischerweise 2–8 sinnvoll.

---

## 3. Modellinitialisierung

Das Modell wird mit `TemporalFusionTransformer.from_dataset(...)` aus `pytorch_forecasting` aufgebaut.  
Die meisten Parameter stammen direkt aus der YAML-Konfiguration:

- `learning_rate = cfg.learning_rate`
- `hidden_size`, `hidden_continuous_size`, `attention_head_size`
- `dropout`
- `reduce_on_plateau_patience`
- `output_size` (bei QuantileLoss meist Anzahl der Quantile)

Verlustfunktion:

- Wenn `cfg.model.loss == "mse"`:
  - `loss = torch.nn.MSELoss()`
  - `output_size = 1`
- Sonst:
  - `loss = QuantileLoss()`
  - `output_size` aus der YAML (z. B. 3 für P10/50/90)

Zusätzlich werden Standardmetriken für das Logging aktiviert:

- `MAE`, `RMSE`, `MAPE`, `SMAPE`

---

## 4. Logging, Checkpoints und Run-ID

Für jeden Lauf wird eine eindeutige **Run-ID** erzeugt, z. B.:

```text
run_20251114_162530
```

Diese Run-ID steuert die Pfade für Logs und Checkpoints:

- **Checkpoints:**  
  `results/tft/checkpoints/run_YYYYMMDD_HHMMSS/`
- **Logs (CSV):**  
  `logs/tft/run_YYYYMMDD_HHMMSS/`

### 4.1 Logger

Es wird ein `CSVLogger` verwendet:

- `save_dir="logs"`
- `name="tft"`
- `version=run_id`

Dadurch entsteht pro Run ein Ordner:

```text
logs/
└─ tft/
   └─ run_YYYYMMDD_HHMMSS/
      ├─ metrics.csv
      ├─ hparams.yaml
      └─ (weitere Hilfsdateien)
```

### 4.2 Checkpoints

Checkpoints werden über `ModelCheckpoint` erstellt:

- `dirpath = results/tft/checkpoints/run_YYYYMMDD_HHMMSS/`
- `filename = "tft-{epoch:02d}-{val_loss:.4f}"`
- `monitor = "val_loss"`
- `mode = "min"`
- `save_top_k = 1` (nur das beste Checkpoint pro Run)

Im Log erscheint nach dem Training u. a.:

```text
[trainer_tft] Bestes Checkpoint: results/tft/checkpoints/run_.../tft-XX-YY.YYYY.ckpt
```

Dieses `.ckpt` kann später für Vorhersagen geladen werden.

---

## 5. Trainingsphase und Konsolenausgabe

Während des Trainings zeigt Lightning pro Epoche u. a. an:

- aktuelle Epoche (`Epoch 1`, `Epoch 2`, …)
- `train_loss_step` (Loss eines Batches)
- `train_loss_epoch` (durchschnittlicher Trainings-Loss)
- `val_loss` (Validierungs-Loss)

Wichtige Punkte zur Interpretation:

- **Tendenz wichtiger als Einzelwerte** – idealerweise sinken `train_loss_epoch` und `val_loss` im Verlauf.
- Ein deutlich höherer `val_loss` im Vergleich zu `train_loss_epoch` kann auf Overfitting hindeuten.
- Die absolute Höhe des Losses ist von der Lossdefinition abhängig (z. B. MSE vs. QuantileLoss).

---

## 6. JSON-Export der Ergebnisse

Nach Abschluss des Trainings ruft der Trainer:

```python
results_path, summary_path = export_run_jsons_from_metrics(...)
```

auf. Dieser Helper:

- liest die `metrics.csv` des Runs aus `logs/tft/run_.../`
- schreibt zwei JSON-Dateien nach:

```text
results/evaluation/run_YYYYMMDD_HHMMSS/results.json
results/evaluation/run_YYYYMMDD_HHMMSS/summary.json
```

### Inhalt:

- `results.json`: detaillierte Verläufe der Metriken (z. B. pro Epoche)
- `summary.json`: Meta-Infos zum Run, z. B.:
  - verwendete YAML-Konfiguration (`config_file`, `config_values`)
  - Seed, Laufzeit, Anzahl der Epochen
  - bestes Checkpoint
  - zentrale Hyperparameter (Learning Rate, Batch Size, max_epochs, …)

Die Konsole zeigt am Ende z. B.:

```text
[trainer_tft] JSON-Export abgeschlossen:
  - results/evaluation/run_.../results.json
  - results/evaluation/run_.../summary.json
```

---

## 7. Typischer Gesamtpfad eines Runs

Zusammengefasst entstehen bei einem vollständigen TFT-Training folgende Artefakte:

```text
data/
  processed/
    train.parquet
    val.parquet
    test.parquet
    dataset_spec.json

logs/
  tft/
    run_YYYYMMDD_HHMMSS/
      metrics.csv
      hparams.yaml
      (weitere Log-Dateien)

results/
  tft/
    checkpoints/
      run_YYYYMMDD_HHMMSS/
        tft-XX-YY.YYYY.ckpt

  evaluation/
    run_YYYYMMDD_HHMMSS/
      results.json
      summary.json
```

---

## 8. Nächste sinnvolle Schritte

- **Evaluator:** Skript schreiben bzw. nutzen, das alle `summary.json` in `results/evaluation/` einliest und eine Vergleichstabelle (z. B. `runs_summary.csv`) erzeugt.
- **Visualizer:** Lernkurven und Vergleichsplots aus `metrics.csv` und `runs_summary.csv` erstellen.
- **Hyperparameter-Studien:** Neue YAML-Dateien in `configs/` anlegen, um systematisch Varianten von Learning Rate, Fensterlängen oder Modellgrößen zu testen.
