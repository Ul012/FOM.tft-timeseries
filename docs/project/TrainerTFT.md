# TrainerTFT – Training des Temporal Fusion Transformer

**Datum:** 2025-11-17  
**Script:** `src/modeling/trainer_tft.py`  
**Ziel & Inhalt:** Beschreibt den Ablauf des TFT-Trainings auf Basis der vorbereiteten Datensätze und der YAML-Konfiguration. Erläutert Eingaben, Konfigurationsblöcke, Artefakte pro Run und die interne Auswertung der Trainingsmetriken.

---

## Überblick

`trainer_tft.py` führt das eigentliche Training des **Temporal Fusion Transformer (TFT)** aus.  
Ausgehend von der Datensatzspezifikation (`dataset_spec.json`) und den Train-/Val-Daten werden:

- ein `TimeSeriesDataSet` für Training und Validation aufgebaut,
- ein TFT-Modell mit den in der YAML-Datei definierten Hyperparametern erzeugt,
- das Modell mit PyTorch Lightning trainiert (inkl. Early Stopping und Checkpointing),
- Trainings- und Validierungsmetriken protokolliert und als Run-Artefakte abgelegt.

Das Script selbst enthält **keine Feature-Erzeugung** mehr – es arbeitet ausschließlich auf den modellfertigen Eingaben aus `dataset_tft.py`.

---

## Eingaben und Ausgaben

### Eingaben

1. **Modellfertige Datensätze** (aus `model_dataset.py` und `dataset_tft.py`):  
   - `data/processed/train.parquet`  
   - `data/processed/val.parquet`  
   - `data/processed/test.parquet` (optional, für spätere Evaluation)  
   - `data/processed/dataset_spec.json` (Feature-Listen, Sequenzlängen, Spaltenkonfiguration)

2. **YAML-Konfiguration** im Ordner `configs/`  
   Eine ausgewählte Datei (z. B. `configs/tft_baseline.yaml`) steuert u. a.:
   - Trainingsparameter (Epochenzahl, Batchgröße, Early-Stopping-Patience),
   - Modellarchitektur (Hidden Size, Anzahl Attention-Heads, Dropout),
   - Optimierer-Einstellungen (Lernrate, Gewichtung der Loss-Funktion),
   - Logging-Optionen (Ausgabeintervall, Speichern von Metriken).

3. **Target Normalizer Transformation** (dataset-spezifisch)  
   Optional in Model-YAML unter `model.target_normalizer_transformation`:
   - `"softplus"` (Default) - Funktioniert für positive Werte (Booksales)
   - `null` - Standard-Normalisierung (z-score) für robustere Daten (Walmart)
   - `"relu"` - Clippt negative Werte auf 0
   - `"log"` - Für log-normalverteilte Daten

   **Beispiel (walmart/baseline.yaml):**
   ```yaml
   model:
     target_normalizer_transformation: null  # Standard statt softplus
   ```

4. **Globale Konstanten aus `src/config.py`**  
   - Pfade wie `PROCESSED_DIR` und `BASE_DIR`,  
   - Spaltenkonstanten (`TIME_COL`, `ID_COLS`, `TARGET_COL`),  
   - Sequenzlängen (`TFT_DATASET["max_encoder_length"]`, `max_prediction_length`).

### Ausgaben

Pro Training entsteht ein **Run-Ordner**, typischerweise:

```text
results/
└─ tft/
   └─ runs/
      └─ run_YYYYMMDD_HHMMSS_suffix/
         ├─ checkpoints/
         │  ├─ best.ckpt
         │  └─ last.ckpt
         ├─ metrics.csv
         └─ train_summary.json
```

Zusätzlich werden Logger-Artefakte (z. B. für TensorBoard oder CSV-Logger) im Ordner `logs/tft/` abgelegt, meist gespiegelt zur Run-ID.

- **`checkpoints/`** – enthält mindestens das beste und das letzte Modellcheckpoint.  
- **`metrics.csv`** – tabellarische Übersicht der Trainings- und Validierungsmetriken pro Epoche.  
- **`train_summary.json`** – kompakte Zusammenfassung des Runs (beste Epoche, zugehörige Metriken, verwendete Konfiguration).

Diese Artefakte dienen als Grundlage für spätere Auswertungen (`evaluate_tft.py`, Plot-Skripte und ggf. MLflow-Integration).

---

## Konfiguration (YAML-Struktur auf hoher Ebene)

Die konkrete YAML-Struktur kann variieren, folgt aber typischerweise diesen Blöcken:

1. **Allgemein / Run-Metadaten**  
   - Name oder Kürzel des Experiments  
   - Seed-Einstellungen für Reproduzierbarkeit

2. **Datenblock**  
   - Pfad zu `dataset_spec.json`  
   - Hinweis, welche Splits verwendet werden (Train/Val, optional Test)

3. **Modellblock**  
   - TFT-spezifische Hyperparameter (z. B. `hidden_size`, `attention_head_size`, `dropout`, `loss`)  
   - Aktivierung der Quantile-Loss-Funktion (falls genutzt)

4. **Trainerblock**  
   - `max_epochs`  
   - `gradient_clip_val`  
   - Early-Stopping-Kriterien (z. B. überwachte Metrik, Patience)  
   - Anzahl Devices / GPU-Nutzung

5. **Logging- und Checkpointing-Block**  
   - Speicherorte für Logs und Checkpoints (unterhalb von `BASE_DIR`)  
   - Namensschema für Run-Ordner und Checkpoints

Die YAML-Datei wird im Script geladen und in geeignete Konfigurationsobjekte oder Dictionaries überführt.

---

## Ablauf (End-to-End)

1. **Aufruf des Scripts**  

   Im Projektkontext wird das Training über das Modul gestartet, z. B.:

   ```bash
   python -m src.modeling.trainer_tft
   ```

   Die Auswahl der YAML-Konfiguration erfolgt entweder über ein Kommandozeilenargument oder eine im Script hinterlegte Standardkonfiguration.

2. **Laden von Konfiguration und Datensatzspezifikation**  
   - YAML-Datei einlesen  
   - `dataset_spec.json` aus `PROCESSED_DIR` laden  
   - Pfade zu Train-/Val-Dateien bestimmen

3. **Aufbau von `TimeSeriesDataSet` und Dataloaders**  
   - `TimeSeriesDataSet` für Training gemäß `dataset_spec` erzeugen  
   - separates `TimeSeriesDataSet` für Validation erzeugen  
   - Konfiguration von Batchgröße und Num-Workers gemäß YAML

4. **Initialisierung des TFT-Modells**  
   - Erzeugen eines `TemporalFusionTransformer`-Modells mit den Hyperparametern aus dem Modellblock  
   - Anbindung der Loss-Funktion und weiteren Optionen (z. B. Quantile-Loss, Output-Quantile)

5. **Einrichten von Trainer und Callbacks**  
   - PyTorch-Lightning-Trainer mit:
     - Early-Stopping-Callback (Überwachung einer Validierungsmetrik)  
     - ModelCheckpoint-Callback (Speichern des besten Checkpoints)  
     - optionalem Learning-Rate-Monitor
   - Konfiguration von Device (CPU/GPU) und Präzision

6. **Durchführen des Trainingslaufs**  
   - Aufruf von `trainer.fit(model, train_dataloader, val_dataloader)`  
   - Pro Epoche: Logging von Trainings- und Validierungsmetriken  
   - Speichern der Checkpoints im Run-Ordner

7. **Interne Auswertung der Trainingsmetriken**  
   Nach Abschluss des Trainings werden die protokollierten Metriken ausgewertet, typischerweise:

   1. Laden der Metrikhistorie (z. B. aus dem Logger).  
   2. Identifikation der Epoche mit der besten Validierungskennzahl (z. B. minimaler `val_loss`).  
   3. Schreiben eines kompakten `train_summary.json` mit:
      - `run_id`  
      - Pfad zum besten Checkpoint  
      - Metriken der besten Epoche (Train/Val)  
      - Hinweisen auf die verwendete YAML-Konfiguration.

   Diese Auswertung verändert keine Modellparameter, sondern fasst lediglich die wichtigsten Ergebnisse des Runs zusammen.

---

## Ergebnis und Nutzen

Nach einem erfolgreichen Lauf von `trainer_tft.py` liegen für einen Run:

- ein trainiertes TFT-Modell (inkl. bestmöglichem Checkpoint),
- vollständige Trainings- und Validierungsmetriken je Epoche,
- eine verdichtete Run-Zusammenfassung in JSON-Form,
- Logger-Artefakte für weitergehende Analysen (z. B. in TensorBoard)

vor.

Danach geht es weiter mit:

- detaillierte Evaluierungen mit `evaluate_tft.py`,  
- den Vergleich mehrerer Runs (z. B. mit `aggregate_tft_eval.py`),  
- die spätere Einbindung von Tools wie MLflow, ohne die Struktur des Trainers grundlegend ändern zu müssen.