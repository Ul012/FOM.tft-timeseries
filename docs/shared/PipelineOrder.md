# PipelineOrder – Reihenfolge und Abhängigkeiten

**Datum:** 2026-02-21  
**Ziel & Inhalt:** Definiert die Reihenfolge der Pipeline-Schritte und die zentralen Abhängigkeiten zwischen Preprocessing, Training und Evaluation.

---

## Überblick

Die Pipeline ist modular aufgebaut. Jeder Schritt erzeugt Artefakte, die vom nachfolgenden Schritt als Eingabe genutzt werden. Dadurch bleiben Runs reproduzierbar und Schritte austauschbar.

---

## Reihenfolge (typisch)

1. **LoadRaw** – Rohdaten einlesen und ablegen  
2. **DataCleaning** – Qualitätschecks und Bereinigung  
3. **FeatureEngineer** – Feature-Erzeugung (z. B. Kalender/zyklisch)  
4. **LagFeatures** – Lag-/Rolling-Features, Gruppenfilter (falls vorgesehen)  
5. **ModelDataset** – Train/Val/Test Split als Parquet  
6. **DatasetTFT** – `dataset_spec.json` für TFT  
7. **TrainerTFT** – Training aus YAML + `dataset_spec.json`  
8. **EvaluatorTFT** – Bewertung eines Runs auf Val/Test (Checkpoint-basiert)  

---

## Artefakt-Schnittstellen

### Von Preprocessing zu Modeling
- `train.parquet`, `val.parquet`, `test.parquet`  
- (TFT) `dataset_spec.json` als Spezifikation der Feature-Listen  

### Von Training zu Evaluation
- Checkpoints (bestes Modell)  
- Run-Metadaten / JSON-Summaries / Logs  

Hinweis: Die TFT-Evaluation verwendet Dataset-Parameter aus dem Modell-Checkpoint, um Trainingsbedingungen exakt zu reproduzieren.

---

## Ergebnis und Nutzen

- Klare Abhängigkeiten zwischen Schritten  
- Einheitliche Artefakte ermöglichen Modellvergleiche  
- Reduzierte Kopplung zwischen Modulen durch definierte Schnittstellen
