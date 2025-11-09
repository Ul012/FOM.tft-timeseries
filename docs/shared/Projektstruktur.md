# Projektstruktur-Erklärung: Verzeichnisaufbau im TFT-Booksales-Projekt

**Datum:** 2025-11-08 17:05  
**Ziel:** Übersicht und Bedeutung der wichtigsten Verzeichnisse im Projekt.

---

## 🗂️ 1. `src/` – Source Code

Das Verzeichnis `src/` (engl. *source*) enthält den **gesamten Python-Quellcode** des Projekts.
Es ist die zentrale Arbeitsstruktur, in der jede Datei eine **klare Verantwortlichkeit** hat.

Typischer Aufbau:
```
src/
├── data/
├── modeling/
├── utils/
└── config.py
```

---

## 📊 2. `data/` – Datenstruktur

Hier liegen **alle Datensätze**, sowohl Rohdaten als auch verarbeitete Versionen.

| Unterordner | Zweck |
|--------------|-------|
| `raw/` | Originaldaten, unverändert. Werden nie überschrieben. |
| `interim/` | Zwischenstände (z. B. nach Preprocessing oder Feature-Building). |
| `processed/` | Modellfertige Daten, z. B. Trainings- und Validierungssets. |

Beispiel:
```
data/
├── raw/
├── interim/
└── processed/
    └── model_dataset/
        └── tft/
            ├── train.parquet
            ├── val.parquet
            ├── checkpoints/
            │   └── tft-00-15.9128.ckpt
            └── dataset_spec.json
```

---

## 🤖 3. `modeling/` – Modellierung und Training

Hier liegt alles, was sich auf **Modelle** und **Training** bezieht.

| Datei / Ordner | Beschreibung |
|----------------|---------------|
| `trainer_tft.py` | Startet das Training des Temporal Fusion Transformer (TFT). |
| `load_trained_tft.py` | Lädt ein gespeichertes TFT-Modell aus einem `.ckpt`. |
| `predict_tft.py` *(optional)* | Für spätere Vorhersagen auf neuen Daten. |
| `evaluation_tft.py` *(optional)* | Bewertung der Vorhersagequalität. |

➡️ **Ziel dieses Ordners:** alle Schritte, die direkt mit Modellarchitektur, Training oder Evaluation zu tun haben.

---

## 🧰 4. `utils/` – Werkzeuge & Hilfsfunktionen

`utils` enthält **allgemeine Helfer** und kleine Tools, die unabhängig vom Modell sind.

| Datei | Zweck |
|--------|--------|
| `inspect_checkpoint.py` | Liest `.ckpt`-Dateien aus und zeigt ihre Inhalte an. |
| `file_utils.py` *(optional)* | Hilfsfunktionen zum Lesen/Schreiben von Dateien. |
| `checkpoint_utils.py` *(optional)* | Automatische Suche nach dem neuesten Checkpoint. |

➡️ Diese Module sind **modellunabhängig** und können in mehreren Projekten wiederverwendet werden.

---

## ⚙️ 5. `config.py` – Zentrale Projektkonfiguration

Diese Datei ist das **Kontrollzentrum** des Projekts.

Sie enthält:
- allgemeine Pfadangaben (`DATA_DIR`, `PROCESSED_DIR`, …),
- Konstanten für Spaltennamen (`TARGET_COL`, `GROUP_COLS`),
- Parameter für Split-Logik,
- und Hyperparameter für das TFT-Training (`TRAINER_TFT`-Dictionary).

Beispiel:
```python
TRAINER_TFT = {
    "max_epochs": 30,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "limit_train_batches": 1.0,
}
```

➡️ Vorteil: Du steuerst dein gesamtes Projekt zentral, **ohne Code zu ändern**.

---

## 🧩 6. Empfehlung für eigene Erweiterungen

| Neues Modul | Empfohlener Ort | Beispiel |
|--------------|----------------|-----------|
| Neue Modellklasse | `src/modeling/` | `trainer_lstm.py` |
| Feature Engineering | `src/data/` | `features.py` |
| Preprocessing-Skripte | `src/data/` | `preprocess.py` |
| Logging oder Utility-Skripte | `src/utils/` | `logger.py` |
| Zentrale Konfiguration | `src/config.py` | bleibt dort |

---

## 🧠 Zusammenfassung

| Ordner | Zweck | Beispiel |
|---------|--------|-----------|
| `src/data` | Datenverarbeitung und Feature Engineering | `features.py`, `split_data.py` |
| `src/modeling` | Modellarchitektur, Training, Laden, Evaluation | `trainer_tft.py`, `load_trained_tft.py` |
| `src/utils` | Hilfsfunktionen, Diagnose, Logging | `inspect_checkpoint.py` |
| `data/raw` | Rohdaten (unverändert) | `book_sales_raw.csv` |
| `data/interim` | Zwischenstände | `aligned_features.parquet` |
| `data/processed` | fertige Datasets und Modelle | `tft/checkpoints/*.ckpt` |

---

**Kurz gesagt:**  
- 🔹 `data` = alles rund um Daten.  
- 🔹 `modeling` = alles rund um Modelle.  
- 🔹 `utils` = universelle Werkzeuge.  
- 🔹 `config.py` = das Gehirn, das alles steuert.
