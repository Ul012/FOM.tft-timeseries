# 📁 Projektstruktur – FOM.tft-timeseries

**Stand:** November 2025  
**Ziel:** Übersicht über Aufbau, Zuständigkeiten und künftige Erweiterungen des Projekts (TFT-, ARIMA- und Prophet-Pipelines).

---

## 🗂️ 1. `src/` – Hauptverzeichnis

```
src/
├── data/
├── modeling/
├── utils/
├── visualization/
└── config.py
```

---

## 📊 2. `data/` – Datenaufbereitung (Preprocessing)

Beinhaltet alle Schritte bis zur Erstellung eines modellfertigen Datensatzes.

| Datei | Aufgabe |
|-------|----------|
| `data_alignment.py` | Harmonisierung und optionale Normalisierung der Zeitachsen. |
| `data_cleaning.py` | Bereinigung, Imputation, Konsistenzprüfungen. |
| `feature_engineering.py` | Erstellung von Kalender- und Feiertags-Features. |
| `cyclical_encoder.py` | Zyklische Kodierung periodischer Variablen (sin/cos). |
| `lag_features.py` | Erzeugt Lag- und Rolling-Features per `groupby().shift()`. |
| `view_data.py` | Kurze visuelle Kontrolle der Roh- und Zwischendaten. |

> **Hinweis:** Diese Module bilden die ersten Schritte der Pipeline und erzeugen den Input für `model_dataset.py`.

---

## 🤖 3. `modeling/` – Modellierung und Training

Enthält alle Skripte zur Vorbereitung, Spezifikation und zum Training der Modelle.

| Datei | Aufgabe |
|-------|----------|
| `model_dataset.py` | Split in Train/Validation/Test, Metadaten erzeugen. |
| `dataset_tft.py` | Leitet Feature-Listen (known/unknown/static) ab, erstellt `dataset_spec.json`. |
| `trainer_tft.py` | Trainiert den Temporal Fusion Transformer, speichert Logs + Checkpoints. |
| *(geplant)* `trainer_arima.py` | ARIMA-Modelltraining auf aggregierten oder einzelnen Zeitreihen. |
| *(geplant)* `trainer_prophet.py` | Prophet-Training mit automatischer Saisonalitätserkennung. |

---

## 🧰 4. `utils/` – Hilfsfunktionen & Werkzeuge (nicht Pipeline-Pflicht)

Dient zur Wiederverwendung und modularen Wartung.

| Datei | Aufgabe |
|-------|----------|
| `config_loader.py` | Lädt und validiert die Projekt-Konfiguration. |
| `json_results.py` | Zusammenfassung, Konvertierung und Export von Ergebnis-JSONs. |
| `load_trained_tft.py` | Lädt das zuletzt trainierte oder beste TFT-Checkpoint-Modell (optional). |
| `__init__.py` | Kennzeichnung als Paket; ggf. globale Utility-Imports. |

> Utils-Skripte können **importiert** oder **manuell ausgeführt** werden, erzeugen aber keine neuen Datenstufen.

---

## 📈 5. `visualization/` – Plots und Diagnosen (Evaluationsebene)

Fasst alle Visualisierungen zusammen, die nach oder während des Trainings benötigt werden.

| Datei | Aufgabe |
|-------|----------|
| `data_alignment_plot.py` | Visualisierung der Datenharmonisierung. |
| `data_cleaning_plot.py` | Darstellung bereinigter Werte, Vergleich Vorher/Nachher. |
| `plot_learning_rate.py` | Verläufe der Learning-Rate und der Loss-Kurven. |
| `view_data_plot.py` | Allgemeine Explorations-Plots für Datenverständnis. |
| *(geplant)* `evaluation_plot.py` | Darstellung der finalen Modellvergleiche (TFT vs. ARIMA vs. Prophet). |

---

## 📊 6. Evaluierung (geplant)

Geplant ist ein eigener Ordner `src/evaluation/`, der folgende Skripte enthalten wird:

| Datei | Aufgabe |
|-------|----------|
| `evaluate_tft.py` | Evaluation der TFT-Runs (Metriken, Fehlermaße, JSON-Reports). |
| `evaluate_comparison.py` | Cross-Modell-Vergleich (TFT vs. ARIMA vs. Prophet). |

---

## ⚙️ 7. `config.py` – Zentrale Steuerung

- Globale Konstanten: `TIME_COL`, `TARGET_COL`, `GROUP_COLS`  
- Pfade: `RAW_DIR`, `PROCESSED_DIR`, `MODEL_DIR`  
- Parameter: `LAG_CONF`, `TFT_TRAIN_CONF`, u. a.  
- Keine Hardcodierung in Skripten – jede Komponente importiert Konfigurationswerte.

---

## ✅ 8. Einordnung

- **Pipeline-relevant:** `src/data/` → `src/modeling/`  
- **Unterstützend, optional:** `src/utils/`, `src/visualization/`, später `src/evaluation/`  
- **Erweiterbar:** Zusätzliche Trainer-Module (`trainer_arima.py`, `trainer_prophet.py`) folgen demselben Muster wie `trainer_tft.py`.

---
