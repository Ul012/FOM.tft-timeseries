# Konfigurationen – Zusammenspiel von `src/config.py` und `configs/*.yaml`

**Datum:** 2026-02-21  
**Script:** —  
**Ziel & Inhalt:** Beschreibt die Konfigurationslogik des Projekts mit Fokus auf YAML-first, klare Verantwortlichkeiten und reproduzierbare Runs. Keine konkreten Hyperparameterwerte, keine Ergebnisinterpretation.

---

## Überblick

Die Projektkonfiguration ist konsequent **konfigurationsgetrieben** aufgebaut:

- `src/config.py` enthält **projektweite Konstanten** und Basis-Pfade (selten geändert).
- `configs/datasets/*.yaml` definiert **Dataset-Schema und Preprocessing/Split**.
- `configs/models/*.yaml` definiert **Training und Modellparameter** (experimentabhängig).
- `dataset_spec.json` ist eine **operative Schnittstelle** (v. a. für TFT), die aus der Pipeline erzeugt und von Training/Evaluation genutzt wird.

---

## Rollen der Konfigurationsartefakte

### 1) `src/config.py`
Enthält stabile Projektkonstanten, z. B.:

- Basisverzeichnisse und Default-Pfade
- Default-Konstanten/Enums (falls genutzt)
- ggf. projektweite Namenskonventionen

Wichtig: `config.py` ist **nicht** der primäre Ort für experimentelle Modellparameter.

---

### 2) Dataset-Configs: `configs/datasets/*.yaml`
Definieren dataset-spezifisch:

- `paths` (raw/interim/processed)
- `schema` (time_col, id_cols, target_col)
- `preprocessing` (Schrittfolge und Parameter)
- `split` (zeitbasierte Split-Logik)
- ggf. (TFT) dataset-nahe Einstellungen, die die Datensatzrepräsentation betreffen

Damit lassen sich mehrere Datensätze ohne Codeänderungen betreiben.

---

### 3) Model-Configs: `configs/models/*.yaml`
Definieren experiment-/run-spezifisch:

- Trainingseinstellungen (z. B. Seed, Epochen, Accelerator)
- Modellparameter (Architektur-/Loss-Konfiguration)

Ziel: Experimente variieren über YAML, nicht über Code.

---

### 4) `dataset_spec.json` (TFT)
Wird durch die Pipeline generiert und dient als Schnittstelle zwischen Preprocessing und Modeling.

Typischer Inhalt (konzeptionell):

- Feature-Listen / Merkmalsgruppen für TFT
- Sequenz-/Fensterlängen (datasetbezogen)
- Meta-Informationen zum Schema

Hinweis: Die TFT-Evaluation nutzt die im Checkpoint gespeicherten Dataset-Parameter, um exakt die Trainingsbedingungen zu reproduzieren.

---

## Hierarchie und Prioritäten

1. YAML-Konfigurationen (`configs/datasets`, `configs/models`) sind **führend** für Run-Verhalten.  
2. `src/config.py` liefert projektweite Defaults/Grundlagen, sollte aber nicht zum Experimentieren genutzt werden.  
3. `dataset_spec.json` ist ein generiertes Artefakt und wird als Input für Training/Evaluation verwendet (TFT).

---

## Nutzen

- Klare Trennung: Projektkonstanten vs. Dataset vs. Modell/Training  
- Reproduzierbare Runs (Konfiguration + Artefakte)  
- Multi-Dataset- und Multi-Model-Support ohne Codeänderungen  
- Saubere Schnittstellen entlang der Pipeline
