# Optuna – Vorbereitungsschritte für die Hyperparameter-Optimierung

**Datum:** 2025-11-17  
**Ziel & Inhalt:** Übersicht der notwendigen Schritte zur Vorbereitung einer Hyperparameter-Optimierung mit Optuna für den TFT.

---

## Überblick
Das Dokument beschreibt grundlegende Voraussetzungen, Struktur einer Optuna-Study sowie typische Parameterbereiche für das Hyperparameter-Tuning eines TFT.

---

## Ziel
Ziel ist eine klare Vorbereitung für den späteren Einsatz von Optuna ohne direkte Implementierung des Optimierungsprozesses.

---

## Technische Voraussetzungen
- installierte Optuna-Bibliothek
  ```
  pip install optuna
  ```  
- stabile Baseline-Konfiguration des TFT  
- vorbereitete Daten und funktionierender Trainingsablauf  

---

## Study-Struktur (Beispiel)
```python
optuna.create_study(
    study_name="tft_hyperparam_search",
    direction="minimize",
    sampler=TPESampler(),
    pruner=MedianPruner(),
    load_if_exists=True
)
```

---

## Parameterbereiche (Beispiele)
- hidden_size: 16–128  
- dropout: 0.05–0.3  
- learning_rate: 1e-4 – 5e-3  
- attention_head_size: 1–4  
- gradient_clip_val: 0.01–0.2  

---

## Ergebnis und Nutzen
- standardisierte Vorbereitung für spätere HPO-Prozesse  
- klare Definition der relevanten Parameterbereiche  
