# DataCleaning – Bereinigung von Ausreißern und Datenqualitätsproblemen

**Datum:** 2025-11-24 (aktualisiert)  
**Script:** `src/data/data_cleaning.py`  
**Ziel & Inhalt:** Bereinigung von Ausreißern, Lockdown-Zeiträumen, negativen Werten und NaN im Target. Universell für verschiedene Datensätze (Booksales, Walmart, etc.).

---

## Überblick
Das Modul **DataCleaning** korrigiert Datenqualitätsprobleme in den Verkaufsdaten:

- **Einzelausreißer** an bestimmten Daten (z.B. 01.01.2020)
- **Lockdown-Perioden** mit Glättung (März–Mai 2020)
- **Negative Werte** im Target clippen (z.B. auf 0)
- **NaN im Target** entfernen (TFT-Anforderung)
- **Datentyp-Konvertierung** auf float32

---

## Konfiguration (YAML)

Die Parameter werden aus der Dataset-YAML geladen:

```yaml
preprocessing:
  - step: "cleaning"
    enabled: true
    params:
      # Booksales-spezifisch:
      outlier_dates: ["2020-01-01"]
      lockdown_start: "2020-03-15"
      lockdown_end: "2020-05-31"
      
      # Walmart-spezifisch:
      clip_target_min: 0  # Negative Werte auf 0 setzen
      remove_nan: true    # NaN im Target entfernen
```

### Parameter

| Parameter | Typ | Beschreibung |
|-----------|-----|--------------|
| `outlier_dates` | Liste | Einzelne Ausreißer-Daten (werden auf NaN gesetzt und interpoliert) |
| `lockdown_start` | String | Start der Lockdown-Periode |
| `lockdown_end` | String | Ende der Lockdown-Periode |
| `clip_target_min` | Float | Minimaler Target-Wert (kleinere werden geclippt) |
| `clip_target_max` | Float | Maximaler Target-Wert (größere werden geclippt) |
| `remove_nan` | Bool | Ob NaN im Target entfernt werden (default: true) |

---

## Vorgehen

### 1. Ausreißer (Einzeldaten)
```python
self.handle_single_day_outlier("2020-01-01")
self._fill_with_shifted_mean(periods=365, repeats=3)
```

### 2. Lockdown-Glättung
```python
self.handle_lockdown_period("2020-03-15", "2020-05-31")
self._fill_with_shifted_mean(periods=365, repeats=3)
```

### 3. Target clippen (NEU)
```python
self.clip_target(clip_min=0)  # Negative → 0
```

### 4. Datentyp konvertieren (NEU)
```python
self.convert_target_dtype()  # → float32
```

### 5. Target-NaN entfernen (NEU)
```python
self.remove_target_nan()  # Zeilen mit NaN im Target entfernen
```

Alle Schritte arbeiten **gruppenweise** (nach ID-Spalten sortiert).

---

## Beispielaufruf

```bash
# Walmart
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.data_cleaning

# Booksales
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.data_cleaning
```

**Output:**  
`data/interim/<dataset_name>/train_cleaned.parquet`

```

---

## Ergebnis und Nutzen
- Korrektur von Einzelausreißern und Lockdown-Zeiträumen
- Negative Werte werden auf 0 geclippt (wichtig für Walmart)
- Target ist float32 und NaN-frei (TFT-Anforderung)
- Konsistente Zeitreihen für das weitere Feature-Engineering