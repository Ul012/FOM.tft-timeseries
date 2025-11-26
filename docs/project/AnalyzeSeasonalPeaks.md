# Seasonal Scripts – Saisonale Muster analysieren

**Datum:** 2025-11-25  
**Scripts:** `src/data/seasonal_overview.py`, `src/data/seasonal_analysis.py`  
**Ziel & Inhalt:** Analyse saisonaler Verkaufsmuster zur Identifikation von Peak-Zeiträumen für die `date_flags` Konfiguration.

---

## Überblick

Es gibt zwei Scripts mit unterschiedlichem Detailgrad:

| Script | Zweck | Ausgabe |
|--------|-------|---------|
| `seasonal_overview.py` | Schnelle Übersicht (KISS) | Nur Daten |
| `seasonal_analysis.py` | Detaillierte Analyse | YAML-Empfehlung |

Beide analysieren Rohdaten auf wiederkehrende saisonale Muster und helfen bei der Konfiguration von `date_flags` in der Dataset-YAML.

---

## seasonal_overview.py (KISS-Version)

Zeigt nur Daten – der Nutzer entscheidet selbst, welche Tage als Flag markiert werden.

### Ausgabe

- **Monatliche Übersicht** – Welche Monate liegen über/unter Durchschnitt?
- **Top N Tage** – Die verkaufsstärksten Tage im Jahr
- **Bottom N Tage** – Die verkaufsschwächsten Tage

### Beispielaufruf

```powershell
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.seasonal_overview
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.seasonal_overview

# Mehr Tage anzeigen
python -m src.data.seasonal_overview --top 30
```

### Parameter

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `--top` | 20 | Anzahl der Top/Bottom Tage |

---

## seasonal_analysis.py (Detaillierte Version)

Erweitert die Übersicht um automatische Peak-Erkennung und YAML-Empfehlung.

### Zusätzliche Ausgabe

- **Peak-Tage mit Stufen** – Intensitätsstufen ⭐ und ⭐⭐
- **Perioden-Erkennung** – Zusammenhängende Peak-Zeiträume
- **YAML-Empfehlung** – Copy-Paste-fertige Konfiguration

### Beispielaufruf

```powershell
$env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.seasonal_analysis
$env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.seasonal_analysis

# Angepasste Schwellwerte
python -m src.data.seasonal_analysis --elevated-threshold 10 --peak-threshold 25
```

### Parameter

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `--elevated-threshold` | 15.0 | Schwellwert für Stufe 1 (%) |
| `--peak-threshold` | 30.0 | Schwellwert für Stufe 2 (%) |
| `--min-days` | 2 | Mindestanzahl aufeinanderfolgender Tage |

### Automatische Flag-Benennung

| Zeitraum | Flag-Name |
|----------|-----------|
| Dezember / Januar | `is_newyear` |
| November | `is_thanksgiving` |
| Juni / Juli / August | `is_summer_peak` |
| Sonstige | `is_seasonal_peak` |

---

## Legende

| Symbol | Bedeutung |
|--------|-----------|
| ⭐⭐ | >30% über Durchschnitt (Peak) |
| ⭐ | >15% über Durchschnitt (Erhöht) |
| 📈 | >5% über Durchschnitt |
| 📉 | >10% unter Durchschnitt |

---

## Workflow

1. **Script ausführen** → Daten ansehen
2. **Peaks identifizieren** → z.B. "27.-31. Dez + 1.-2. Jan sind ⭐⭐"
3. **In YAML eintragen**:

```yaml
preprocessing:
  - step: "feature_engineering"
    params:
      date_flags:
        is_newyear:
          - {month: 12, day_start: 27, day_end: 31}
          - {month: 1, day_start: 1, day_end: 2}

tft:
  flag_cols: ["is_newyear"]
```

4. **Preprocessing neu durchführen**:

```powershell
python -m src.pipeline --dataset configs/datasets/booksales.yaml --steps preprocessing,model_dataset,dataset_tft
```

---

## Sonderfälle

### Wöchentliche Daten
Bei wöchentlichen Daten (z.B. Walmart) werden automatisch Kalenderwochen statt Tage analysiert.

### Existierendes IsHoliday
Falls `IsHoliday` bereits in den Rohdaten existiert (wie bei Walmart), wird kein zusätzliches Flag empfohlen – nur `flag_cols: ["IsHoliday"]` in der YAML prüfen.

---

## Ergebnis und Nutzen

- Schnelle Identifikation von Peak-Zeiträumen (Weihnachten, Black Friday, etc.)
- Grundlage für `date_flags` Konfiguration in der Dataset-YAML
- Funktioniert für tägliche und wöchentliche Daten
- KISS-Version für schnelle Übersicht, Detail-Version für automatische Empfehlungen