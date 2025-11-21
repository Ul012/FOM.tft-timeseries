# DatasetTFT – Erstellung der TFT-Datensatzspezifikation

**Datum:** 2025-11-15  
**Script:** `src/modeling/dataset_tft.py`  
**Ziel & Inhalt:** Ableitung einer konsistenten Datensatzspezifikation für den Temporal Fusion Transformer (TFT) auf Basis der vorbereiteten Splits.

---

## Überblick
`dataset_tft.py` erzeugt eine zentrale Spezifikation (`dataset_spec.json`), die beschreibt, wie der TFT die Spalten interpretiert. Die Spezifikation legt Feature-Listen, Zeit- und ID-Spalten sowie Sequenzlängen für das Modelltraining fest.
**Wichtig:** Liest/schreibt in `data/processed/<dataset_name>/` - Dataset-Name kommt aus Config.

---

## Ziel
Ziel ist eine eindeutige, reproduzierbare Definition aller Eingabemerkmale.  
Die Spezifikation dient als verbindliche Schnittstelle zwischen den vorbereiteten Daten (`train/val/test.parquet`) und dem Trainingsskript `trainer_tft.py`.

---

## Eingaben & Ausgaben

### Eingaben
- `train.parquet`  
- `val.parquet`  
- `test.parquet`  
- Konfiguration aus `src/config.py`:
  - `TIME_COL`  
  - `ID_COLS`  
  - `TARGET_COL`  
  - `TFT_DATASET`

### Ausgabe
- `dataset_spec.json` im gleichen Verzeichnis wie die Splits

---

## Zentrale Konstanten

Aus `src/config.py`:

- `TIME_COL` – Zeitspalte  
- `ID_COLS` – Identität der Zeitreihen (z. B. `country`, `store`, `product`)  
- `TARGET_COL` – Zielvariable (z. B. `num_sold`)  
- `TFT_DATASET` – Dict mit TFT-spezifischen Einstellungen, u. a.:
  - `known_real_prefixes` – z. B. `["cyc_"]` für zyklische Features  
  - `lag_prefixes` – z. B. `["lag_"]` für Lag-Features  
  - `treat_calendar_as_known` – Bool, ob Kalenderfeatures als „known“ gelten  
  - `flag_cols` – explizite Flag-Spalten (z. B. `is_lockdown_period`)  
  - `max_encoder_length`, `max_prediction_length` – Encoder- und Decoderlängen

Im Script sind zusätzlich Heuristiken für Kalender- und Feiertagsmerkmale hinterlegt:

- **Kalenderspalten:**  
  `CALENDAR_COLS = {"year", "month", "day", "dayofweek", "weekofyear", "is_weekend"}`  
- **Feiertagspräfixe:**  
  `HOLIDAY_PREFIXES = ("is_holiday",)` – z. B. `is_holiday_de` oder andere `is_holiday_*`-Spalten

---

## Vorgehen

### 1. Grundprüfung
Es wird geprüft, ob `TIME_COL`, alle `ID_COLS` und `TARGET_COL` im Trainingsdatensatz vorhanden sind.  
Fehlende Pflichtspalten führen zu einem Abbruch mit Fehlermeldung.

---

### 2. Ableitung der Feature-Listen

Die folgenden Listen werden automatisch aus Spaltennamen, Datentypen und Konfiguration abgeleitet:

- `static_categoricals`  
- `time_varying_known_reals`  
- `time_varying_unknown_reals`  
- `time_varying_known_categoricals` (derzeit leer)

#### 2.1 `static_categoricals`
- enthält alle ID-Spalten aus `ID_COLS`, die im Datensatz vorhanden sind  
- typisches Beispiel: `["country", "store", "product"]`

#### 2.2 `time_varying_known_reals`
Beginnend mit einer leeren Liste werden schrittweise ergänzt:

1. **Zyklische Features**  
   Alle numerischen Spalten, deren Name mit einem der `known_real_prefixes` beginnt  
   (z. B. `cyc_dow_sin`, `cyc_month_cos`).

2. **Kalenderfeatures (optional)**  
   Nur, wenn `treat_calendar_as_known = True`.  
   Es werden numerische Spalten aus `CALENDAR_COLS` aufgenommen, sofern vorhanden  
   (z. B. `year`, `month`, `dayofweek`, `weekofyear`, `is_weekend`).

3. **Feiertagsfeatures**  
   Alle numerischen Spalten, deren Name mit einem Prefix aus `HOLIDAY_PREFIXES` beginnt  
   (z. B. `is_holiday_de`).

4. **Explizite Flags**  
   Alle Spalten aus `flag_cols`, falls sie existieren und numerisch oder boolesch sind  
   (z. B. `is_lockdown_period`).

5. **Zeitindex**  
   Falls ein numerischer `time_idx` vorhanden ist, wird er ebenfalls als known real ergänzt.

Am Ende werden doppelte Einträge entfernt; die Reihenfolge der ersten Vorkommen bleibt erhalten.

#### 2.3 `time_varying_unknown_reals`
Ausgangspunkt ist die Menge aller numerischen (inkl. boolescher) Spalten:

1. **Target**  
   `TARGET_COL` wird als erster Eintrag aufgenommen.

2. **Lag-Spalten**  
   Alle numerischen Spalten, deren Name mit einem Prefix aus `lag_prefixes` beginnt  
   (z. B. `lag_1`, `lag_7`, `lag_14`, `lag_7_mean`) werden gesammelt.

3. **Weitere numerische Spalten**  
   Alle verbliebenen numerischen Spalten, die  
   - nicht Target sind,  
   - nicht in `time_varying_known_reals` enthalten sind,  
   - nicht zu `ID_COLS` gehören,  
   - nicht als Lag-Spalten markiert sind.

4. **Reihenfolge der Lags**  
   Die zuvor gesammelten Lag-Spalten werden am Ende der Liste hinzugefügt.  
   Dies erleichtert die Lesbarkeit von `dataset_spec.json`.

#### 2.4 `time_varying_known_categoricals`
- wird derzeit als leere Liste gespeichert  
- dient als Platzhalter für zukünftige kategoriale Zeitmerkmale

---

## Sequenzlängen
Aus `TFT_DATASET` werden die Fensterlängen übernommen:

- `max_encoder_length` – Länge des historischen Fensters  
- `max_prediction_length` – Länge der Vorhersageperiode

Beide Werte werden in `dataset_spec.json` unter `lengths` abgelegt.

---

## Beispielaufruf
```bash
python -m src.modeling.dataset_tft
```

---

## Ergebnis und Nutzen
- konsistente, maschinenlesbare Beschreibung der TFT-Datenstruktur  
- automatisierte, konfigurationsgestützte Ableitung der Feature-Listen  
- zentrale Spezifikation als Grundlage für das Modelltraining in `trainer_tft.py`  
