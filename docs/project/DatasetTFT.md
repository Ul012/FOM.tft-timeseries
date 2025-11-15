# DatasetTFT -- Erstellung der TFT-Datensatzspezifikation

**Datum:** 2025-11-15\
**Script:** `src/modeling/dataset_tft.py`\
**Ziel & Inhalt:** Beschreibt, wie aus den vorbereiteten
Train/Val/Test-Dateien eine konsistente Datensatzspezifikation für den
Temporal Fusion Transformer (TFT) abgeleitet wird. Erläutert Heuristiken
zur automatischen Feature-Einteilung, die verwendeten
Konfigurationsparameter und die erzeugte `dataset_spec.json`.

------------------------------------------------------------------------

## Zweck

`dataset_tft.py` erstellt **keine neuen Daten**, sondern eine
**Spezifikation** darüber, wie der TFT die vorhandenen Spalten
interpretieren soll.

Konkret:

-   Train/Val/Test werden aus `PROCESSED_DIR` geladen,
-   Feature-Listen (z. B. `static_categoricals`,
    `time_varying_known_reals`) werden heuristisch abgeleitet,
-   Sequenzlängen (Encoder/Decoder) werden aus der Konfiguration
    übernommen,
-   alles wird in einer zentralen Datei `dataset_spec.json` gespeichert.

Diese Spezifikation ist der verbindliche Input für `trainer_tft.py`.

------------------------------------------------------------------------

## Eingaben und Ausgaben

### Eingaben

-   **Verzeichnis mit Splits:**\
    `PROCESSED_DIR` aus `config.py`, üblich:
    -   `data/processed/train.parquet`\
    -   `data/processed/val.parquet`\
    -   `data/processed/test.parquet`
-   **Spalten- und Dataset-Konfiguration:**
    -   `TIME_COL`, `ID_COLS`, `TARGET_COL`\
    -   `TFT_DATASET` (enthält u. a. Prefix-Konfigurationen und
        Sequenzlängen)

### Ausgaben

-   **`data/processed/dataset_spec.json`** (genauer:
    `PROCESSED_DIR / "dataset_spec.json"`)

Inhalt von `dataset_spec.json` (vereinfacht):

-   `time_col`, `id_cols`, `target_col`\
-   Pfade zu `train/val/test.parquet`\
-   `feature_lists`:
    -   `static_categoricals`\
    -   `time_varying_known_reals`\
    -   `time_varying_unknown_reals`\
    -   `time_varying_known_categoricals` (bewusst leer)\
-   `lengths`:
    -   `max_encoder_length`\
    -   `max_prediction_length`\
-   `notes`:
    -   verwendete Prefix-Heuristiken\
    -   Flags und Kalenderbehandlung

------------------------------------------------------------------------

## Konfiguration und Heuristiken

Die Logik basiert auf einer Kombination aus **Konfigurationswerten**
(`TFT_DATASET`) und **Spaltenmustern** im Trainingsdatensatz.

### 1. Zentrale Konstanten

Aus `src/config.py`:

-   `TIME_COL` -- Zeitspalte\

-   `ID_COLS` -- Identität der Zeitreihen (z. B. `country`, `store`,
    `product`)\

-   `TARGET_COL` -- Zielvariable (z. B. `num_sold`)\

-   `TFT_DATASET` -- Dict mit TFT-spezifischen Einstellungen, u. a.:

    -   `known_real_prefixes` -- z. B. `["cyc_"]` für zyklische
        Features\
    -   `lag_prefixes` -- z. B. `["lag_"]` für Lag-Features\
    -   `treat_calendar_as_known` -- Bool, ob Kalender-Features als
        „known" gelten\
    -   `flag_cols` -- explizite Flag-Spalten (z. B.
        `is_lockdown_period`)\
    -   `max_encoder_length`, `max_prediction_length` -- Sequenzlängen

### 2. Kalender- und Feiertagsfeatures

Im Script sind zusätzlich Heuristiken hinterlegt:

-   **Kalenderspalten:**\
    `CALENDAR_COLS = {"year", "month", "day", "dayofweek", "weekofyear", "is_weekend"}`

-   **Feiertagspräfixe:**\
    `HOLIDAY_PREFIXES = ("is_holiday",)`\
    (z. B. `is_holiday_de` oder andere `is_holiday_*`-Spalten)

------------------------------------------------------------------------

## Ableitung der Feature-Listen

Die Methode `TFTDatasetSpecBuilder.run()` führt die Kernlogik aus.

### 1. Basic Checks

-   Prüft, ob `TIME_COL`, alle `ID_COLS` und `TARGET_COL` im
    Trainingsdatensatz vorhanden sind.
-   Bricht mit Fehlermeldung ab, falls eine erwartete Spalte fehlt.

### 2. `static_categoricals`

-   Enthält alle ID-Spalten aus `ID_COLS`, die tatsächlich im Datensatz
    vorhanden sind.
-   Typisch: `["country", "store", "product"]`.

### 3. `time_varying_known_reals`

Aufbau in mehreren Schritten:

1.  **Zyklische Features**
    -   Alle numerischen Spalten, deren Name mit einem der
        `known_real_prefixes` beginnt\
        (z. B. `cyc_dow_sin`, `cyc_dow_cos`, `cyc_month_sin`, ...).
2.  **Kalenderfeatures (optional)**
    -   Nur, wenn `treat_calendar_as_known = True`.\
    -   Fügt numerische Spalten aus `CALENDAR_COLS` hinzu, sofern
        vorhanden\
        (z. B. `year`, `month`, `dayofweek`, `weekofyear`,
        `is_weekend`).
3.  **Feiertags-Features**
    -   Alle numerischen Spalten, deren Name mit einem Prefix aus
        `HOLIDAY_PREFIXES` beginnt\
        (z. B. `is_holiday_de`).
4.  **Explizite Flags**
    -   Alle spalten aus `flag_cols` (z. B. `is_lockdown_period`), falls
        sie existieren\
    -   dürfen numerisch oder bool sein.
5.  **`time_idx` (falls vorhanden)**
    -   Wird als weiteres known real aufgenommen, wenn numerisch
        vorhanden.

Am Ende werden Duplikate entfernt, die Reihenfolge der ersten Vorkommen
bleibt erhalten.

### 4. `time_varying_unknown_reals`

Ausgangspunkt ist die Menge aller numerischen Spalten (inkl. bool):

1.  **Target hinzufügen**
    -   `TARGET_COL` (z. B. `num_sold`) wird als erstes Unknown Real
        aufgenommen.
2.  **Lag-Spalten identifizieren**
    -   Alle numerischen Spalten, deren Name mit einem Prefix aus
        `lag_prefixes` beginnt\
        (z. B. `lag_1`, `lag_7`, `lag_14`, `lag_7_mean`).
3.  **Weitere numerische Spalten**
    -   Alle numerischen Spalten, die **nicht**:
        -   Target sind,
        -   in `known_reals` enthalten sind,
        -   zu den ID-Spalten gehören,
        -   als Lag-Spalten markiert sind.
4.  **Lags ans Ende**
    -   Die Lag-Spalten werden gesammelt und **am Ende** der Liste
        angehängt.\
        Das hat lediglich dokumentarischen Charakter (bessere
        Lesbarkeit).

### 5. `time_varying_known_categoricals`

-   Wird derzeit bewusst als **leere Liste** gespeichert.\
-   Platzhalter für zukünftige Erweiterungen (z. B. kategoriale
    Zeitmerkmale).

------------------------------------------------------------------------

## Sequenzlängen

Aus `TFT_DATASET` werden übernommen:

-   `max_encoder_length` -- Länge des historischen Fensters\
-   `max_prediction_length` -- Länge der Prognoseperiode

Beide Werte werden in `spec["lengths"]` gespeichert und später im
Trainer verwendet.

------------------------------------------------------------------------

## Ablauf (End-to-End)

1.  **Aufruf des Scripts**

    ``` bash
    python -m src.modeling.dataset_tft
    ```

2.  **Erstellen des Builders**

    ``` python
    builder = TFTDatasetSpecBuilder(
        datasets_dir=PROCESSED_DIR,
        time_col=TIME_COL,
        id_cols=list(ID_COLS),
        target_col=TARGET_COL,
        tft_cfg=TFT_DATASET,
    )
    ```

3.  **Ausführen von `run()`**

    -   lädt `train/val/test.parquet`,
    -   führt Basic Checks durch,
    -   leitet Feature-Listen ab,
    -   liest Sequenzlängen aus `TFT_DATASET`,
    -   schreibt `dataset_spec.json`,
    -   gibt eine kurze Konsolenübersicht aus.

------------------------------------------------------------------------

## Einordnung in die Pipeline

Übliche Pipeline:

1.  `data_alignment.py` (optional)\
2.  `data_cleaning.py` (optional)\
3.  `feature_engineering.py`\
4.  `cyclical_encoder.py`\
5.  `lag_features.py`\
6.  `model_dataset.py`\
7.  **dataset_tft.py**\
8.  `trainer_tft.py`

------------------------------------------------------------------------
