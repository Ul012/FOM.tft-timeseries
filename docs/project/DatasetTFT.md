# DatasetTFT – Spezifikation für Temporal Fusion Transformer

## Zweck
Das Modul `dataset_tft.py` erzeugt aus den bereits gesplitteten Parquet-Dateien (`train`, `val`, `test`) eine **Datasetspezifikation** für den Temporal Fusion Transformer (TFT). Die Spezifikation beschreibt, welche Spalten als statische Kategorische, zeitvariierende bekannte Reals (z. B. Kalender- und zyklische Features) und zeitvariierende unbekannte Reals (z. B. Zielvariable, Lags) verwendet werden. Der Schritt beinhaltet **kein** Modelltraining und **keine** Bibliotheksabhängigkeit zu PyTorch-Forecasting – es wird nur eine JSON-Datei erzeugt, die ein Trainer später konsumieren kann.

## Eingaben und Ausgaben
- **Eingabe:**  
  - `data/processed/model_dataset/train.parquet`  
  - `data/processed/model_dataset/val.parquet`  
  - `data/processed/model_dataset/test.parquet`
- **Ausgabe:**  
  - `data/processed/model_dataset/tft/dataset_spec.json`

## Konfiguration (aus `config.py`)
- Pflicht:
  - `DATASETS_DIR`: Basisordner des Splits (enthält `train/val/test.parquet`)
  - `TIME_COL`: Zeitspalte (z. B. `date`)
  - `ID_COLS`: Gruppenspalten (z. B. `["country", "store", "product"]`)
  - `TARGET_COL`: Zielvariable (z. B. `num_sold`)
- Optional in `TFT_DATASET` (falls nicht gesetzt, gelten Defaults):
  - `max_encoder_length` (Default: 28)
  - `max_prediction_length` (Default: 7)
  - `known_real_prefixes` (Default: `["cyc_"]` – alle Spalten, die mit `cyc_` beginnen)
  - `lag_prefixes` (Default: `["lag_"]` – alle Spalten, die mit `lag_` beginnen)
  - `treat_calendar_as_known` (Default: `True` – Kalender-/Feiertags-/Flags als known reals)

## Ablauf
1. **Prüfen und Einlesen** der drei Parquet-Dateien aus `DATASETS_DIR`.
2. **Schema-Inspektion** des Trainingssatzes: Ermittlung aller Spalten und der numerischen Spalten.
3. **Heuristische Zuordnung** der Featurelisten:
   - `static_categoricals`: `ID_COLS` (statisch je Serie)
   - `time_varying_known_reals`: Kalender-/zyklische/Flag-Spalten (z. B. `year`, `month`, `dayofweek`, `is_weekend`, `cyc_*`, `is_holiday_*`, `is_lockdown_period`, `time_idx` wenn vorhanden)
   - `time_varying_unknown_reals`: Zielvariable, `lag_*`-Features und übrige numerische Spalten, die nicht bereits als known/IDs klassifiziert wurden
4. **Festlegen der Sequenzlängen** (`max_encoder_length`, `max_prediction_length`) aus `TFT_DATASET` oder per Default.
5. **Speichern** der Spezifikation in `dataset_spec.json`.

## Bedeutung

## Begriffserklärung: known vs. unknown reals

In Temporal-Fusion-Transformer-(TFT)-Modellen werden numerische (reale) Features danach unterschieden, ob sie zum Zeitpunkt der Vorhersage bereits bekannt sind oder nicht. Diese Unterscheidung ist entscheidend, um Datenlecks zu vermeiden und realistische Vorhersagen zu ermöglichen.

| Kategorie | Beschreibung | Beispiele |
|------------|---------------|-----------|
| **Known Reals** | Zeitabhängige numerische Variablen, die auch für zukünftige Zeitpunkte bekannt sind. Das Modell darf sie während der Vorhersage nutzen. | Kalenderinformationen (Tag, Monat, Jahr), geplante Preise, Feiertagsflags, Promotionzeiträume, `cyc_*` Features |
| **Unknown Reals** | Zeitabhängige numerische Variablen, deren zukünftige Werte unbekannt sind. Sie werden nur im Training (Vergangenheit) beobachtet und dienen als Zielvariable oder zur Modellierung der Dynamik. | Absatzmenge (`num_sold`), Temperatur, tatsächliche Nachfrage, Umsatz, `lag_*` Features |

**Beispiel:**  
In einem Buchverkaufs-Forecast sind `date`, `month`, `price` und `is_holiday` bekannte Reals, während `sales_volume` und `lag_sales_volume` unbekannte Reals sind.

Die JSON-Spezifikation (`dataset_spec.json`) spiegelt diese Trennung wider, indem sie Felder wie `time_varying_known_reals` und `time_varying_unknown_reals` definiert. Der Temporal Fusion Transformer verwendet diese Information, um beim Dekodieren nur auf zulässige (bekannte) zukünftige Eingaben zuzugreifen.

Die Spezifikation entkoppelt die **Datenbeschreibung** vom **Training**. Dadurch bleibt die Pipeline schlank, reproduzierbar und verständlich. Trainer-Module können sich darauf verlassen, dass die Spaltenlisten und Sequenzlängen stabil bereitgestellt werden.

## Überprüfung
- Konsole gibt die wichtigsten Listenlängen aus (z. B. Anzahl `known_reals`).
- In `dataset_spec.json` prüfen:
  - Pfade (`train/val/test`) korrekt?
  - `feature_lists` plausibel?
  - Sequenzlängen passend zum Use-Case?

## Weiterführender Schritt
Ein Trainingsmodul (z. B. `trainer_tft.py`) lädt `dataset_spec.json`, öffnet `train/val/test.parquet` und baut daraus die framework-spezifischen Datenobjekte (z. B. `TimeSeriesDataSet` für PyTorch-Forecasting).


### Schematische Darstellung

```text
Zeitachse (T)
│────────────────────────────────────────────────────────▶ Zukunft
│
│ Vergangene Beobachtungen:         Zukünftige Vorhersage:
│
│   ┌──────────────┐                ┌──────────────┐
│   │ unknown real │                │ known real   │
│   │ (z. B. Sales)│                │ (z. B. Price)│
│   └──────────────┘                └──────────────┘
│        ↑                              ↑
│      Trainingdaten               Forecast-Horizont
│
│ Das Modell sieht in der Zukunft nur die bekannten (known) Variablen,
│ während die unbekannten (unknown) Reals vorhergesagt werden.
```
