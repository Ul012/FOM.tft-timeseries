# ModelDataset – Erstellung eines modellfertigen Datensatzes

## Zweck
Das Modul `model_dataset.py` erstellt aus der durch die Feature-Pipeline vorbereiteten Tabelle ein modellfertiges Datenset.  
Die Daten werden dabei **zeitlich sortiert** und in drei Abschnitte unterteilt: **Training**, **Validation** und **Test**.  
Dieser Schritt dient ausschließlich der strukturierten Datenaufteilung – ohne Modellabhängigkeiten oder Training.

## Eingaben und Ausgaben
- **Eingabe:**  
  - `data/processed/train_features_cyc_lag.parquet` (entspricht `MODEL_INPUT_PATH` aus `config.py`)
- **Ausgabe:**  
  - `data/processed/train.parquet`  
  - `data/processed/val.parquet`  
  - `data/processed/test.parquet`  
  - `data/processed/meta.json` (enthält Spaltennamen, Zeitgrenzen, Zeilenzahlen und Basis-Metadaten)

## Konfiguration (aus `config.py`)
- Pflicht:
  - `MODEL_INPUT_PATH`: Pfad zur verarbeiteten Eingabedatei (Feature-Tabelle mit Lags/Zyklen)
  - `PROCESSED_DIR`: Zielordner für die Teilmengen (`train/val/test.parquet`)
  - `TIME_COL`: Zeitspalte (z. B. `date`)
  - `ID_COLS`: Gruppenspalten, z. B. `["country", "store", "product"]`
  - `TARGET_COL`: Zielvariable, z. B. `num_sold`
- Optionale Parameter:
  - `VAL_START`, `TEST_START`: feste Datumsgrenzen für Validation/Test  
  - `SPLIT_RATIOS`: Verhältnis bei automatischem Split (z. B. `(0.8, 0.1, 0.1)`), wird genutzt, wenn `VAL_START`/`TEST_START` nicht gesetzt sind  
  - `SCALE_COLS`: Spalten, die gruppenweise z-standardisiert werden sollen (Fit nur auf Train, Anwendung auf Val/Test)

## Ablauf
1. **Einlesen** der verarbeiteten Datei (`MODEL_INPUT_PATH`), unterstützt werden Parquet und CSV.
2. **Sortierung** nach Zeit (`TIME_COL`) und Gruppen-IDs (`ID_COLS`).
3. **Bestimmung der Split-Grenzen**:
   - Entweder über feste Datumswerte (`VAL_START`, `TEST_START`), oder
   - automatisch nach Verhältnis (`SPLIT_RATIOS`) über die gesamte Zeitachse.
4. **Aufteilung der Daten** in Train-, Validation- und Test-Abschnitte entlang der Zeitachse.  
   Dabei gilt: ältere Daten → Training, jüngere Daten → Test.
5. **(Optional)** gruppenspezifische Z-Standardisierung für Spalten in `SCALE_COLS` (Fit auf Train, Anwendung auf Val/Test je Gruppe).
6. **Speichern** der drei Teilmengen (`train/val/test.parquet`) und des begleitenden Metafiles `meta.json` mit:
   - verwendeten Spaltennamen (`time_col`, `id_cols`, `target_col`)
   - den effektiven Datumsgrenzen für Validation/Test
   - Zeilenzahlen pro Split
   - Pfadangaben für Ein- und Ausgabe
   - ggf. Liste der skalierten Spalten

## Bedeutung des Splits
Der Split stellt sicher, dass:
- das Modell nur aus der Vergangenheit lernt (Train),
- Hyperparameter anhand eines unabhängigen Zeitraums überprüft werden können (Validation),
- und die finale Bewertung auf völlig neuen Daten erfolgt (Test).  
Dies verhindert Überanpassung und erlaubt eine objektive Leistungsbewertung.

## Annahmen und Grenzen
- Die Zeitspalte muss als `datetime` interpretierbar sein und eine eindeutige zeitliche Reihenfolge besitzen.
- Bei Verwendung von `SPLIT_RATIOS` erfolgt der Split auf globaler Ebene, nicht pro Gruppe.
- Es werden keine Modell- oder Framework-Abhängigkeiten geladen; der Fokus liegt ausschließlich auf reproduzierbarer Datenstrukturierung.

## Überprüfung
- Kontrolle der Konsolenausgabe und der Inhalte von `meta.json`:
  - Stimmen die Zeiträume mit der Erwartung überein?
  - Sind alle drei Teilmengen befüllt?
  - Passen die Pfade und die Zeilenzahlen?
- Stichprobenprüfung per:
  ```python
  import pandas as pd
  pd.read_parquet("data/processed/train.parquet").head()
  ```

## Weiterführender Schritt
Das Ergebnis dieses Moduls wird von `dataset_tft.py` oder anderen Trainingsmodulen weiterverarbeitet,  
um daraus framework-spezifische Datenobjekte (z. B. für PyTorch Forecasting) zu erzeugen.
