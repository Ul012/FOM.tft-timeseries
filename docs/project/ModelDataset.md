# ModelDataset – Erstellung eines modellfertigen Datensatzes

## Zweck
Das Modul `model_dataset.py` erstellt aus der durch `features.py` vorbereiteten Tabelle ein modellfertiges Datenset.  
Die Daten werden dabei **zeitlich sortiert** und in drei Abschnitte unterteilt: **Training**, **Validation** und **Test**.  
Dieser Schritt dient ausschließlich der strukturierten Datenaufteilung – ohne Modellabhängigkeiten oder Training.

## Eingaben und Ausgaben
- **Eingabe:**  
  `data/processed/features_train.parquet` (oder eine vergleichbare, aufbereitete Datei)
- **Ausgabe:**  
  - `data/processed/model_dataset/train.parquet`  
  - `data/processed/model_dataset/val.parquet`  
  - `data/processed/model_dataset/test.parquet`  
  - `data/processed/model_dataset/manifest.json` (enthält Spaltennamen, Zeitgrenzen, Zeilenzahlen)

## Konfiguration (aus `config.py`)
- `DATA_PROCESSED_PATH`: Pfad zur verarbeiteten Eingabedatei  
- `DATASETS_DIR`: Zielordner für die Teilmengen  
- `TIME_COL`: Zeitspalte (z. B. `date`)  
- `ID_COLS`: Gruppenspalten, z. B. `["country", "store", "product"]`  
- `TARGET_COL`: Zielvariable, z. B. `num_sold`  
- Optionale Parameter:  
  - `VAL_START`, `TEST_START`: feste Datumsgrenzen für Validation/Test  
  - `SPLIT_RATIOS`: Verhältnis bei automatischem Split (z. B. `(0.8, 0.1, 0.1)`)  
  - `SCALE_COLS`: Spalten, die gruppenweise z-standardisiert werden sollen  

## Ablauf
1. **Einlesen** der verarbeiteten Datei (`CSV` oder `Parquet`).
2. **Sortierung** nach Zeit (`TIME_COL`) und Gruppen-IDs (`ID_COLS`).
3. **Bestimmung der Split-Grenzen**  
   - Entweder über feste Datumswerte (`VAL_START`, `TEST_START`)  
   - Oder automatisch nach Verhältnis (`SPLIT_RATIOS`).
4. **Aufteilung der Daten** in Train-, Validation- und Test-Abschnitte entlang der Zeitachse.  
   Dabei gilt: ältere Daten → Training, jüngere Daten → Test.
5. **(Optional)** Gruppenspezifische Z-Standardisierung für angegebene Spalten.
6. **Speichern** der drei Teilmengen und des begleitenden Manifests mit Metadaten.

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
- Kontrolle der Grenzen in der Konsolenausgabe oder in `manifest.json`:
  - Stimmen die Zeiträume mit der Erwartung überein?
  - Sind alle drei Teilmengen befüllt?
- Stichprobenprüfung per:
  ```python
  import pandas as pd
  pd.read_parquet("data/processed/model_dataset/train.parquet").head()
  ```

## Weiterführender Schritt
Das Ergebnis dieses Moduls wird von `dataset_tft.py` oder anderen Trainingsmodulen weiterverarbeitet,  
um daraus framework-spezifische Datenobjekte (z. B. für PyTorch Forecasting) zu erzeugen.
