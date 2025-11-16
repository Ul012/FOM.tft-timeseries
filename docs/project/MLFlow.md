# MLflow – Erste Konzeptskizze für das TFT-TimeSeries-Projekt
Datum: 2025-11-16  
Ziel & Inhalt:  
Dieses Dokument skizziert eine erste Idee für den möglichen Einsatz von **MLflow** im TFT-TimeSeries-Projekt.  
Es handelt sich **noch nicht um einen konkreten Umsetzungsplan**, sondern um eine Orientierung, welche Funktionen MLflow bieten kann und wie es perspektivisch in die Pipeline integriert werden könnte.

---

## 1. Zweck von MLflow im Projekt

MLflow eignet sich besonders für Projekte, die:

- mehrere Modellvarianten evaluieren,
- Hyperparameter-Optimierung durchführen,
- Modelle versionieren,
- oder ihre Ergebnisse systematisch vergleichen möchten.

Für das TFT-TimeSeries-Projekt bietet MLflow daher einen klaren Mehrwert als **Experiment-Tracking-Tool**.

---

## 2. Tracking von Hyperparametern und Metriken

MLflow ermöglicht das automatische Loggen von:

- Learning Rate  
- Dropout  
- Hidden Size  
- Anzahl Layer  
- Loss-Verlauf  
- MAE, RMSE, sMAPE  
- Trainingsdauer & Ressourcen  

Dadurch entsteht eine **saubere, durchsuchbare Versionierung aller Trainingsläufe**, zentral sichtbar in der MLflow-Weboberfläche.

---

## 3. Vergleich zwischen Modellvarianten

MLflow kann genutzt werden, um verschiedene Modelle systematisch zu vergleichen, z. B.:

- TFT (verschiedene Encoder-Längen)  
- TFT 15-Minuten vs. TFT 60-Minuten  
- LSTM  
- N-BEATS  
- Prophet  
- ARIMA  

MLflow speichert sämtliche Runs und macht Unterschiede unmittelbar sichtbar.

---

## 4. Speichern und Laden von Modellen

MLflow bietet eingebaute Funktionen zum:

- Speichern des besten Modells  
- Wiederladen eines Modells für spätere Nutzung  
- optional: Deployment per API (zu einem späteren Zeitpunkt relevant)

Damit kann später ein vollständiger Modell-Lifecycle abgebildet werden.

---

## 5. Integration mit Optuna

MLflow und Optuna besitzen eine gut dokumentierte, stabile Integration.

Mögliche Kombination:

1. Optuna führt Hyperparameter-Suche durch  
2. MLflow loggt:
   - jedes Trial
   - alle Parameter
   - jede Metrik
3. Ergebnisse und beste Modelle bleiben langfristig nachvollziehbar

Dies gilt als **Best Practice** in vielen HPO-Workflows.

---

## 6. Fazit & Ausblick

Der Einsatz von MLflow bietet im Projekt:

- experimentelles Tracking,
- Vergleichbarkeit,
- Modellversionierung,
- zukünftige Einsatzmöglichkeiten (Deployment, APIs).

Dies ist eine **erste konzeptionelle Idee**, die Struktur und mögliche Vorteile aufzeigt.  
Ein späteres Konzeptpapier definiert die konkrete technische Umsetzung innerhalb der Pipeline.