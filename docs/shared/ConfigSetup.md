# Konfigurationen (`configs/`)

Der Ordner `configs/` dient zur Ablage von **experimentellen YAML-Dateien**, in denen Trainings- und Modellparameter festgelegt werden.  
Aktuell befindet sich dort eine **Baseline-Konfiguration**; weitere Varianten werden im Projektverlauf ergänzt, um verschiedene Hyperparameter-Kombinationen zu testen.

---

## Struktur und Zweck

- **`src/config.py`**  
  Enthält **statische Projektkonstanten**, die sich selten ändern. Dazu zählen Dateipfade, Spaltennamen, Zeitfensterdefinitionen und grundlegende TFT-Parameter wie Encoder- und Prediction-Länge.  
  Diese Datei stellt sicher, dass alle Module auf dieselben Basiseinstellungen zugreifen.

- **`configs/*.yaml`**  
  Enthalten **experimentelle Laufzeitparameter** wie Epochenzahl, Batch-Größe, Lernrate und Modellarchitektur.  
  Jede YAML steht für eine eigene Konfiguration oder ein Experiment und kann unabhängig angepasst oder versioniert werden.

---

## Verwendung im Training

Der **Trainer** lädt zur Laufzeit eine YAML-Datei aus dem Ordner `configs/` und verwendet deren Inhalte zur Initialisierung des Trainingsprozesses.  
Dabei kombiniert er die Werte aus `config.py` (statische Konstanten) mit den spezifischen Parametern aus der YAML-Datei (experimentelle Einstellungen).

Dieses Zusammenspiel erlaubt es, stabile Projektstrukturen beizubehalten und gleichzeitig flexibel verschiedene Modell- und Trainingsvarianten zu testen.
