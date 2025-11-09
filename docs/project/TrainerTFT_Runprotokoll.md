# Trainer_TFT – Lauf erklärt (Schritt für Schritt)
 
**Datum:** 2025-11-08 17:10  
**Script:** `src/modeling/trainer_tft.py`  
**Ausgangslage:** Der Lauf wurde ohne Fehler beendet. Dieses Dokument erklärt jede wichtige Logzeile und was sie praktisch bedeutet.
 
---
 
## 1. Startmeldungen und Vorbereitungen
 
### 1.1 `[trainer_tft] Gefüllte NA in Lags: 2400`
- **Was es heißt:** In den Lag-Features (verzögerte Werte wie `lag_num_sold_7`) gab es an Serienanfängen Lücken. Die Pipeline füllt diese gruppenweise auf (erst Vorwärts-/Rückwärtsfüllen, dann ggf. 0).  
- **Warum normal:** Für den allerersten Tag existiert „gestern“ nicht – Lags starten daher naturgemäß mit NAs.
- **Folge:** Das Modell bekommt vollständige numerische Eingaben und kann trainieren.
 
### 1.2 Lightning-Hinweise zu `save_hyperparameters`
- **Log:**  
  `Attribute 'loss' is an instance of nn.Module ...`  
  `Attribute 'logging_metrics' is an instance of nn.Module ...`
- **Bedeutung:** Diese Komponenten würden doppelt in den Checkpoints landen. Das ist **harmlos**; man kann sie optional aus `save_hyperparameters(ignore=[...])` ausschließen.
 
### 1.3 Geräteauswahl
- **Log:** `GPU available: False, used: False`, `TPU/HPU ... False`
- **Bedeutung:** Training lief auf **CPU**. Das ist in Ordnung. Mit einer CUDA‑fähigen GPU kann man später beschleunigen (Trainer z. B. mit `accelerator="gpu", devices=1`).
 
### 1.4 Logger-Hinweis (TensorBoard vs. CSV)
- **Log:** Hinweis, dass ohne `tensorboard` der **CSVLogger** genutzt wird.
- **Praktik:** Protokolle liegen als CSV vor. Optional `pip install tensorboard` und dann `tensorboard --logdir lightning_logs`.
 
### 1.5 Batch-Limits
- **Log:**  
  `Trainer(limit_train_batches=1.0)` und `Trainer(limit_val_batches=1.0)`  
- **Bedeutung:** Pro Epoche werden **100 %** der Trainings‑ bzw. Validierungs‑Batches genutzt. Für schnelle Smoke‑Tests könnte man z. B. `0.2` setzen (nur 20 %).
 
---
 
## 2. Modellzusammenfassung verstehen
Lightning listet die Bausteine des Temporal Fusion Transformer (TFT) auf. Die wichtigsten Blöcke in einfachen Worten:
 
| Block | Kurzbeschreibung | Zweck |
|---|---|---|
| `QuantileLoss` | Verlustfunktion auf Quantilen (z. B. P10/50/90) | Liefert Vorhersagebänder statt nur einen Mittelwert; robuster bei Ausreißern. |
| `MultiEmbedding` | Einbettungen für kategoriale Merkmale | Wandelt IDs/Kalenderkategorien in dichte Vektoren um. |
| `prescalers` | Vor‑Skalierung numerischer Variablen | Bringt Größenordnungen auf vergleichbare Skalen. |
| `VariableSelection*` | Auswahl/Gewichtung von Variablen | Filtert und kombiniert relevante Features (statisch, historisch, zukünftig). |
| `GatedResidualNetwork (GRN)` | Nichtlineare Transformation mit Gates/Skip | Stellt flexible, stabile Transformationen bereit. |
| `LSTM encoder/decoder` | Sequenzmodellierung | Erkennt Muster und Abhängigkeiten über die Zeit. |
| `InterpretableMultiHeadAttention` | Aufmerksamkeitsmechanismus | Hebt wichtige Zeitpunkte in der Sequenz hervor (interpretierbar). |
| `Gate/Add/Norm` | Stabilisierende Schichten | Normalisierung und Gating nach LSTM/Attention. |
| `Linear output_layer` | Projektion auf Zielformat | Übersetzt interne Repräsentation in Quantil‑Outputs. |
 
**Parameterzahlen in deinem Lauf:**  
- Trainierbar: **≈45.1 k**  
- Gesamtgröße: **≈0.18 MB**  
→ Kompaktes Modell, gut für CPU‑Runs und schnelle Iteration.
 
---
 
## 3. Datenpfad, Sanity‑Check und DataLoader‑Workers
 
### 3.1 Sanity‑Check
- **Was passiert:** Vor Epoche 1 wird kurz die Validierung getestet (ein Mini‑Durchlauf), um sicherzustellen, dass Vorwärts‑/Rückwärtsfluss und Logging funktionieren.
 
### 3.2 Hinweis zu „nicht vielen Workern“
- **Log:** Empfehlung, `num_workers` zu erhöhen (z. B. 21).  
- **Praxis unter Windows/CPU:** Realistisch sind **4–8**. Mehr ist nicht immer schneller (Overhead, RAM). Beginne konservativ und steigere nach Bedarf.
- **Bedeutung von „Loader‑Workern“:** Das sind parallele Hintergrundprozesse, die Batches vorbereiten, während das Modell rechnet. Ziel: das Modell wartet nicht auf Daten.
 
---
 
## 4. Trainingsphase – Zeile für Zeile
 
### 4.1 `Epoch 5: ... 425/425 ... train_loss_step=4.480, val_loss=17.00, train_loss_epoch=4.810`
- **Epochenanzahl:** Es wurden **5 Epochen** trainiert.
- **Batches pro Epoche:** **425** Trainings‑Batches (100 % genutzt).
- **`train_loss_step` (~4.48):** Momentaner Loss eines Mini‑Batches – schwankt naturgemäß.
- **`train_loss_epoch=4.810`:** Gemittelter Trainings‑Loss über die ganze Epoche.
- **`val_loss=17.00`:** Validierungs‑Loss auf den Hold‑out‑Daten in derselben Einheit wie der Trainings‑Loss (QuantileLoss).
- **Interpretation:** Absolute Zahlen sind weniger wichtig als die **Tendenz** über Epochen und der **Vergleich** zwischen Konfigurationen (Features, Fensterlängen, Learning‑Rate usw.).
 
### 4.2 Bestes Checkpoint
- **Log:**  
  `[trainer_tft] Bestes Checkpoint: data/processed/model_dataset/tft/checkpoints/tft-00-15.9128.ckpt`
- **Was ist ein Checkpoint:** Binäre Momentaufnahme des Modellzustands (Gewichte, Metadaten).  
- **Wofür gut:** Späteres Laden für Vorhersagen/Evaluation oder Fine‑Tuning.  
- **Namenssuffix (`15.9128`):** Spiegelt i. d. R. den Val‑Loss wider.
 
---
 
## 5. Was du damit praktisch anfangen kannst
 
### 5.1 Vorhersagen erzeugen (Inference)
- **Weg A:** Modell direkt laden und auf einen passenden `DataLoader` anwenden.  
  Beispielaufruf: `python -m src.modeling.load_trained_tft` (lädt automatisch das neueste Checkpoint und schaltet auf Eval‑Modus).
- **Weg B:** Im Notebook/Skript:
  ```python
  from pytorch_forecasting.models import TemporalFusionTransformer
  model = TemporalFusionTransformer.load_from_checkpoint("data/processed/model_dataset/tft/checkpoints/tft-00-15.9128.ckpt")
  model.eval()
  # predictions = model.predict(dataloader)
  ```
 
### 5.2 Qualität bewerten (Evaluation)
- Vergleiche Vorhersagen mit echten Zielwerten (z. B. RMSE/MAE).  
- Betrachte zusätzlich Quantil‑Bänder (z. B. P10/50/90), um Unsicherheit zu kommunizieren.
 
### 5.3 Performance optimieren
- **DataLoader:** `num_workers` schrittweise erhöhen (4 → 8) und Laufzeit vergleichen.  
- **Hardware:** Bei verfügbarer GPU auf `accelerator="gpu", devices=1` umstellen.  
- **Features:** Kalender/Feiertage/Preis/Promo‑Signale gezielt erweitern.
 
---
 
## 6. Konfigurationswerte und deren Rolle
 
Die Hyperparameter kommen aus `src/config.py` → `TRAINER_TFT`. Wichtige Felder:
- `batch_size`: Größe eines Mini‑Batches.
- `learning_rate`: Schrittweite beim Gewichts‑Update.
- `max_epochs`: Maximalzahl an Trainingsrunden.
- `gradient_clip_val`: Begrenzt Gradienten, stabilisiert Training.
- `early_stopping_patience`: Stoppt, wenn sich `val_loss` über X Epochen nicht verbessert.
- `limit_train_batches` / `limit_val_batches`: Anteil bzw. Anzahl der Batches pro Epoche (z. B. 1.0 = 100 %, 0.2 = 20 %, oder ein Integer).
- `num_workers`: Anzahl der DataLoader‑Hintergrundprozesse.
- Modellsektion (`model`): Größen und Dropout‑Raten für TFT‑Bausteine.
 
**Warum die Limits in Config *und* im Trainer auftauchen:**  
- In `config.py` werden die Werte **zentralseitig festgelegt**.  
- In `trainer_tft.py` werden sie **ausgelesen und an `pl.Trainer(...)` übergeben**.  
Das ist keine Redundanz, sondern eine saubere Trennung: **Konfiguration steuert, Code führt aus**.
 
---
 
## 7. Troubleshooting‑Hinweise
- **Checkpoint als Text geöffnet:** `.ckpt` ist binär. Mit Texteditor nicht lesbar.  
  Richtig laden mit `TemporalFusionTransformer.load_from_checkpoint(...)` oder für reinen Blick in die Inhalte mit `torch.load(..., weights_only=False)`.
- **PyTorch 2.6 Pickling‑Fehler:** Falls `torch.load` meckert, `weights_only=False` setzen oder fehlende Klassen via `torch.serialization.add_safe_globals([...])` allow‑listen.
- **Langsame Datenzufuhr:** `num_workers` moderat erhöhen; bei Windows auf RAM‑Verbrauch achten.
 
---
 
## 8. Quintessenz
- Der Lauf war erfolgreich; das Modell wurde gespeichert und kann sofort für Vorhersagen genutzt werden.  
- Die Logwerte belegen: Datenpfad ok, Architektur instanziiert, Training durchgelaufen, bestes Checkpoint vorhanden.  
- Nächste Schritte: `predict()`, Evaluationsplots, gezieltes Feintuning (Features, `num_workers`, ggf. GPU).
