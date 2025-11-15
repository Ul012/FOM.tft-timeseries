# Trainer_TFT – Runprotokoll (Schritt für Schritt)

**Datum:** 2025-11-15  
**Script:** `src/modeling/trainer_tft.py`  
**Ziel & Inhalt:** Dieses Dokument erklärt die typischen Logausgaben eines TFT-Trainingslaufs zeilenweise. Es dient dazu, jeden Abschnitt der Konsole technisch zu verstehen, ohne laufabhängige Werte oder konkrete Zahlen zu nennen.

---

## 1. Startmeldungen und Vorbereitungen

### 1.1 `[trainer_tft] Gefüllte NA in Lags: ...`
- Lag-Features erzeugen am Anfang jeder Zeitreihe zwangsläufig fehlende Werte.  
- Diese werden (vorwärts/rückwärts) aufgefüllt, damit der Datensatz vollständig ist.  
- Typisch und erwartbar bei Zeitreihen. Kein Fehler.

### 1.2 Lightning-Hinweise zu `save_hyperparameters`
- Lightning weist darauf hin, dass einige Module doppelt gespeichert würden.  
- Das ist unkritisch; optional kann man diese ignorieren.  
- Wichtig ist nur: Das Training läuft normal weiter.

### 1.3 Geräteauswahl
- Log-Meldungen wie `GPU available: False, used: False`.  
- Zeigen, dass der Lauf vollständig auf CPU erfolgt.  
- Unter Windows ohne CUDA völlig normal.

### 1.4 Logger-Hinweis (CSV statt TensorBoard)
- Lightning nutzt hier automatisch `CSVLogger`.  
- Alle Epochen-Werte erscheinen später in `metrics.csv`.

### 1.5 Batch-Limits
- `limit_train_batches=1.0` und `limit_val_batches=1.0` bedeuten:  
  **100 % aller Batches** pro Epoche werden verwendet.  
- Reduktion wäre nur für Tests sinnvoll.

---

## 2. Modellzusammenfassung verstehen  
Lightning listet die Bausteine des Temporal Fusion Transformer (TFT) im Detail auf.  
Hier eine verständliche Einordnung der wichtigsten Module:

| Block | Kurzbeschreibung | Zweck |
|---|---|---|
| `QuantileLoss` | Verlustfunktion auf Quantilen (z. B. P10/50/90) | Liefert Vorhersagebänder; robuster gegen Ausreißer. |
| `MultiEmbedding` | Einbettungen für kategoriale Features | Wandelt IDs/Kalendermerkmale in dichte Vektoren um. |
| `prescalers` | Vor-Skalierung numerischer Features | Vereinheitlicht Größenordnungen, stabilisiert Training. |
| `VariableSelection*` | Variable Selection Networks | Wählen dynamisch die wichtigsten Features je Zeitschritt. |
| `GatedResidualNetwork (GRN)` | Residual + Gate-Mechanismus | Flexible nichtlineare Transformationen, stabil durch Skip Connections. |
| `LSTM encoder/decoder` | Recurrent-Bausteine | Modellieren zeitliche Abhängigkeiten über Sequenzen. |
| `InterpretableMultiHeadAttention` | Attention-Layer | Lenkt Fokus auf relevante Zeitpunkte (interpretierbar). |
| `Gate/Add/Norm` | Normalisierung + Gating | Stabilisieren und glätten Signale nach LSTM/Attention. |
| `Linear output_layer` | End-Projektion | Formt interne Repräsentation in Quantile für den Forecast um. |

**Gesamtbewertung:**  
Der TFT kombiniert rekurrente Schichten, Variable Selection, Attention und Gating in einer Architektur, die **sowohl leistungsfähig als auch interpretierbar** ist.

---

## 3. Datenpfad, Sanity-Check und DataLoader-Workers

### 3.1 Sanity-Check vor Start der ersten Epoche
- Lightning führt vor Epoche 1 eine Mini-Validierung aus.  
- Dadurch werden Probleme früh erkannt (Shapes, Forward-Pass, Loss).

### 3.2 Hinweis zu DataLoader-Workern
- Log schlägt teilweise extrem hohe Worker-Zahlen vor (z. B. 21).  
- Unter Windows sind **4–8 realistisch**.  
- Mehr Worker beschleunigen die Datenbereitstellung — aber nur, wenn CPU/RAM es zulassen.

---

## 4. Trainingsphase – typische Logzeilen verstehen

### 4.1 Epochenmeldungen
Typische Zeile:

```
Epoch X/Y: ... train_loss_step=..., val_loss=..., train_loss_epoch=...
```

**Bedeutung im Detail:**

- **`Epoch X/Y`**  
  Trainingsfortschritt.

- **`train_loss_step`**  
  Loss eines *einzelnen* Batches. Schwankt stark, völlig normal.

- **`train_loss_epoch`**  
  Durchschnittsloss aller Batches dieser Epoche.

- **`val_loss`**  
  Loss auf den Validierungsdaten – wichtigste Lernfortschrittsmetrik.

- **Interpretation:**  
  Entscheidend ist nicht der absolute Wert, sondern die **Tendenz**  
  & der **Vergleich zwischen Runs**.

---

## 5. Checkpoints

### 5.1 `Saving latest checkpoint`
- Regelmäßig gespeicherter Zwischenstand.

### 5.2 `Saving best checkpoint`
- Speichert das Modell mit dem **niedrigsten Validierungs-Loss bisher**.  
- Wird später für Vorhersagen und Evaluierung verwendet.  
- Lightning vergibt Dateinamen, die den Val-Loss enthalten.

---

## 6. Finale Meldungen nach Abschluss

### 6.1 `Training finished`
- Der Trainingsloop ist abgeschlossen.

### 6.2 `Best model stored at: ...`
- Pfad zum **Best-Checkpoint**.

### 6.3 `Best validation loss: ...`
- Niedrigster Validierungs-Loss des gesamten Trainings.

### 6.4 `Final epoch: ...`
- Tatsächlich letzte Epoche (z. B. durch Early Stopping reduziert).

---

## 7. Erzeugte Dateien

### `metrics.csv`
- Eine Zeile pro Epoche:
  - train_loss  
  - val_loss  
  - Lernrate (falls vorhanden)  
  - Epochennummer  
- Grundlage für Plots & Evaluator.

### `summary.json`
- Kompakte Kennzahlen:
  - final_train_loss  
  - final_val_loss  
  - best_val_loss  
  - best_val_epoch  
  - initial_lr / final_lr (falls geloggt)

### `checkpoints/best.ckpt`
- Binärer Modellzustand.  
- Später nutzbar für:
  - Vorhersagen
  - Evaluierung
  - Reload & Feintuning

### `full_config.yaml`
- Speichert die exakte Konfiguration dieses Runs.

---

## 8. Troubleshooting-Hinweise

- **Checkpoint lässt sich nicht öffnen** → nicht im Texteditor öffnen, sondern via  
  `TemporalFusionTransformer.load_from_checkpoint(...)`.
- **Langsame DataLoader** → moderat `num_workers` erhöhen.
- **Hoher Speicherverbrauch** → kleinere Batchgröße wählen.
- **Unklare Lightning-Warnungen** → meist harmlos; beziehen sich auf Hyperparameter-Speicherung.

---

## 9. Quintessenz

- Das Protokoll zeigt, dass der Lauf sauber durchlief.  
- Alle relevanten Artefakte wurden erstellt.  
- Jeder Logabschnitt ist inhaltlich nachvollziehbar, ohne dass konkrete Zahlen benötigt werden.  
- Dieses Dokument dient als Referenz, um die Konsolenlogs eines beliebigen TFT-Trainingslaufes zeilenweise zu verstehen.

