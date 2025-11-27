# plot_tft_interpretation.py

## Zweck

Visualisiert die Feature Importance eines trainierten TFT-Modells. Zeigt, welche Features wie stark zur Vorhersage beitragen.

---

## Input

- Trainierter TFT-Checkpoint (via `--run-id` oder `--checkpoint`)
- Split: `val` oder `test`
- DATASET_CONFIG Umgebungsvariable

## Output

| Datei | Beschreibung |
|-------|--------------|
| `{run_id}_variable_importance.png` | Encoder & Decoder Importance |
| `{run_id}_static_importance.png` | Static Variables (z.B. Store, Dept) |
| `{run_id}_attention_heatmap.png` | Attention über vergangene Zeitpunkte |
| `{run_id}_attention_summary.png` | Aggregierte Attention |
| `{run_id}_interpretation.json` | Alle Werte als JSON |

Speicherort: `results/tft/plots/interpretation/`

---

## Encoder vs Decoder

| Aspekt | Encoder | Decoder |
|--------|---------|---------|
| Zeitraum | Vergangenheit (Historie) | Zukunft (Vorhersagezeitraum) |
| Frage | Welche historischen Features helfen? | Welche bekannten Zukunfts-Features helfen? |
| Beispiele | Lags, vergangene Zielwerte, Rolling-Means | Kalender, Feiertage, Wochentag |

---

## Interpretation

### Encoder Variables

Zeigt die Importance von Features aus der Vergangenheit:

- Zielwert-Historie (z.B. `num_sold`, `Weekly_Sales`)
- Lag-Features (z.B. `lag_1`, `lag_7`)
- Rolling-Statistiken
- Vergangene Kalender-Features

### Decoder Variables

Zeigt die Importance von bekannten Zukunfts-Features:

- Kalender-Features (Wochentag, Monat, Kalenderwoche)
- Feiertags-Flags
- Zyklische Encodings

### Static Variables

Zeigt die Importance von Gruppen-IDs (falls vorhanden):

- Store, Department, Produkt-ID etc.

---

## Hinweise zur Interpretation

- Die Importance-Werte sind relativ zueinander zu interpretieren
- Nur das Ranking innerhalb eines Modells ist aussagekräftig
- Absolute Werte sind nicht über verschiedene Modelle vergleichbar

---

## Attention Heatmap

Zeigt, welche vergangenen Zeitpunkte für welchen Vorhersage-Zeitpunkt wichtig sind:

- X-Achse: Encoder-Position (vergangene Zeitschritte)
- Y-Achse: Decoder-Position (Vorhersagehorizont)
- Farbe: Attention-Gewicht (dunkel = höher)