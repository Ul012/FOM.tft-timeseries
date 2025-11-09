# DataCleaning – Zweck und Funktionsweise

## Überblick
Die Klasse **DataCleaner** bereinigt offensichtliche Ausreißer und ersetzt fehlende Werte in den (bereits auf 2020 skalierten) Verkaufsdaten. Der Ansatz folgt dem Medium‑Artikel (Teil 2): Einzelne Ausreißer (z. B. **01.01.2020**) sowie der Lockdown‑Zeitraum **März–Mai 2020** werden auf *NaN* gesetzt und anschließend mithilfe verschobener („geshifteter“) Zeitreihen durch **den Mittelwert vergleichbarer Vorjahrestage** ersetzt. Ziel ist es, Lücken zu glätten, **ohne** echte saisonale Muster zu verlieren.

**Input:** `data/interim/train_aligned.parquet` (aus DataAlignment)  
**Output:** `data/interim/train_cleaned.parquet`

---

## Ziel
- **Entfernen/Glätten von Ausreißern**, die das Modell verzerren würden.
- **Erhalt von Saisonalität**: Imputation basiert auf denselben Kalendertagen in anderen Jahren.
- **Stabile Trainingsdaten** für den Temporal Fusion Transformer (TFT).

---

## Kernidee der Methode
1. **Markieren**: Bestimmte Zeitpunkte/Zeiträume auf *NaN* setzen (z. B. 01.01.2020; März–Mai 2020).
2. **Imputation durch Vorjahre**: Eine Kopie der Zielspalte `num_sold` wird um ganze Jahres‑ bzw. Wochen‑Perioden **verschoben** (*shift*). Aus mehreren verschobenen Spalten wird **zeilenweise der Mittelwert** gebildet.  
   *Beispiel:* Für 01.01.2020 wird der Wert aus (01.01.2019, 01.01.2018, …) gemittelt.
3. **Auffüllen**: *NaN*-Stellen werden mit diesen Mittelwerten ersetzt. So bleiben Muster (z. B. Feiertagseffekte) erhalten.

---

## Klassen‑API

### `class DataCleaner`
**Konstruktor**
```python
DataCleaner(df: pd.DataFrame)
```
- Erwartet ein DataFrame mit Spalten mindestens `date` (Datum) und `num_sold` (Zielvariable).
- Stellt sicher, dass `date` ein Datums-Typ ist, und setzt `date` als Index (Zeitachse).

**Methoden**
- `handle_single_day_outlier(date_str: str) -> None`  
  Setzt den Wert an einem bestimmten Datum auf *NaN* (Ausreißer wird „entfernt“).
- `handle_lockdown_period(year: int, months: tuple[int, ...]) -> None`  
  Setzt einen Zeitraum (z. B. März–Mai 2020) auf *NaN*.
- `_fill_with_shifted_mean(periods: int, repeats: int = 3) -> None`  
  Erzeugt `repeats` verschobene Kopien von `num_sold` und füllt *NaN* mit dem **Zeilenmittel** dieser Kopien.
- `clean() -> pd.DataFrame`  
  Orchestriert die Schritte:
  1) 01.01.2020 → *NaN* → Auffüllen per shift (Jahres‑Periode)  
  2) März–Mai 2020 → *NaN* → Auffüllen per shift (Wochen‑Periode)  
  Gibt das bereinigte DataFrame mit **zurückgesetztem Index** (Spalte `date`) zurück.

---

## Algorithmik & Parameter

### 1) Einzelausreißer (01.01.2020)
- Setzen auf *NaN*
- Auffüllen mit Mittelwert aus **Jahres‑Shifts**:
  ```python
  periods = 365 * 48
  # 48 ≙ 30‑Minuten‑Slots pro Tag (wie im Medium‑Snippet verwendet).
  # Wenn deine Daten TÄGLICH sind, setze periods z. B. auf 365.
  ```

### 2) Lockdown‑Monate (März–Mai 2020)
- Setzen auf *NaN*
- Auffüllen mit Mittelwert aus **Wochen‑Shifts**:
  ```python
  periods = 52 * 7 * 48  # Wochen * Tage * (z. B. 30‑Minuten‑Slots)
  # Bei täglichen Daten genügt 52 * 7.
  ```

> **Hinweis:** Die Multiplikatoren `* 48` stammen aus dem im Artikel gezeigten Code (feinere Zeitauflösung). Für den Kaggle‑Datensatz (tägliche Frequenz) kannst du ohne Informationsverlust auf `* 1` gehen. Passe bei Bedarf die Konstanten in `_fill_with_shifted_mean(...)` an.

---

## Edge Cases & Robustheit
- **Fehlende Datumswerte**: Der Code setzt `errors="coerce"`, nicht interpretierbare Datumswerte werden *NaT* und fließen nicht in die Imputation ein.
- **Keine Vergleichswerte vorhanden**: Wenn alle verschobenen Spalten an einer Stelle *NaN* sind, bleibt *NaN* bestehen (kommt praktisch selten vor). Optional könnte man dann einen Fallback (z. B. Vorwärts‑/Rückwärtsfüllung) ergänzen.
- **Idempotenz**: Mehrfaches Ausführen ändert das Ergebnis nicht wesentlich (sofern bereits bereinigt).

---

## Nutzung

**Ausführen im Terminal**
```bash
cd C:\Users\ulrik\Projekte\TFT-Booksales
python -m src.data.data_cleaning
```
- Erwartet: `data/interim/train_aligned.parquet`
- Ergebnis: `data/interim/train_cleaned.parquet`

**In Python importieren**
```python
from src.data.data_cleaning import DataCleaner
import pandas as pd

df = pd.read_parquet("data/interim/train_aligned.parquet")
cleaned = DataCleaner(df).clean()
```

---

## Designprinzipien
- **Single Responsibility**: Eine Klasse kümmert sich ausschließlich um Datenbereinigung.
- **Explizit statt magisch**: Betroffene Zeiträume sind benannt und nachvollziehbar.
- **Keine Über‑Abstraktion**: Minimaler, klarer API‑Umfang; leicht zu testen und zu erweitern.
- **Separation of Concerns**: DataAlignment (Skalierung) und DataCleaning (Ausreißer) bleiben getrennt.
