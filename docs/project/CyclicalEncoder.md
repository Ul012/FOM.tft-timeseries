# CyclicalEncoder – Zweck und Funktionsweise

**Datum:** 2025-11-17  
**Script:** `src/data/cyclical_encoder.py`  
**Ziel & Inhalt:** Beschreibung der Sinus-/Kosinus-Kodierung zyklischer Zeitmerkmale. Erläutert Prinzip, Konfiguration und erzeugte Features.

---

## Überblick
Der CyclicalEncoder wandelt zyklische Zeitmerkmale wie Wochentag oder Monat in Sinus- und Kosinuswerte um. Dadurch wird die kreisförmige Struktur zeitlicher Merkmale korrekt repräsentiert, ohne künstliche numerische Sprünge an Periodenübergängen.

---

## Ziel
Ziel der Kodierung ist es, zyklische Muster maschinenlesbar abzubilden.  
Die Sin/Cos-Repräsentation stellt sicher, dass:
- Periodenanfang und -ende nahtlos verbunden sind  
- relative Abstände korrekt erhalten bleiben  
- Modelle zyklische Nähe erkennen können (z. B. Sonntag und Montag)

---

## Mathematisches Prinzip
Für ein zyklisches Merkmal mit Wert \(x\) und Periode \(P\) wird ein Winkel berechnet:

\[
\theta = 2\pi \times \frac{x}{P}
\]

Daraus entstehen zwei Features:
- `sin(θ)`
- `cos(θ)`

---

## Verwendete Merkmale
Der Encoder erzeugt für folgende Merkmale je zwei Spalten (`*_sin`, `*_cos`):

- `dow` – Wochentag (7)  
- `month` – Monat (12)  
- `doy` – Tag des Jahres (366)  
- `week` – Kalenderwoche (53)  
- `hour` – Stunde des Tages (24)

---

## Konfiguration

```python
class CyclicalEncoderConfig:
    datetime_col = "date"
    periodicities = {
        "dow": ("dow", 7),
        "month": ("month", 12),
        "doy": ("doy", 366),
        "week": ("week", 53),
        "hour": ("hour", 24),
    }
    prefix = "cyc"
    drop_source_cols = True
```

---

## Beispiel

| Wochentag | Zahl | sin(x) | cos(x) |
|---------|------|---------|---------|
| Montag | 0 | 0.000 | 1.000 |
| Dienstag | 1 | 0.781 | 0.624 |
| Sonntag | 6 | −0.781 | 0.624 |

Montag (0) und Sonntag (6) liegen trotz des Zahlenabstands wieder nahe beieinander.

---

## Beispielaufruf

```python
from src.data.cyclical_encoder import CyclicalEncoder, CyclicalEncoderConfig

enc = CyclicalEncoder(CyclicalEncoderConfig())
df_cyc = enc.fit_transform(df)
```

Ausgabe: `data/processed/train_features_cyc.parquet`

---

## Ergebnis und Nutzen
- Abbildung zyklischer Zeitstrukturen ohne Sprünge  
- kontinuierliche Sin/Cos-Darstellung  
- zwei Feature-Spalten pro Merkmal  
- verbesserte Modellierung periodischer Muster
