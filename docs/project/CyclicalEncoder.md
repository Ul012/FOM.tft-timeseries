# CyclicalEncoder – Zweck und Funktionsweise

## Überblick

Der CyclicalEncoder wandelt zyklische Zeitmerkmale (z. B. Wochentag, Monat, Tag im Jahr) in Sinus- und Kosinuswerte um. Dadurch kann ein Modell wie der Temporal Fusion Transformer (TFT) die kreisförmige Struktur zeitlicher Merkmale korrekt erfassen.

---

## Zweck der Sin/Cos-Transformation

Viele Zeitmerkmale sind zyklisch:  
- Nach Sonntag folgt wieder Montag  
- Nach Dezember beginnt erneut Januar  
- Nach Stunde 23 folgt Stunde 0  

Die numerische Darstellung (0–6, 0–11, 0–23…) erzeugt jedoch künstliche Sprünge.  
Die Sin/Cos-Kodierung projiziert Werte auf einen Kreis und stellt die zyklische Nähe korrekt dar.

---

## Mathematisches Prinzip

Ein zyklisches Merkmal wird in einen Winkel transformiert:

θ = 2π × (x / Periode)

Daraus entstehen zwei Features:
- sin(θ)
- cos(θ)

Diese Abbildung:
- ist kontinuierlich  
- verbindet Periodenanfang und -ende  
- wahrt die relativen Abstände  

---

## Verwendete zyklische Merkmale im Projekt

Die Standard-Konfiguration des Encoders erzeugt folgende Merkmale:

- **dow** – Wochentag (7)
- **month** – Monat (12)
- **doy** – Tag des Jahres (366, inkl. Schaltjahr)
- **week** – ISO-Kalenderwoche (53)
- **hour** – Stunde des Tages (24)

Für jedes Merkmal entstehen zwei Spalten:  
`cyc_<name>_sin` und `cyc_<name>_cos`.

---

## Konfiguration

Die Konfigurationswerte sind fest in der Klasse `CyclicalEncoderConfig` definiert:

```python
@dataclass(frozen=True)
class CyclicalEncoderConfig:
    datetime_col: str = "date"
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

Eine globale `CYCLICAL_CONF` in der Projekt-Config wird aktuell **nicht** verwendet.

---

## Beispiel

| Wochentag | Zahl | sin(x) | cos(x) |
|---------|------|---------|---------|
| Montag | 0 | 0.000 | 1.000 |
| Dienstag | 1 | 0.781 | 0.624 |
| Sonntag | 6 | −0.781 | 0.624 |

Montag (0) und Sonntag (6) liegen trotz des Zahlenabstands wieder nahe beieinander.

---

## Verwendung im Projektkontext

Der Encoder folgt im Pipeline-Ablauf direkt auf  
**feature_engineering.py** und erzeugt die Datei:

```
data/processed/train_features_cyc.parquet
```

Diese dient anschließend als Input für **lag_features.py**.

---

## Beispielaufruf

```python
from src.data.cyclical_encoder import CyclicalEncoder, CyclicalEncoderConfig

enc = CyclicalEncoder(CyclicalEncoderConfig())
df_cyc = enc.fit_transform(df)
```

---

## Hinweise & Fallstricke

- Zeitzonen werden korrekt berücksichtigt (Default: Europe/Berlin)  
- NaT-Werte bleiben bewusst NaN  
- `drop_source_cols=True` entfernt Hilfsspalten wie `cyc_dow_idx`  
- Für jedes Merkmal entstehen zwei Features (sin/cos)

---

## Zusammenfassung

| Aspekt | Beschreibung |
|--------|-------------|
| Zweck | Abbildung zyklischer Zeitmerkmale |
| Methode | Sinus- und Kosinus-Transformation |
| Vorteile | Keine Sprünge, korrekte Distanzwahrung |
| Eingesetzte Merkmale | dow, month, doy, week, hour |
| Ausgabe | Zwei Features pro Merkmal |
| Position in Pipeline | Nach FeatureEngineering, vor LagFeatures |
