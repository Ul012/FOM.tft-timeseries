# CyclicalEncoder

## Überblick

Der CyclicalEncoder wandelt zyklische Zeitmerkmale (z. B. Wochentag, Monat, Stunde) in Sinus- und Kosinuswerte um.  
Dadurch kann ein Modell wie der Temporal Fusion Transformer (TFT) oder ein anderer Regressor die kreisförmige Struktur solcher Merkmale korrekt erfassen.

---

## Zweck der Sin/Cos-Transformation

Ziel ist es, zyklische Beziehungen kontinuierlich und korrekt abzubilden, ohne künstliche Sprünge im Zahlenraum zu erzeugen.

| Ohne Kodierung | Mit Sin/Cos-Kodierung |
|----------------|------------------------|
| 0 (Montag) und 6 (Sonntag) wirken weit entfernt | Beide liegen nah beieinander |
| 11 (Dezember) und 0 (Januar) erscheinen unverbunden | Jahreswechsel wird als glatter Übergang dargestellt |

Durch die zweidimensionale Projektion auf den Einheitskreis bleibt die zyklische Nähe mathematisch erhalten. Das hilft insbesondere neuronalen Netzen, periodische Muster über Wochen oder Jahre hinweg zu erkennen.

---

## Mathematisches Prinzip

Jede Periode (z. B. eine Woche, ein Jahr oder ein Tag) wird als Kreis dargestellt.  
Jeder Zeitpunkt x wird durch einen Winkel θ auf dem Kreis abgebildet:

θ = 2π × x / Periode

Daraus entstehen zwei orthogonale Features:

sin(θ) und cos(θ)

Diese zweidimensionale Repräsentation:
- ist kontinuierlich (keine Sprünge),
- bewahrt Abstände korrekt,
- und verbindet Periodenanfang und -ende nahtlos.

---

## Beispiel

| Wochentag | Integer | sin(2π × x / 7) | cos(2π × x / 7) |
|------------|----------|----------------|----------------|
| Montag | 0 | 0.000 | 1.000 |
| Dienstag | 1 | 0.781 | 0.624 |
| Mittwoch | 2 | 0.975 | −0.222 |
| Donnerstag | 3 | 0.433 | −0.901 |
| Freitag | 4 | −0.433 | −0.901 |
| Samstag | 5 | −0.975 | −0.222 |
| Sonntag | 6 | −0.781 | 0.624 |

Montag und Sonntag liegen trotz der numerischen Distanz (0 ↔ 6) wieder nah beieinander.

---

## Verwendung im Projektkontext

In der TFT-Pipeline (Book Sales Forecasting) wird dieser Schritt nach der allgemeinen Feature-Extraktion eingesetzt, um

- wöchentliche und saisonale Muster zu modellieren,
- periodische Abhängigkeiten zwischen Zeitpunkten abzubilden,
- und Eingabedaten konsistent mit den Anforderungen des TFT-Modells vorzubereiten.

Beispiele für Merkmale:
- day_of_week, month, hour, day_of_year
- saisonale Schwankungen im Buchverkauf (z. B. Vorweihnachtszeit, Wochenende)

---

## Implementierungsdetails

### Klasse

```python
class CyclicalEncoder:
    def fit(self, df): ...
    def transform(self, df): ...
    def fit_transform(self, df): ...
```

Der Encoder ist zustandslos – fit() dient nur der Pipeline-Kompatibilität.

### Konfigurationsdatei

```python
CYCLICAL_CONF = {
    "datetime_col": "date",
    "periodicities": {
        "dow": ("dow", 7),
        "month": ("month", 12),
        "doy": ("doy", 366),
        "week": ("week", 53),
        "hour": ("hour", 24),
    },
    "prefix": "cyc",
    "drop_source_cols": True,
    "tz": "Europe/Berlin",
}
```

### Beispielaufruf

```python
from features.cyclical_encoding import CyclicalEncoder, CyclicalEncoderConfig
enc = CyclicalEncoder(CyclicalEncoderConfig(**CYCLICAL_CONF))
df_encoded = enc.fit_transform(df)
```

---

## Eingebaute Mini-Tests (Selbstprüfung)

Der Encoder enthält einen kleinen Selbsttestblock, der bei direkter Ausführung (`python cyclical_encoding.py`) automatisch läuft.  
Er prüft:

1. Alle erwarteten Spalten sind vorhanden  
2. Wertebereich aller Sin/Cos liegt in [-1, 1]  
3. NaT-Handling: Fehlende Datumswerte erzeugen NaN statt Fehler  

Beispielausgabe:

```
[TEST] Prüfe erzeugte Spalten ...
→ OK
[TEST] Prüfe Wertebereich ...
→ OK
[TEST] Prüfe NaT-Handling ...
→ OK
[CyclicalEncoder] Alle Tests bestanden
```

Nur bei direkter Ausführung bedeutet, dass diese Tests nur dann ausgeführt werden, wenn das Skript direkt gestartet wird:

```bash
python src/features/cyclical_encoding.py
```

Beim Import in andere Module läuft der Testcode nicht automatisch.

---

## Hinweise und Fallstricke

- Zeitzonen beachten (tz-Parameter)  
- NaT-Werte bleiben NaN – bewusste Entscheidung  
- Kein inverse_transform, da die Abbildung nicht eindeutig umkehrbar ist  
- Frequenzen mischen (z. B. Stunde und Tag) nur sinnvoll, wenn beide konsistent im Datensatz vorkommen  
- doy=366 vermeidet Winkelsprung im Schaltjahr, week=53 deckt ISO-Jahre vollständig ab

---

## Zusammenfassung

| Aspekt | Beschreibung |
|--------|---------------|
| Zweck | Zyklische Zeitmerkmale kontinuierlich abbilden |
| Methode | Projektion auf Einheitskreis mit Sinus und Kosinus |
| Vorteil | Keine Diskontinuitäten, korrekte Distanzwahrung |
| Beispiele | Stunde, Wochentag, Monat, Tag im Jahr |
| Ausgabe | Zwei Features pro Merkmal (_sin, _cos) |
| Nutzen | Bessere Mustererkennung für periodische Zyklen |

