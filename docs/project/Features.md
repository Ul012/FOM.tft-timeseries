# Feature-Erstellung (FeatureEngineer + Lag/Rolling intern + CyclicalEncoder)

## Überblick
`src/data/features.py` erzeugt **modellfertige Features** in einem Lauf – inklusive
1) Basis-Zeitfeatures,  
2) **Lag- und Rolling-Features (intern, leakage-sicher)** und  
3) zyklischer Sin/Cos-Kodierung.

Das Skript unterstützt **zwei Rohdateien** (`train.csv`, `test.csv`) **oder** eine einzelne Datei (`dataset.csv`) mit internem 80/20-Split (zeitbasiert, wenn eine Datumsspalte vorhanden ist).

---

## Ziele
- Einheitliche, reproduzierbare Feature-Erzeugung für Train und Test  
- **Kein Datenleck**: Lags/Rollings nutzen ausschließlich Vergangenheit; Skalierungen/Encodings erfolgen später modellnah **nur** auf Basis von Train  
- Harmonisierung von Datentypen (insb. `date`)

---

## Inputs
- **Variante A (empfohlen):**  
  `data/raw/.../train.csv`, `data/raw/.../test.csv`
- **Variante B (Fallback):**  
  `data/raw/.../dataset.csv` → interner 80/20-Split

> Pfade werden in `src/config.py` unter `RAW` konfiguriert.

## Outputs
- `data/processed/features_train.parquet`  
- `data/processed/features_test.parquet`

---

## Komponenten

### 1) FeatureEngineer
Erzeugt grundlegende Zeitmerkmale:
- `year`, `month`, `day`, `dayofweek`, `weekofyear`, `is_weekend`, `time_idx`
- `is_holiday_de` (Feiertagsflag)

### 2) **Interner Lag-/Rolling-Helper**
- Lags: z. B. `lag_num_sold_1`, `lag_num_sold_7`, `lag_num_sold_28`  
- Rolling-Stats (auf Vergangenheitswerten): z. B. `lag_num_sold_roll7_mean`, `lag_num_sold_roll28_std`  
- Gruppierung optional über `GROUP_COLS` (z. B. `["country","store","product"]`)

### 3) CyclicalEncoder
- Sin/Cos-Kodierung zyklischer Merkmale (z. B. DOW, Monat, DOY)

---

## Orchestrierung (vereinfacht)
```python
from src.data.features import build_features

# jeweils separat für Train & Test (oder intern gesplittet)
df_features = build_features(df, use_lags=True, use_cyclical=True)
```

**Datentypen:** `date` wird vor Verarbeitung auf `datetime64[ns]` gebracht; vorm Merge werden Typen harmonisiert.  
**Target im Test:** darf fehlen – Lags/Rollings werden dann ausgelassen (kein Crash, kein Leak).

---

## Projektstruktur (Auszug)
```
src/
├── data/
│   ├── feature_engineering.py
│   ├── cyclical_encoder.py
│   └── features.py              # Orchestriert Basis, Lag/Rolling (intern) und Cyclical
└── modeling/
    ├── model_dataset.py         # Train/Valid-Test-Datasets + meta.json
    └── dataset_tft.py           # Encoding/Skalierung (fit auf Train), Artefakte
```

---

## Sanity-Checks
- `features_train.parquet` und `features_test.parquet` haben **gleiche Spalten** (Test-Target kann NaN sein).  
- Lags der ersten `k` Zeilen sind erwartungsgemäß `NaN` (abhängig von `lags`).  
- `date` ist in beiden Dateien `datetime64[ns]`.

---

## Historie
Das frühere separate Modul **`lag_features.py`** ist **ersetzt**: Lag/Rolling ist jetzt **in `features.py` integriert**.  
Die Dokumentation wurde entsprechend bereinigt.
