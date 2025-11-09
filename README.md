# 📈 FOM.tft-timeseries – Zeitreihenprognose mit Temporal Fusion Transformer

Dieses Projekt untersucht die Anwendung des **Temporal Fusion Transformer (TFT)** für Zeitreihenprognosen anhand des Kaggle-Datensatzes *Tabular Playground Series – September 2022*.
Es entsteht im Rahmen des Studienprojekts *Big Data Consulting* (FOM) und orientiert sich methodisch an dem Medium-Artikel
[*Forecasting Book Sales with Temporal Fusion Transformer*](https://medium.com/dataness-ai/forecasting-book-sales-with-temporal-fusion-transformer-dd482a7a257c).

---

## 🎯 Zielsetzung
Das Projekt verfolgt das Ziel, eine modulare, nachvollziehbare und erweiterbare **Forecasting-Pipeline** auf Basis des TFT-Modells zu entwickeln.
Im Vordergrund steht die Reproduzierbarkeit und eine klare Trennung der Teilschritte für Datenaufbereitung, Modellierung, Training und Evaluation.

---

## 🧩 Vorgehensweise
Das Projekt ist objektorientiert aufgebaut und folgt einer sauberen, skriptbasierten Pipeline.
Jeder Schritt ist in einem eigenen Modul gekapselt und dokumentiert.

**Hauptphasen:**
1. **Datenvorbereitung** – Laden, Bereinigung und Transformation der Rohdaten (`src/data/`).
2. **Feature Engineering** – Erzeugung von Lags, Zeit- und Kalendermerkmalen.
3. **Modellerstellung** – Aufbau des `TimeSeriesDataSet` und Konfiguration des TFT-Modells.
4. **Training** – Durchführung mit PyTorch Lightning; Speicherung von Logs und Checkpoints.
5. **Evaluierung** – Berechnung relevanter Metriken und Visualisierung der Vorhersagen.
6. **Dokumentation** – Die Dokumentation entsteht mit **MkDocs** aus dem Ordner `docs/`.

---

## 📂 Projektstruktur
```
FOM.tft-timeseries/
├─ configs/                 # YAML-Experimente (aktuell: Baseline; weitere Varianten folgen)
│  └─ v01_baseline.yaml
├─ data/
│  └─ raw/                  # Originaldaten (nicht im Repo)
├─ src/
│  ├─ data/                 # Datensatzaufbereitung und Feature-Engineering
│  ├─ modeling/             # Modelldefinition und Training
│  ├─ evaluation/           # Metriken und Visualisierung
│  └─ utils/                # Hilfsfunktionen und Konfigurationen
├─ docs/
│  ├─ index.md
│  ├─ project/              # Projektspezifische Notizen (Module, Methoden, Pipelines)
│  └─ shared/               # Allgemeine Hinweise (Config, MkDocs, Struktur)
├─ requirements.txt
├─ .gitignore
└─ README.md
```

Alle Rohdaten werden **lokal** abgelegt unter:
```
data/raw/tabular-playground-series-sep-2022/
├─ train.csv
└─ test.csv
```
und sind **nicht Teil des Repositories**.

---

## 📘 Dokumentation mit MkDocs
Die Site wird aus `docs/` generiert. Die Navigation gruppiert projektspezifische Inhalte unter `docs/project/` und
allgemeine Hilfeseiten unter `docs/shared/`. Lokale Vorschau:

```bash
mkdocs serve
```

---

## 🧾 Lizenz und Quellen
Dieses Projekt dient ausschließlich **Lehr- und Forschungszwecken**.
Es orientiert sich konzeptionell an dem Medium-Artikel
[*Forecasting Book Sales with Temporal Fusion Transformer*](https://medium.com/dataness-ai/forecasting-book-sales-with-temporal-fusion-transformer-dd482a7a257c)
von *Dataness AI*.

Der ursprüngliche Datensatz stammt aus dem öffentlichen **Kaggle-Wettbewerb**
[Tabular Playground Series – September 2022](https://www.kaggle.com/competitions/tabular-playground-series-sep-2022/data)
und bleibt im Besitz der jeweiligen Urheber.
