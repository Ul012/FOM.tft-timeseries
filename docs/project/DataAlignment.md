# DataAlignment – Zweck und Funktionsweise

## Überblick
Das Modul **DataAlignment** gleicht die Verkaufszahlen der Jahre 2017–2019 auf das Niveau von 2020 an. Dieser Schritt ist notwendig, um die Daten für den **Temporal Fusion Transformer (TFT)** vergleichbar zu machen. Ohne diese Angleichung würde das Modell Unterschiede in der Höhe der Jahreswerte (z. B. generell höhere Absätze im Jahr 2020) als wichtiges Muster interpretieren, obwohl es sich nur um ein Skalierungsproblem handelt. 

Durch die Angleichung werden alle Zeitreihen auf ein gemeinsames Niveau gebracht, sodass der TFT die **zeitlichen und saisonalen Muster** innerhalb eines Jahres lernt – nicht die absoluten Unterschiede zwischen den Jahren.

---

## Ziel
Ziel ist die **Nivellierung der Verkaufsniveaus** über die Jahre hinweg. Dadurch konzentriert sich das Modell auf:
- **Saisonale Schwankungen** (z. B. Feiertagseffekte oder Quartalsveränderungen),
- **Wochen- und Monatstrends**,
- **relative Muster** zwischen Zeitpunkten,
und nicht auf absolute Mengenunterschiede.

Die Angleichung verbessert somit die Stabilität des Trainings und die Generalisierungsfähigkeit des TFT.

---

## Begriffliche Einordnung
Der beschriebene Vorgang entspricht einer **Normalisierung auf ein Referenzniveau** (auch „Baseline-Normalisierung“ genannt). Dabei werden alle Zeitreihen auf dasselbe Skalenniveau gebracht, hier das von 2020. Im Gegensatz zur **Standardisierung**, bei der Werte so transformiert werden, dass sie Mittelwert = 0 und Standardabweichung = 1 besitzen, geht es bei der Normalisierung um eine relative Anpassung an ein gemeinsames Vergleichsniveau. 

Durch diese Normalisierung werden die Daten skalentechnisch harmonisiert, ohne dass ihre inneren Strukturen oder saisonalen Muster verloren gehen.

---

## Vorgehensweise

### 1. Berechnung der Jahresmittelwerte
Für jedes Land und Jahr wird der durchschnittliche Absatz berechnet:
\[
\text{mean\_year}_{c,y} = \text{mean}(\text{num\_sold}_{c,y})
\]
Dabei steht \(c\) für das Land und \(y\) für das Jahr.

### 2. Wahl der Referenzperiode
Die Wahl des Referenzjahres erfolgt in der Regel auf Basis des Datensatzes. In diesem Projekt wurde **2020** als Referenz gewählt, da es das aktuellste und vollständigste Jahr darstellt. 

Allgemein gilt:
- Für **reale Wirtschaftsdaten** empfiehlt sich das **neueste stabile Jahr**, um aktuelle Marktverhältnisse abzubilden.
- Für **synthetische oder experimentelle Daten** kann das **erste Jahr** als Ausgangsbasis dienen.
- Wenn es deutliche Strukturbrüche gibt (z. B. Pandemie, Systemwechsel), sollte das **repräsentativste Jahr** als Basis dienen.

### 3. Berechnung der Skalierungsfaktoren
Für jedes Land wird ein Faktor berechnet, der angibt, um wie viel die Werte eines Jahres angepasst werden müssen, um das gleiche Mittel wie 2020 zu erreichen:
\[
\text{factor}_{c,y} = \frac{\text{mean\_2020}_c}{\text{mean\_year}_{c,y}}
\]

- Für 2020 selbst gilt: \(\text{factor} = 1.0\)
- Für Länder oder Jahre mit fehlenden Werten oder Null-Durchschnitten bleibt der Faktor ebenfalls 1.0, um Divisionen durch Null zu vermeiden.

### 4. Anwendung der Skalierung
Der Faktor wird direkt auf die Verkaufszahlen angewendet:
\[
\text{num\_sold\_aligned} = \text{num\_sold} \times \text{factor}_{c,y}
\]
Damit werden die Jahresmittelwerte auf dasselbe Niveau gebracht, die innerjährigen Schwankungen bleiben jedoch erhalten.

---

## Technische Umsetzung
Die Umsetzung erfolgt vektorbasiert in **pandas**, ohne explizite Schleifen:
- `groupby(["country", "year"]).mean()` berechnet die Jahresmittelwerte.
- `merge()` verbindet die Mittelwerte mit den 2020-Referenzen.
- Die Multiplikation mit den Faktoren wird elementweise auf `num_sold` angewendet.

Das Ergebnis wird als **Parquet-Datei** gespeichert, da dieses Format im Vergleich zu CSV deutlich effizienter, typensicher und schneller ladbar ist.

---

Optionale Visualisierung

Zur qualitativen Überprüfung des Angleichungsschritts existiert im Ordner
`src/visualization/` das Skript `data_alignment_plot.py`.
Es ist nicht Bestandteil der Datenpipeline, sondern dient ausschließlich der visuellen Kontrolle und dem besseren Verständnis des Effekts der Skalierung. Das Skript lädt die Datei `data/interim/train_aligned.parquet` und stellt die auf 2020-Niveau skalierten Verkaufszahlen als Liniendiagramm dar.

---

## Ergebnis und Nutzen
Nach der Skalierung sind die durchschnittlichen Verkaufsniveaus aller Jahre pro Land nahezu identisch (Verhältnis ≈ 1,0). Dadurch:
- werden die Zeitreihen **direkt vergleichbar**,
- bleibt die **Saisonalität** erhalten,
- kann der TFT **stabiler trainieren**, da Wertebereiche konsistent sind.

Diese Normalisierung entspricht inhaltlich einer Form der **Skalenharmonisierung** – sie eliminiert globale Niveauunterschiede und fokussiert das Modell auf die **zeitliche Dynamik** der Daten.

Dieser Schritt verändert zwar die absoluten Verkaufsniveaus, erhält jedoch alle relativen Muster und saisonalen Strukturen. Für Modelle wie den Temporal Fusion Transformer, die empfindlich auf unterschiedliche Skalen reagieren, führt diese Harmonisierung erfahrungsgemäß zu einer stabileren Konvergenz und reproduzierbareren Trainingsergebnissen.