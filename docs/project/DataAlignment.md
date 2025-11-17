# DataAlignment – Zweck und Funktionsweise

**Datum:** 2025-11-17  
**Script:** src/data/data_alignment.py  
**Ziel & Inhalt:** Normalisierung der Verkaufsniveaus auf ein gemeinsames Referenzjahr (2020). Beschreibt Motivation, mathematische Herleitung, Implementierungsschritte und Nutzen für stabile zeitreihenbasierte Modellierung.


## Überblick
Das Modul DataAlignment passt die Verkaufszahlen der Jahre 2017–2019 an das Niveau von 2020 an. Dadurch liegen alle Jahre auf einer gemeinsamen Skala, was die spätere Bereinigung, Feature-Erzeugung und Analyse unterstützt. Die saisonalen und innerhalb eines Jahres relevanten Muster bleiben vollständig erhalten.

---

## Ziel
Ziel ist die **Nivellierung der Jahresniveaus**, damit sich folgende Analyseschritte auf relative zeitliche Muster konzentrieren können:
- saisonale und kalendarische Effekte  
- Wochen- und Monatstrends  
- innerjährige Dynamiken  

Absolute Jahresunterschiede werden vereinheitlicht; die Struktur der Zeitreihen bleibt erhalten.

---

## Begriffliche Einordnung
Der Prozess entspricht einer **Normalisierung auf ein Referenzniveau** („Baseline-Normalisierung“). Dabei werden alle Jahre eines Landes auf das gleiche Durchschnittsniveau gebracht.  
Im Unterschied zur **Standardisierung** (Mittelwert = 0, Standardabweichung = 1) bleibt die ursprüngliche Struktur bestehen; lediglich die Skala wird angeglichen.

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
Es ist nicht Bestandteil der Datenpipeline, sondern dient ausschließlich der visuellen Kontrolle und dem besseren Verständnis des Effekts der Skalierung. Das Skript lädt die Datei `data/interim/train_aligned.parquet`.

---

## Ergebnis und Nutzen
Nach der Anpassung besitzen alle Jahre eines Landes ein vergleichbares Verkaufsniveau.  
Dies führt zu:
- einheitlicheren Wertebereichen,  
- besserer Vergleichbarkeit,  
- stabileren Modelltrainings.  

Die relevanten inneren Muster der Zeitreihen bleiben vollständig erhalten.
