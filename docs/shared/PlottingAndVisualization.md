# PlottingAndVisualization – Visualisierung und Plot-Skripte

**Datum:** 2026-02-21  
**Ziel & Inhalt:** Bündelt projektweite Plot- und Visualisierungsskripte. Fokus: reproduzierbare Visualisierung von Datenzuständen, Modelloutputs und Interpretationsartefakten. Keine Ergebnisinterpretation, keine Hyperparameterempfehlungen.

---

## Überblick

Dieses Dokument fasst die Visualisierungs-Module zusammen, die im Projekt für Diagnose, Plausibilitätschecks und Ergebnisaufbereitung genutzt werden. Die Plots sind als Artefakte gedacht und unterstützen die Nachvollziehbarkeit der Pipeline-Ausgaben.

---

## Inhalte (Module)

### DataCleaningPlot
Visualisierung nach Datenbereinigung (z. B. Verteilungen, Ausreißerindikationen, fehlende Werte). Ziel ist die schnelle Prüfung, ob Cleaning-Schritte wie erwartet greifen.

### EvaluatorTFTPlot
Visualisierung von Vorhersagen und Ist-Werten auf Basis gespeicherter Run-Artefakte (z. B. Forecast-Verläufe pro Serie). Die Darstellung dient der qualitativen Plausibilitätsprüfung ergänzend zu numerischen Kennzahlen.

### PlotTFTInterpretation
Visualisierung von TFT-Interpretationsartefakten (z. B. Feature-Importances / Attention-bezogene Ausgaben), sofern diese vom Modell/Run bereitgestellt werden. Ziel ist die strukturierte Aufbereitung für Dokumentation und Reporting.

---

## Eingaben

- Run-Artefakte (z. B. Evaluation/Predictions, Interpretationsausgaben) je nach Plot-Modul
- Verarbeitete Datensplits (optional, abhängig vom Plot-Modul)

---

## Ausgaben

- Plot-Dateien als Artefakte (z. B. PNG/PDF) in den projektüblichen Ergebnisordnern

---

## Ergebnis und Nutzen

- Schnelle visuelle Plausibilitätschecks entlang der Pipeline  
- Einheitliche Aufbereitung von Run-Artefakten für Reporting/Docs  
- Reduzierung redundanter Plot-Dokumentation durch zentrale Bündelung
