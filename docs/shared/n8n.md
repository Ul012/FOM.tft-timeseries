# n8n – Prozessautomatisierung und Systemintegration

**Datum:** 2026-02-21  
**Script:** —  
**Ziel & Inhalt:** Dokumentiert den Einsatz von n8n als nachgelagerte Orchestrierungs-, Interpretations- und Distributionsschicht für erzeugte Analyseartefakte. Grundlage ist ausschließlich das in der Projektarbeit beschriebene Kapitel zu n8n.

---

## Überblick

In der Projektarbeit wird **n8n** als Workflow-Automation-Tool eingesetzt, um erzeugte Ergebnisse zu **orchestrieren**, **anzureichern** und in geeigneter Form zu **distribuieren**. Workflows bestehen aus verbundenen Knoten (Nodes), über die Prozesse automatisiert und Integrationen umgesetzt werden. fileciteturn4file0

Workflow- und Prozessautomationssysteme übernehmen dabei wiederkehrende Ausführung, Abhängigkeitsmanagement und Monitoring als koordinierende Ebene, die zeit- und ereignisgesteuerte Aufgaben ausführt und Informationen kontrolliert weitergibt. fileciteturn4file2turn4file0

---

## Systemabgrenzung

Die Systemabgrenzung ist in der Projektarbeit explizit:  
n8n bildet eine **nachgelagerte Verarbeitungsschicht** und **beeinflusst die Prognosemodelle nicht**. Training, Parametrisierung und Inferenz der Modelle (ARIMA, Prophet, TFT) verbleiben vollständig im Analyseframework. n8n greift ausschließlich auf erzeugte Ausgaben zu, reichert diese an und verteilt sie. fileciteturn4file0turn4file2

Diese Trennung dient der methodischen Vergleichbarkeit und verhindert Eingriffe in die Modelllogik. fileciteturn4file0

---

## Rolle im Gesamtsystem

n8n unterstützt die „technische Kommunikation und Verteilung“ von Forecast-Ergebnissen über eine Orchestrierungsschicht. Analyseartefakte werden dadurch konsistent bereitgestellt und an relevante Zielsysteme oder Empfängergruppen verteilt. fileciteturn4file2turn4file0

---

## Optionale Einbindung von LLM-Komponenten (nachgelagert)

Die Projektarbeit beschreibt n8n zusätzlich als möglichen Ausführungskontext für **LLM-basierte Auswertungen**, ohne die Prognosemodelle zu verändern. Genannt werden zwei komplementäre Funktionen:

- (Teil-)automatisierte Anomalie-Identifikation bzw. Plausibilisierung auffälliger Muster in Forecast-Ergebnissen  
- Generierung verständlicher textueller Erläuterungen, Trendzusammenfassungen und handlungsorientierter Hinweise als Entscheidungsunterstützung fileciteturn4file0

Die Einbindung bleibt dabei nachgelagert: Prognosemodelle liefern Forecasts und Bewertungsmetriken als numerische Grundlage; n8n unterstützt Orchestrierung/Distribution, und LLM-Komponenten können als interpretierende Ebene ergänzt werden. fileciteturn4file0

---

## Ergebnis und Nutzen

- Nachgelagerte Orchestrierung und Distribution von Analyseartefakten  
- Klare Trennung zwischen Modelllogik und Kommunikations-/Integrationsschicht  
- Optionale Erweiterbarkeit um interpretierende Komponenten (LLM) ohne Eingriff in die Prognosemodelle fileciteturn4file0turn4file2
