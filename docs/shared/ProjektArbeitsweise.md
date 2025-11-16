# Projektarbeitsweise – Struktur, Konventionen und Vorgehen

Dieses Dokument beschreibt die zentralen Arbeitsregeln und Konventionen für das TFT-TimeSeries-Projekt.  
Alle neuen Skripte, Erweiterungen und Prompts sollen sich an diesen Leitlinien orientieren.

---

## A. Grundprinzipien

### 1. Minimalinvasives Arbeiten
Änderungen erfolgen nur dort, wo sie notwendig sind. Die bestehende Struktur wird beibehalten.

### 2. Clean Code
- Klare, schlanke Funktionen  
- Keine unnötige Objektorientierung  
- Keine doppelten Pfad- oder Parameterdefinitionen  
- Keine versteckten Defaults  

### 3. Konfigurationsgetriebene Pipeline
- Alle Parameter werden zentral über `src/config.py` oder YAML-Dateien gesteuert.  
- Kein Hardcoding von Pfaden oder Modellparametern.

### 4. Vermeidung von Over-Engineering
- Keine komplexen Framework-Schichten  
- Keine indirekten oder try/except-Fallback-Importe  
- Fehler sollen klar erkennbar sein

---

## B. Skriptaufbau

Einheitliche Struktur:

1. Imports  
2. Konstanten aus `src/config`  
3. Funktionen / kompakte Klassen  
4. `main()`  
5. Modulstart:
   ```python
   if __name__ == "__main__":
       main()
   ```

---

## C. Dokumentation

- Jedes Modul erhält eine kurze, sachliche `.md`-Dokumentation.  
- Fokus auf Input, Output und Rolle im Pipeline-Ablauf.

---

## D. Zielsetzung der Pipeline

Die Pipeline soll:
- reproduzierbar,
- klar strukturiert,
- konfigurationsbasiert,
- leicht erweiterbar (ARIMA, Prophet, Optuna, MLflow)

sein.

Dies ermöglicht paralleles Arbeiten ohne strukturelle Änderungen an der Gesamtpipeline.
