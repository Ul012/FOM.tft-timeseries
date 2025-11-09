# Verwendung von MkDocs

Die Projektdokumentation wird mit **MkDocs** erstellt.  
Alle Inhalte liegen im Ordner `docs/` und werden automatisch aus den Markdown-Dateien generiert.

## Aufbau
- `mkdocs.yml` definiert Navigation, Themen und Struktur.  
- Unterordner `docs/` enthält technische Notizen und Modulerläuterungen.  
- Die Hauptübersicht befindet sich in `docs/index.md`.

## Lokale Vorschau
Zum Starten der Dokumentation:
```bash
mkdocs serve
```
Danach ist die Site unter [http://localhost:8000](http://localhost:8000) abrufbar.

## Veröffentlichung
Optional kann die Dokumentation mit:
```bash
mkdocs build
```
generiert und anschließend auf GitHub Pages oder einem anderen Hosting-Dienst bereitgestellt werden.
