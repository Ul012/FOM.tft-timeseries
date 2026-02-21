# src/utils/analyze_tft_optuna.py
"""
Analysiert und fasst Optuna-Studien für TFT-Experimente zusammen.

Das Script liest eine bestehende Optuna-Storage-Datei (z. B. SQLite),
listet enthaltene Studies auf und gibt eine strukturierte Übersicht
zu Trials, Status-Verteilung und bestem Objective-Wert aus.

Es verändert keine Daten und dient ausschließlich der Auswertung
und Diagnose bereits durchgeführter Hyperparameter-Suchen.
"""

import optuna
from pathlib import Path
from src.config import BASE_DIR
from datetime import datetime

db = BASE_DIR / 'results' / 'tft' / 'optuna' / 'booksales' / 'tft_studies.db'
storage = f'sqlite:///{db}'

print('='*80)
print('TFT BOOKSALES - OPTUNA STUDIES ÜBERSICHT')
print('='*80)

# Hole alle Study-Namen
try:
    studies = optuna.study.get_all_study_names(storage)
    print(f'Gefundene Studies: {len(studies)}\n')
    
    for study_name in sorted(studies):
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            trials = study.trials
            completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
            pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
            failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]
            
            print(f'Study: {study_name}')
            print(f'  Total Trials:     {len(trials)}')
            print(f'  Completed:        {len(completed)}')
            print(f'  Pruned:           {len(pruned)}')
            print(f'  Failed:           {len(failed)}')
            
            if completed:
                print(f'  Best val_loss:    {study.best_value:.4f}')
                print(f'  Best trial:       #{study.best_trial.number}')
                
                # Zeige erste und letzte Trial-Zeit
                times = [t.datetime_complete for t in completed if t.datetime_complete]
                if times:
                    print(f'  Erstellt:         {min(times).strftime("%Y-%m-%d %H:%M")}')
                    print(f'  Letztes Trial:    {max(times).strftime("%Y-%m-%d %H:%M")}')
            else:
                print(f'  Status:           KEINE ABGESCHLOSSENEN TRIALS')
            
            print()
            
        except Exception as e:
            print(f'Study: {study_name}')
            print(f'  ERROR: {str(e)}')
            print()
            
except Exception as e:
    print(f'Fehler beim Laden der Studies: {e}')

print('='*80)

# Zeige auch Trial-Ordner
print('\nTRIAL-ORDNER:')
print('='*80)
optuna_dir = BASE_DIR / 'results' / 'tft' / 'optuna' / 'booksales'
trial_dirs = sorted([d for d in optuna_dir.iterdir() if d.is_dir() and d.name.startswith('trial_')])
print(f'Gefundene Trial-Ordner: {len(trial_dirs)}')
if trial_dirs:
    print(f'  Erste 5: {[d.name for d in trial_dirs[:5]]}')
    print(f'  Letzte 5: {[d.name for d in trial_dirs[-5:]]}')
print('='*80)
