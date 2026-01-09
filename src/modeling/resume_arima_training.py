"""
ARIMA Training Resume Script
=============================
Setzt das unterbrochene ARIMA Training fort, indem nur fehlende Modelle trainiert werden.

Usage:
    python resume_arima_training.py --run_dir results/arima/runs/run_20260106_180327_arima_baseline
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import warnings

warnings.filterwarnings('ignore')


def load_existing_models(run_dir):
    """Lädt Liste der bereits trainierten Modelle"""
    models_dir = Path(run_dir) / "models"
    if not models_dir.exists():
        return set()

    trained = set()
    for f in models_dir.glob("*.pkl"):
        group_name = f.stem.replace('arima_', '')
        trained.add(group_name)

    print(f"✓ {len(trained)} bereits trainierte Modelle gefunden")
    return trained


def load_all_groups(data_path):
    """Lädt alle Gruppen aus dem Original-Dataset"""
    print(f"\n📁 Lade Dataset von: {data_path}")
    df = pd.read_csv(data_path)

    # Erstelle Store_Dept Kombinationen
    df['Store_Dept'] = df['Store'].astype(str) + '_' + df['Dept'].astype(str)
    all_groups = sorted(df['Store_Dept'].unique())

    print(f"✓ {len(all_groups)} Gruppen gefunden")
    return all_groups, df


def prepare_group_data(df, group_name, lookback=100):
    """Bereitet Daten für eine Gruppe vor"""
    group_df = df[df['Store_Dept'] == group_name].copy()
    group_df = group_df.sort_values('Date')

    # Features und Target
    y = group_df['Weekly_Sales'].values

    # Validation/Test Split (wie im Original)
    # Annahme: Letzte 52 Wochen = Test, davor 52 Wochen = Val, Rest = Train
    n = len(y)
    test_size = 52
    val_size = 52

    train_end = n - test_size - val_size
    val_end = n - test_size

    return {
        'train': y[:train_end],
        'val': y[train_end:val_end],
        'test': y[val_end:],
        'full': y
    }


def train_arima_model(y_train, group_name, config):
    """Trainiert ein ARIMA Modell für eine Gruppe"""
    try:
        # auto_arima mit Config
        model = auto_arima(
            y_train,
            seasonal=True,
            m=config['seasonal_period'],
            max_p=config['max_p'],
            max_q=config['max_q'],
            max_d=config['max_d'],
            max_P=config['max_P'],
            max_Q=config['max_Q'],
            max_D=config['max_D'],
            suppress_warnings=True,
            stepwise=True,
            error_action='ignore',
            trace=False
        )
        return model, None
    except Exception as e:
        return None, str(e)


def make_forecasts(model, data_dict, prediction_length=4):
    """Erstellt Forecasts für Val und Test"""
    try:
        # Validation Forecast
        val_pred = model.predict(n_periods=len(data_dict['val']))

        # Refit auf Train+Val für Test Forecast
        train_val = np.concatenate([data_dict['train'], data_dict['val']])
        model_test = model.fit(train_val)
        test_pred = model_test.predict(n_periods=len(data_dict['test']))

        return {
            'val_pred': val_pred,
            'val_true': data_dict['val'],
            'test_pred': test_pred,
            'test_true': data_dict['test']
        }
    except Exception as e:
        return None


def calculate_metrics(y_true, y_pred):
    """Berechnet MAE, RMSE, MAPE"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'mae': mae, 'rmse': rmse, 'mape': mape}


def main():
    parser = argparse.ArgumentParser(description='Resume ARIMA Training')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Run directory mit bereits trainierten Modellen')
    parser.add_argument('--data_path', type=str,
                        default='data/raw/walmart/train.csv',
                        help='Pfad zum Original-Dataset')
    parser.add_argument('--dry_run', action='store_true',
                        help='Zeigt nur an was gemacht würde, ohne Training')
    args = parser.parse_args()

    # Config (anpassen an deine baseline.yaml!)
    config = {
        'seasonal_period': 52,  # Walmart: wöchentlich
        'max_p': 3,
        'max_q': 3,
        'max_d': 2,
        'max_P': 2,
        'max_Q': 2,
        'max_D': 1,
        'prediction_length': 4
    }

    print("=" * 70)
    print("🔄 ARIMA TRAINING RESUME")
    print("=" * 70)
    print(f"Run Directory: {args.run_dir}")
    print(f"Data Path: {args.data_path}")
    print(f"Config: m={config['seasonal_period']}, "
          f"max_p={config['max_p']}, max_q={config['max_q']}, "
          f"max_P={config['max_P']}, max_Q={config['max_Q']}")
    print("=" * 70)

    # 1. Lade bereits trainierte Modelle
    trained_groups = load_existing_models(args.run_dir)

    # 2. Lade alle Gruppen aus Dataset
    all_groups, df = load_all_groups(args.data_path)

    # 3. Berechne fehlende Gruppen
    missing_groups = sorted(set(all_groups) - trained_groups)

    print(f"\n📊 STATUS:")
    print(f"  Total Gruppen: {len(all_groups)}")
    print(f"  ✓ Trainiert: {len(trained_groups)} ({len(trained_groups) / len(all_groups) * 100:.1f}%)")
    print(f"  ✗ Fehlend: {len(missing_groups)} ({len(missing_groups) / len(all_groups) * 100:.1f}%)")

    if len(missing_groups) == 0:
        print("\n✅ Alle Modelle bereits trainiert!")
        return

    print(f"\n🎯 ZU TRAINIEREN:")
    print(f"  Erste 10: {missing_groups[:10]}")
    print(f"  Letzte 10: {missing_groups[-10:]}")

    # Zeitschätzung
    avg_time_per_model = 2  # Minuten (aus deiner Erfahrung)
    estimated_hours = len(missing_groups) * avg_time_per_model / 60
    print(f"\n⏱️  ZEITSCHÄTZUNG:")
    print(f"  ~{len(missing_groups)} Modelle × {avg_time_per_model} Min = ~{estimated_hours:.1f} Stunden")

    if args.dry_run:
        print("\n[DRY RUN] Kein Training durchgeführt.")
        return

    # Bestätigung
    response = input(f"\n▶️  {len(missing_groups)} Modelle trainieren? [y/N]: ")
    if response.lower() != 'y':
        print("Abgebrochen.")
        return

    # 4. Trainiere fehlende Modelle
    print("\n🚀 STARTE TRAINING...")
    print("=" * 70)

    models_dir = Path(args.run_dir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    results = []
    start_time = datetime.now()

    for i, group_name in enumerate(missing_groups, 1):
        group_start = datetime.now()

        # Status
        progress = i / len(missing_groups) * 100
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        avg_per_model = elapsed / i if i > 0 else 0
        remaining = avg_per_model * (len(missing_groups) - i)

        print(f"\n[{i}/{len(missing_groups)} | {progress:.1f}%] {group_name}")
        print(f"  Elapsed: {elapsed:.1f}min | Avg: {avg_per_model:.1f}min/model | ETA: {remaining:.1f}min")

        try:
            # Daten vorbereiten
            data_dict = prepare_group_data(df, group_name)

            if len(data_dict['train']) < 10:
                print(f"  ⚠️  Zu wenig Daten ({len(data_dict['train'])} samples), überspringe")
                continue

            # Training
            print(f"  🔄 Training ARIMA...")
            model, error = train_arima_model(data_dict['train'], group_name, config)

            if model is None:
                print(f"  ❌ Training fehlgeschlagen: {error}")
                results.append({
                    'group': group_name,
                    'status': 'failed',
                    'error': error
                })
                continue

            # Forecasts
            print(f"  📈 Erstelle Forecasts...")
            forecasts = make_forecasts(model, data_dict, config['prediction_length'])

            if forecasts is None:
                print(f"  ❌ Forecasts fehlgeschlagen")
                results.append({
                    'group': group_name,
                    'status': 'forecast_failed'
                })
                continue

            # Metriken
            val_metrics = calculate_metrics(forecasts['val_true'], forecasts['val_pred'])
            test_metrics = calculate_metrics(forecasts['test_true'], forecasts['test_pred'])

            # Speichere Modell
            model_path = models_dir / f"arima_{group_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            duration = (datetime.now() - group_start).total_seconds() / 60
            print(
                f"  ✓ Fertig in {duration:.1f}min | Val MAE: {val_metrics['mae']:.2f} | Test MAE: {test_metrics['mae']:.2f}")

            results.append({
                'group': group_name,
                'status': 'success',
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'duration_min': duration
            })

        except Exception as e:
            print(f"  ❌ Unerwarteter Fehler: {e}")
            results.append({
                'group': group_name,
                'status': 'error',
                'error': str(e)
            })

    # 5. Zusammenfassung
    total_time = (datetime.now() - start_time).total_seconds() / 60
    successful = sum(1 for r in results if r['status'] == 'success')

    print("\n" + "=" * 70)
    print("✅ TRAINING ABGESCHLOSSEN")
    print("=" * 70)
    print(f"Erfolgreich: {successful}/{len(missing_groups)}")
    print(f"Gesamtzeit: {total_time:.1f} Minuten ({total_time / 60:.1f} Stunden)")
    print(f"\nJetzt solltest du summary.json mit allen {len(trained_groups) + successful} Modellen erstellen!")
    print("=" * 70)


if __name__ == "__main__":
    main()

# Aufruf:
#
# Dry-Run (zeigt nur was passieren würde):
# python -m src.modeling.resume_arima_training --run_dir results/arima/runs/run_20260106_180327_arima_baseline --dry_run
#
# Echtes Training starten:
# python -m src.modeling.resume_arima_training --run_dir results/arima/runs/run_20260106_180327_arima_baseline
#
# Mit anderem Dataset-Pfad:
# python -m src.modeling.resume_arima_training --run_dir results/arima/runs/run_20260106_180327_arima_baseline --data_path "C:/path/to/train.csv"