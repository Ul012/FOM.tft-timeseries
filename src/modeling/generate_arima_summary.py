"""
ARIMA Summary Generator
=======================
Erstellt summary.json für alle trainierten ARIMA Modelle und führt Evaluation durch.

Usage:
    python -m src.modeling.generate_arima_summary --run_dir results/arima/runs/run_20260106_180327_arima_baseline
    python -m src.modeling.generate_arima_summary --run_dir results/arima/runs/run_20260106_180327_arima_baseline --save_forecasts
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def load_all_models(run_dir):
    """Lädt alle trainierten ARIMA Modelle"""
    models_dir = Path(run_dir) / "models"
    if not models_dir.exists():
        print(f"❌ Models directory nicht gefunden: {models_dir}")
        return {}

    models = {}
    pkl_files = list(models_dir.glob("*.pkl"))

    print(f"\n📦 Lade {len(pkl_files)} Modelle...")

    for pkl_file in tqdm(pkl_files):
        group_name = pkl_file.stem.replace('arima_', '')
        try:
            with open(pkl_file, 'rb') as f:
                model = pickle.load(f)
                models[group_name] = model
        except Exception as e:
            print(f"⚠️  Fehler beim Laden von {group_name}: {e}")

    print(f"✓ {len(models)} Modelle geladen")
    return models


def load_dataset(data_path):
    """Lädt das Walmart Dataset"""
    print(f"\n📁 Lade Dataset: {data_path}")
    df = pd.read_csv(data_path)
    df['Store_Dept'] = df['Store'].astype(str) + '_' + df['Dept'].astype(str)
    print(f"✓ {len(df)} Zeilen geladen, {df['Store_Dept'].nunique()} Gruppen")
    return df


def prepare_group_data(df, group_name):
    """Bereitet Daten für eine Gruppe vor"""
    group_df = df[df['Store_Dept'] == group_name].copy()
    group_df = group_df.sort_values('Date')

    y = group_df['Weekly_Sales'].values

    # Split wie im Training
    n = len(y)
    test_size = 52
    val_size = 52

    train_end = n - test_size - val_size
    val_end = n - test_size

    return {
        'train': y[:train_end],
        'val': y[train_end:val_end],
        'test': y[val_end:],
        'dates_val': group_df.iloc[train_end:val_end]['Date'].values,
        'dates_test': group_df.iloc[val_end:]['Date'].values
    }


def make_forecasts(model, data_dict):
    """Erstellt Forecasts für Val und Test"""
    try:
        # Validation Forecast
        val_pred = model.predict(n_periods=len(data_dict['val']))

        # Refit auf Train+Val für Test
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
        print(f"    ⚠️  Forecast Error: {e}")
        return None


def calculate_metrics(y_true, y_pred):
    """Berechnet MAE, RMSE, MAPE"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAPE mit Schutz vor Division durch 0
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float('inf')

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape)
    }


def save_forecasts_csv(run_dir, group_name, forecasts, data_dict):
    """Speichert Forecasts als CSV"""
    forecasts_dir = Path(run_dir) / "forecasts"
    forecasts_dir.mkdir(exist_ok=True)

    # Validation Forecasts
    val_df = pd.DataFrame({
        'date': data_dict['dates_val'],
        'y_true': forecasts['val_true'],
        'y_pred': forecasts['val_pred']
    })
    val_df.to_csv(forecasts_dir / f"{group_name}_val.csv", index=False)

    # Test Forecasts
    test_df = pd.DataFrame({
        'date': data_dict['dates_test'],
        'y_true': forecasts['test_true'],
        'y_pred': forecasts['test_pred']
    })
    test_df.to_csv(forecasts_dir / f"{group_name}_test.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description='Generate ARIMA Summary')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Run directory mit trainierten Modellen')
    parser.add_argument('--data_path', type=str,
                        default='data/raw/walmart/train.csv',
                        help='Pfad zum Original-Dataset')
    parser.add_argument('--save_forecasts', action='store_true',
                        help='Speichere Forecasts als CSV')
    args = parser.parse_args()

    print("=" * 70)
    print("📊 ARIMA SUMMARY GENERATOR")
    print("=" * 70)
    print(f"Run Directory: {args.run_dir}")
    print(f"Data Path: {args.data_path}")
    print("=" * 70)

    # 1. Lade Modelle
    models = load_all_models(args.run_dir)
    if len(models) == 0:
        print("❌ Keine Modelle gefunden!")
        return

    # 2. Lade Dataset
    df = load_dataset(args.data_path)

    # 3. Evaluiere alle Modelle
    print(f"\n🔍 Evaluiere {len(models)} Modelle...")

    results = {
        'val': {},
        'test': {}
    }

    failed_groups = []

    for i, (group_name, model) in enumerate(tqdm(models.items()), 1):
        try:
            # Daten vorbereiten
            data_dict = prepare_group_data(df, group_name)

            # Forecasts
            forecasts = make_forecasts(model, data_dict)

            if forecasts is None:
                failed_groups.append(group_name)
                continue

            # Metriken
            val_metrics = calculate_metrics(forecasts['val_true'], forecasts['val_pred'])
            test_metrics = calculate_metrics(forecasts['test_true'], forecasts['test_pred'])

            results['val'][group_name] = val_metrics
            results['test'][group_name] = test_metrics

            # Optional: Speichere Forecasts
            if args.save_forecasts:
                save_forecasts_csv(args.run_dir, group_name, forecasts, data_dict)

        except Exception as e:
            print(f"\n⚠️  Fehler bei {group_name}: {e}")
            failed_groups.append(group_name)

    # 4. Aggregierte Metriken
    print("\n📈 Berechne aggregierte Metriken...")

    def aggregate_metrics(metrics_dict):
        """Berechnet Durchschnitt über alle Gruppen"""
        all_mae = [m['mae'] for m in metrics_dict.values() if not np.isinf(m['mape'])]
        all_rmse = [m['rmse'] for m in metrics_dict.values() if not np.isinf(m['mape'])]
        all_mape = [m['mape'] for m in metrics_dict.values() if not np.isinf(m['mape'])]

        return {
            'mae_mean': float(np.mean(all_mae)),
            'mae_std': float(np.std(all_mae)),
            'rmse_mean': float(np.mean(all_rmse)),
            'rmse_std': float(np.std(all_rmse)),
            'mape_mean': float(np.mean(all_mape)),
            'mape_std': float(np.std(all_mape)),
            'n_groups': len(all_mae)
        }

    summary = {
        'run_info': {
            'run_id': Path(args.run_dir).name,
            'created_at': datetime.now().isoformat(),
            'total_groups': len(models),
            'successful_groups': len(results['val']),
            'failed_groups': len(failed_groups)
        },
        'aggregated_metrics': {
            'val': aggregate_metrics(results['val']),
            'test': aggregate_metrics(results['test'])
        },
        'per_group_metrics': {
            'val': results['val'],
            'test': results['test']
        },
        'failed_groups': failed_groups
    }

    # 5. Speichere Summary
    summary_path = Path(args.run_dir) / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary gespeichert: {summary_path}")

    # 6. Ausgabe
    print("\n" + "=" * 70)
    print("📊 RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Gruppen: {summary['run_info']['total_groups']}")
    print(f"Erfolgreich: {summary['run_info']['successful_groups']}")
    print(f"Fehlgeschlagen: {summary['run_info']['failed_groups']}")

    print(f"\n🎯 VALIDATION METRICS:")
    val_agg = summary['aggregated_metrics']['val']
    print(f"  MAE:  {val_agg['mae_mean']:.2f} ± {val_agg['mae_std']:.2f}")
    print(f"  RMSE: {val_agg['rmse_mean']:.2f} ± {val_agg['rmse_std']:.2f}")
    print(f"  MAPE: {val_agg['mape_mean']:.2f}% ± {val_agg['mape_std']:.2f}%")

    print(f"\n🎯 TEST METRICS:")
    test_agg = summary['aggregated_metrics']['test']
    print(f"  MAE:  {test_agg['mae_mean']:.2f} ± {test_agg['mae_std']:.2f}")
    print(f"  RMSE: {test_agg['rmse_mean']:.2f} ± {test_agg['rmse_std']:.2f}")
    print(f"  MAPE: {test_agg['mape_mean']:.2f}% ± {test_agg['mape_std']:.2f}%")
    print("=" * 70)

    # Speichere auch separate eval files
    eval_val_path = Path(args.run_dir) / "eval_val.json"
    eval_test_path = Path(args.run_dir) / "eval_test.json"

    with open(eval_val_path, 'w') as f:
        json.dump(results['val'], f, indent=2)

    with open(eval_test_path, 'w') as f:
        json.dump(results['test'], f, indent=2)

    print(f"\n✓ Evaluation files gespeichert:")
    print(f"  - {eval_val_path}")
    print(f"  - {eval_test_path}")


if __name__ == "__main__":
    main()

# =============================================================================
# AUFRUFE
# =============================================================================
#
# Summary ohne Forecasts:
# python -m src.modeling.generate_arima_summary --run_dir results/arima/runs/run_20260106_180327_arima_baseline
#
# Summary mit Forecast-CSVs (empfohlen):
# python -m src.modeling.generate_arima_summary --run_dir results/arima/runs/run_20260106_180327_arima_baseline --save_forecasts
#
# Mit anderem Dataset-Pfad:
# python -m src.modeling.generate_arima_summary --run_dir results/arima/runs/run_20260106_180327_arima_baseline --data_path "C:/path/to/train.csv"
#
# =============================================================================