# src/data/analyze_dataset.py
"""
Analysiert neue Datensätze automatisch und generiert eine vorgeschlagene YAML-Config.

Aufruf:
    python -m src.data.analyze_dataset --path data/raw/walmart/train.csv
    python -m src.data.analyze_dataset --path data/raw/walmart/train.csv --name walmart
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import yaml


def detect_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """Findet automatisch die Datetime-Spalte."""
    # Direkte Datetime-Dtypes
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    if datetime_cols:
        return datetime_cols[0]

    # Versuche String-Spalten zu parsen
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                pd.to_datetime(df[col].head(100), errors="raise")
                return col
            except:
                continue

    return None


def detect_frequency(df: pd.DataFrame, time_col: str) -> Tuple[str, int]:
    """Erkennt Frequenz der Zeitreihe (täglich/wöchentlich/monatlich)."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Median der Zeitabstände
    sorted_dates = df[time_col].sort_values().unique()
    if len(sorted_dates) < 2:
        return "unknown", 0

    diffs = pd.Series(sorted_dates[1:]) - pd.Series(sorted_dates[:-1])
    median_days = diffs.median().days

    if median_days <= 1:
        return "daily", 1
    elif 6 <= median_days <= 8:
        return "weekly", 7
    elif 28 <= median_days <= 31:
        return "monthly", 30
    else:
        return f"custom ({median_days} days)", median_days


def detect_id_columns(df: pd.DataFrame, time_col: str, max_unique: int = 1000) -> List[str]:
    """Identifiziert ID-Spalten (kategorisch mit wenigen unique values)."""
    id_cols = []
    for col in df.columns:
        if col == time_col:
            continue
        if df[col].dtype in ["object", "category", "int64", "int32"]:
            n_unique = df[col].nunique()
            if 1 < n_unique <= max_unique:
                id_cols.append(col)
    return id_cols


def suggest_target(df: pd.DataFrame, time_col: str, id_cols: List[str]) -> List[str]:
    """Schlägt Target-Spalten vor (numerisch, keywords im Namen)."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Filter: nicht time_col, nicht id_cols
    candidates = [c for c in numeric_cols if c != time_col and c not in id_cols]

    # Priorisiere Spalten mit Keywords
    keywords = ["sales", "value", "amount", "revenue", "sold", "demand"]
    prioritized = []
    others = []

    for col in candidates:
        if any(kw in col.lower() for kw in keywords):
            prioritized.append(col)
        else:
            others.append(col)

    return prioritized + others


def analyze_group_lengths(df: pd.DataFrame, time_col: str, id_cols: List[str],
                          train_ratio: float = 0.8) -> Dict:
    """Analysiert Zeitreihen-Längen pro Gruppe nach Train-Split."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Simuliere Train-Split
    df_sorted = df.sort_values(time_col)
    split_idx = int(len(df_sorted) * train_ratio)
    df_train = df_sorted.iloc[:split_idx]

    # Gruppenlängen berechnen
    if id_cols:
        lengths = df_train.groupby(id_cols).size()
    else:
        lengths = pd.Series([len(df_train)])

    return {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "median": int(lengths.median()),
        "mean": round(lengths.mean(), 1),
        "distribution": {
            "<20": int((lengths < 20).sum()),
            "<30": int((lengths < 30).sum()),
            "<40": int((lengths < 40).sum()),
            ">=40": int((lengths >= 40).sum()),
        },
        "total_groups": len(lengths),
    }


def calculate_tft_params(group_stats: Dict, freq_days: int) -> Dict:
    """Berechnet TFT-Parameter basierend auf Gruppen-Statistiken."""
    min_length = group_stats["min"]

    # Encoder: 30-40% des kürzesten Zeitraums, gerundet
    encoder_length = max(8, int(min_length * 0.35))

    # Prediction: 10-15% von encoder
    prediction_length = max(1, int(encoder_length * 0.12))

    # Lag-Vorschläge basierend auf Frequenz
    if freq_days == 1:  # täglich
        lags = [1, 7, 14, 30]
        roll_windows = [7, 14]
    elif freq_days == 7:  # wöchentlich
        lags = [1, 4, 8, 12]
        roll_windows = [4, 8]
    elif freq_days == 30:  # monatlich
        lags = [1, 3, 6, 12]
        roll_windows = [3, 6]
    else:  # custom
        lags = [1, 4, 8]
        roll_windows = [4]

    return {
        "max_encoder_length": encoder_length,
        "max_prediction_length": prediction_length,
        "lags": lags,
        "roll_windows": roll_windows,
    }


def analyze_nan_patterns(df: pd.DataFrame) -> Dict[str, float]:
    """Analysiert NaN-Muster pro Spalte."""
    nan_pct = (df.isna().sum() / len(df) * 100).round(2)
    return {col: pct for col, pct in nan_pct.items() if pct > 0}


def generate_yaml_config(name: str, analysis: Dict) -> Dict:
    """Generiert YAML-Config aus Analyse-Ergebnissen."""
    time_col = analysis["time_col"]
    id_cols = analysis["id_cols"]
    target_col = analysis["target_suggestions"][0] if analysis["target_suggestions"] else "UNKNOWN"
    freq_name = analysis["frequency"][0]
    tft_params = analysis["tft_params"]

    config = {
        "name": name,
        "description": f"Auto-generated config for {name} dataset ({freq_name} data)",

        "paths": {
            "raw": f"data/raw/{name}",
            "interim": f"data/interim/{name}",
            "processed": f"data/processed/{name}",
        },

        "raw_data": {
            "type": "single_file",
            "files": [
                {
                    "path": f"data/raw/{name}/train.csv",
                    "role": "main",
                }
            ]
        },

        "schema": {
            "time_col": time_col,
            "id_cols": id_cols,
            "target_col": target_col,
        },

        "preprocessing": [
            {"step": "load_raw", "enabled": True},
            {"step": "alignment", "enabled": False, "description": "Enable if needed"},
            {"step": "cleaning", "enabled": False, "description": "Enable if needed"},
            {"step": "feature_engineering", "enabled": True,
             "params": {"country": "US", "include_holiday_name": False}},
            {"step": "cyclical_encoder", "enabled": True, "params": {
                "periodicities": {
                    "dow": ["dayofweek", 7] if freq_name == "daily" else None,
                    "week": ["weekofyear", 52] if freq_name == "weekly" else None,
                    "month": ["month", 12],
                }
            }},
            {"step": "lag_features", "enabled": True, "params": {
                "lags": tft_params["lags"],
                "roll_windows": tft_params["roll_windows"],
                "roll_stats": ["mean"],
                "prefix": "lag_",
            }},
        ],

        "split": {
            "method": "ratio",
            "ratios": [0.80, 0.10, 0.10],
            "scale_cols": [],
        },

        "tft": {
            "max_encoder_length": tft_params["max_encoder_length"],
            "max_prediction_length": tft_params["max_prediction_length"],
            "known_real_prefixes": ["cyc_"],
            "lag_prefixes": ["lag_"],
            "treat_calendar_as_known": True,
            "flag_cols": [],
        },
    }

    # Entferne None-Werte aus periodicities
    config["preprocessing"][4]["params"]["periodicities"] = {
        k: v for k, v in config["preprocessing"][4]["params"]["periodicities"].items() if v is not None
    }

    return config


def print_report(name: str, analysis: Dict, path: Path) -> None:
    """Druckt ausführlichen Terminal-Report."""
    print("\n" + "=" * 70)
    print(f"Dataset-Analyse: {name}")
    print(f"Datei: {path}")
    print(f"Zeilen: {analysis['rows']:,} | Spalten: {analysis['cols']} | Memory: {analysis['memory_mb']:.1f} MB")

    print("\nZeitreihen-Eigenschaften:")
    print(f"  time_col: {analysis['time_col']} ({analysis['date_range'][0]} bis {analysis['date_range'][1]})")
    print(f"  Frequenz: {analysis['frequency'][0]} (erkannt)")
    print(f"  Zeitschritte: {analysis['timesteps']} {analysis['frequency'][0].split()[0]}")

    print("\nGruppen:")
    if analysis['id_cols']:
        for col in analysis['id_cols']:
            print(f"  id_cols: {col} ({analysis['id_col_stats'][col]} unique)")
        print(f"  Kombinationen: {analysis['group_stats']['total_groups']:,} Gruppen")
    else:
        print("  Keine Gruppen (einzelne Zeitreihe)")

    print("\nGruppen-Längen (nach Split auf Train-Set):")
    gs = analysis['group_stats']
    print(f"  Min: {gs['min']} | Max: {gs['max']} | Median: {gs['median']} | Mean: {gs['mean']}")
    for threshold, count in gs['distribution'].items():
        pct = count / gs['total_groups'] * 100
        print(f"  {threshold:>6}: {count:>4} Gruppen ({pct:>5.1f}%)")

    print("\nTarget-Vorschläge:")
    for i, col in enumerate(analysis['target_suggestions'][:3], 1):
        markers = []
        if "sales" in col.lower() or "sold" in col.lower():
            markers.append("enthält 'sales'")
        markers.append("numerisch")
        print(f"  {i}. {col} ({', '.join(markers)})")

    print("\nTFT-Empfehlungen:")
    tft = analysis['tft_params']
    print(f"  max_encoder_length: {tft['max_encoder_length']} (basierend auf kürzester Zeitreihe)")
    print(f"  max_prediction_length: {tft['max_prediction_length']}")
    print(f"  lags: {tft['lags']} ({analysis['frequency'][0].split()[0]})")

    # Warnungen
    warnings = []

    # Zu kurze Zeitreihen
    short_threshold = tft['max_encoder_length'] + tft['max_prediction_length']
    short_groups = gs['distribution']['<40']
    if short_groups / gs['total_groups'] > 0.2:
        warnings.append(
            f"{short_groups:,} Gruppen ({short_groups / gs['total_groups'] * 100:.1f}%) haben <40 Zeitschritte (zu kurz)")

    # Hohe NaN-Raten
    high_nan_cols = {col: pct for col, pct in analysis['nan_stats'].items() if pct > 50}
    if high_nan_cols:
        nan_summary = ", ".join([f"{col}: {pct:.0f}%" for col, pct in list(high_nan_cols.items())[:3]])
        warnings.append(f"Hohe NaN-Raten: {nan_summary}")

    if warnings:
        print("\n⚠ Warnungen:")
        for w in warnings:
            print(f"  • {w}")

    print("\n✅ Vorgeschlagene Config erstellt: configs/datasets/{}_proposed.yaml".format(name))
    print("=" * 70 + "\n")


def analyze_and_propose(path: Path, name: Optional[str] = None) -> None:
    """Hauptfunktion: Analysiert Dataset und generiert YAML-Config."""

    # Name ableiten falls nicht gegeben
    if name is None:
        name = path.stem.replace("train", "").replace("_", "").strip() or path.parent.name

    # Daten laden
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Basis-Analyse
    time_col = detect_datetime_column(df)
    if time_col is None:
        raise ValueError("Keine Datetime-Spalte gefunden!")

    df[time_col] = pd.to_datetime(df[time_col])

    freq_name, freq_days = detect_frequency(df, time_col)
    id_cols = detect_id_columns(df, time_col)
    target_suggestions = suggest_target(df, time_col, id_cols)

    # Gruppen-Analyse
    group_stats = analyze_group_lengths(df, time_col, id_cols)

    # TFT-Parameter
    tft_params = calculate_tft_params(group_stats, freq_days)

    # NaN-Analyse
    nan_stats = analyze_nan_patterns(df)

    # Zusammenfassung
    analysis = {
        "rows": len(df),
        "cols": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 ** 2,
        "time_col": time_col,
        "date_range": (df[time_col].min().strftime("%Y-%m-%d"), df[time_col].max().strftime("%Y-%m-%d")),
        "frequency": (freq_name, freq_days),
        "timesteps": len(df[time_col].unique()),
        "id_cols": id_cols,
        "id_col_stats": {col: df[col].nunique() for col in id_cols},
        "target_suggestions": target_suggestions,
        "group_stats": group_stats,
        "tft_params": tft_params,
        "nan_stats": nan_stats,
    }

    # Report ausgeben
    print_report(name, analysis, path)

    # YAML generieren und speichern
    config = generate_yaml_config(name, analysis)
    output_path = Path(f"configs/datasets/{name}_proposed.yaml")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"Config gespeichert: {output_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Analysiert Datensätze und generiert YAML-Config")
    parser.add_argument("--path", type=str, required=True, help="Pfad zur CSV/Parquet-Datei")
    parser.add_argument("--name", type=str, help="Dataset-Name (optional, wird sonst abgeleitet)")

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    analyze_and_propose(path, args.name)


if __name__ == "__main__":
    main()

# Aufruf:
#   python -m src.data.analyze_dataset --path data/raw/walmart/train.csv
#   python -m src.data.analyze_dataset --path data/raw/walmart/train.csv --name walmart
#   python -m src.data.analyze_dataset --path data/raw/booksales/train.csv --name booksales