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


def detect_id_columns(df: pd.DataFrame, time_col: str) -> Tuple[List[str], List[Tuple[str, int]]]:
    """
    Identifiziert ID-Spalten konservativ.

    Nur String/Category-Spalten werden automatisch erkannt.
    Integer-Spalten werden als Kandidaten zur manuellen Prüfung zurückgegeben.

    Returns:
        (id_cols, integer_candidates) - erkannte IDs und Integer-Kandidaten
    """
    value_keywords = ["sales", "sold", "value", "amount", "revenue", "demand",
                      "price", "qty", "quantity", "count", "total", "sum"]

    id_cols = []
    integer_candidates = []

    for col in df.columns:
        if col == time_col:
            continue

        # Überspringe Spalten mit Target-Keywords
        if any(kw in col.lower() for kw in value_keywords):
            continue

        n_unique = df[col].nunique()

        # String/Category: sicher als ID
        if df[col].dtype in ["object", "category"]:
            if 1 < n_unique <= 100:
                id_cols.append(col)

        # Integer: als Kandidat merken (nicht automatisch)
        elif df[col].dtype in ["int64", "int32"]:
            if 1 < n_unique <= 100:
                integer_candidates.append((col, n_unique))

    return id_cols, integer_candidates


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
        "p10": int(lengths.quantile(0.10)),
        "p25": int(lengths.quantile(0.25)),
        "distribution": {
            "<20": int((lengths < 20).sum()),
            "<30": int((lengths < 30).sum()),
            "<40": int((lengths < 40).sum()),
            ">=40": int((lengths >= 40).sum()),
        },
        "total_groups": len(lengths),
    }


def validate_split_lengths(group_stats: Dict, tft_params: Dict,
                           split_ratios: List[float]) -> Dict[str, any]:
    """
    Validiert ob Val/Test-Splits lang genug für TFT-Anforderungen sind.

    Returns Dict mit Warnungen und Empfehlungen.
    """
    min_required = tft_params["max_encoder_length"] + tft_params["max_prediction_length"]

    # Berechne erwartete Split-Längen
    min_length = group_stats["min"]
    train_ratio, val_ratio, test_ratio = split_ratios

    expected_val_length = int(min_length * val_ratio)
    expected_test_length = int(min_length * test_ratio)

    issues = []
    recommendations = []

    # Val-Split Check
    if expected_val_length < min_required:
        issues.append(f"Val-Split zu kurz: {expected_val_length} < {min_required} Zeitschritte")
        recommendations.append(
            f"Lösungen: (1) Fixed-Date Split statt Ratio, oder (2) Split-Ratios anpassen (z.B. 0.6/0.2/0.2)")

    return {
        "min_required_length": min_required,
        "expected_val_length": expected_val_length,
        "expected_test_length": expected_test_length,
        "issues": issues,
        "recommendations": recommendations,
        "is_valid": len(issues) == 0,
    }


def calculate_tft_params(group_stats: Dict, freq_days: int) -> Dict:
    """
    Berechnet TFT-Parameter basierend auf Frequenz (praxisübliche Werte).

    Daumenregeln:
    - encoder_length: ~2 Saisonzyklen (genug Historie für Muster)
    - prediction_length: ~1 typischer Planungshorizont
    - Verhältnis encoder:prediction ≈ 6:1

    Typische Werte aus Literatur/Tutorials:
    - Täglich:    encoder=60 (2 Monate),    prediction=7 (1 Woche)
    - Wöchentlich: encoder=26 (6 Monate),   prediction=4 (1 Monat)
    - Monatlich:  encoder=24 (2 Jahre),     prediction=6 (6 Monate)
    """

    # Frequenzbasierte Standardwerte (Praxis-Erfahrungswerte)
    if freq_days == 1:  # täglich
        encoder_length = 60  # ~2 Monate Historie
        prediction_length = 7  # 1 Woche Vorhersage
        lags = [1, 7, 14, 30]
        roll_windows = [7, 14]
        freq_label = "täglich"
        encoder_desc = "~2 Monate"
        prediction_desc = "1 Woche"
    elif freq_days == 7:  # wöchentlich
        encoder_length = 26  # ~6 Monate Historie
        prediction_length = 4  # ~1 Monat Vorhersage
        lags = [1, 4, 8, 12]
        roll_windows = [4, 8]
        freq_label = "wöchentlich"
        encoder_desc = "~6 Monate"
        prediction_desc = "~1 Monat"
    elif freq_days == 30:  # monatlich
        encoder_length = 24  # 2 Jahre Historie
        prediction_length = 6  # 6 Monate Vorhersage
        lags = [1, 3, 6, 12]
        roll_windows = [3, 6]
        freq_label = "monatlich"
        encoder_desc = "2 Jahre"
        prediction_desc = "6 Monate"
    else:  # custom/unbekannt
        encoder_length = 60
        prediction_length = 10
        lags = [1, 4, 8]
        roll_windows = [4]
        freq_label = f"custom ({freq_days} Tage)"
        encoder_desc = f"{encoder_length} Zeitschritte"
        prediction_desc = f"{prediction_length} Zeitschritte"

    # min_group_length: Minimum um trainieren zu können
    min_group_length = encoder_length + prediction_length

    # Für Jahres-Lag (lag_365): Falls täglich, brauchen wir mindestens 400 Tage
    if freq_days == 1:
        min_group_length_with_yearly_lag = 400
    else:
        min_group_length_with_yearly_lag = min_group_length

    return {
        "max_encoder_length": encoder_length,
        "max_prediction_length": prediction_length,
        "min_group_length": min_group_length,
        "min_group_length_with_yearly_lag": min_group_length_with_yearly_lag,
        "lags": lags,
        "roll_windows": roll_windows,
        "freq_label": freq_label,
        "encoder_desc": encoder_desc,
        "prediction_desc": prediction_desc,
    }


def analyze_nan_patterns(df: pd.DataFrame) -> Dict[str, float]:
    """Analysiert NaN-Muster pro Spalte."""
    nan_pct = (df.isna().sum() / len(df) * 100).round(2)
    return {col: pct for col, pct in nan_pct.items() if pct > 0}


def analyze_data_quality(df: pd.DataFrame, target_col: str) -> Dict:
    """
    Prüft Datenqualität: Negative Werte, Inf, extreme Ausreißer.
    KRITISCH: Diese Prüfung verhindert 'Loss not finite' Crashes im Training!
    """
    import numpy as np

    issues = []

    # 1. Target-Analyse
    target_stats = {
        "min": float(df[target_col].min()),
        "max": float(df[target_col].max()),
        "mean": float(df[target_col].mean()),
        "std": float(df[target_col].std()),
        "negative_count": int((df[target_col] < 0).sum()),
        "zero_count": int((df[target_col] == 0).sum()),
    }

    # 2. Inf-Check
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_cols = {}
    for col in numeric_cols:
        n_inf = np.isinf(df[col]).sum()
        if n_inf > 0:
            inf_cols[col] = int(n_inf)
            issues.append(f"{col}: {n_inf} Inf-Werte")

    # 3. Extreme Ausreißer (> 99.9% Perzentil ist > 10x Mean)
    outlier_cols = {}
    p999 = df[target_col].quantile(0.999)
    mean = df[target_col].mean()
    if p999 > 10 * mean:
        ratio = p999 / mean
        outlier_cols[target_col] = {"p999": float(p999), "mean": float(mean), "ratio": float(ratio)}
        issues.append(f"{target_col}: Extreme Ausreißer (99.9% Perzentil = {ratio:.1f}x Mean)")

    # 4. Negative Werte bei Target
    if target_stats["negative_count"] > 0:
        pct = target_stats["negative_count"] / len(df) * 100
        issues.append(f"{target_col}: {target_stats['negative_count']} negative Werte ({pct:.2f}%)")

    return {
        "target_stats": target_stats,
        "inf_cols": inf_cols,
        "outlier_cols": outlier_cols,
        "issues": issues,
        "has_issues": len(issues) > 0,
    }


def generate_yaml_config(name: str, analysis: Dict) -> Dict:
    """Generiert YAML-Config aus Analyse-Ergebnissen."""
    time_col = analysis["time_col"]
    id_cols = analysis["id_cols"]
    target_col = analysis["target_suggestions"][0] if analysis["target_suggestions"] else "UNKNOWN"
    freq_name = analysis["frequency"][0]
    tft_params = analysis["tft_params"]

    # Verwende validierte Split-Ratios falls verfügbar
    split_ratios = [0.80, 0.10, 0.10]  # Default
    if not analysis["split_validation"]["is_valid"] and analysis["split_validation"]["recommendations"]:
        for rec in analysis["split_validation"]["recommendations"]:
            if "split ratios:" in rec:
                import re
                match = re.search(r'\[([\d.]+), ([\d.]+), ([\d.]+)\]', rec)
                if match:
                    split_ratios = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
                    break

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
                "min_group_length": tft_params["min_group_length_with_yearly_lag"],
            }},
        ],

        "split": {
            "method": "ratio",
            "ratios": split_ratios,
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


def print_report_continued(name: str, analysis: Dict, path: Path) -> None:
    """Druckt den Rest des Reports (nach interaktiver Abfrage)."""

    # Gruppen-Statistik
    gs = analysis['group_stats']
    print(f"\n  📊 Gruppen-Statistik:")
    print(f"     Anzahl Gruppen: {gs['total_groups']:,}")
    print(f"     Längen - Min: {gs['min']} | Max: {gs['max']} | Median: {gs['median']}")

    # Warnung für zu kurze Gruppen
    if gs.get('groups_below_min', 0) > 0:
        min_gl = gs.get('min_group_length_threshold', 30)
        print(f"     ⚠️  {gs['groups_below_min']} Gruppen haben < {min_gl} Zeitschritte")
        print(f"        → Ausschließen via: preprocessing.lag_features.params.min_group_length in YAML")

    print("\nTarget-Vorschläge:")
    for i, col in enumerate(analysis['target_suggestions'][:3], 1):
        markers = []
        if "sales" in col.lower() or "sold" in col.lower():
            markers.append("enthält 'sales'")
        markers.append("numerisch")
        print(f"  {i}. {col} ({', '.join(markers)})")

    print("\nTFT-Empfehlungen:")
    tft = analysis['tft_params']
    freq_label = tft.get('freq_label', analysis['frequency'][0])
    encoder_desc = tft.get('encoder_desc', '')
    prediction_desc = tft.get('prediction_desc', '')

    print(f"  Frequenz: {freq_label}")
    print(f"  max_encoder_length: {tft['max_encoder_length']} ({encoder_desc})")
    print(f"  max_prediction_length: {tft['max_prediction_length']} ({prediction_desc})")
    print(f"  min_group_length: {tft['min_group_length']} (encoder + prediction)")
    if freq_label == "täglich" and tft.get('min_group_length_with_yearly_lag'):
        print(f"  min_group_length (mit lag_365): {tft['min_group_length_with_yearly_lag']}")
    print(f"  lags: {tft['lags']}")
    print(f"  ")
    print(f"  Daumenregel: encoder ≈ 2 Saisonzyklen, prediction ≈ 1 Planungshorizont")

    # Split-Validierung
    sv = analysis['split_validation']
    print(f"\nSplit-Validierung (bei Standard-Split 0.8/0.1/0.1):")
    print(f"  Min erforderlich: {sv['min_required_length']} Zeitschritte (encoder + prediction)")
    print(f"  Erwarteter Val-Split: {sv['expected_val_length']} Zeitschritte")
    print(f"  Erwarteter Test-Split: {sv['expected_test_length']} Zeitschritte")

    if sv['is_valid']:
        print(f"  ✅ Splits sind ausreichend lang")
    else:
        print(f"  ❌ Splits NICHT ausreichend!")

    # Datenqualität
    dq = analysis['data_quality']
    print(f"\nDatenqualität:")
    ts = dq['target_stats']
    print(f"  Target ({analysis['target_suggestions'][0] if analysis['target_suggestions'] else 'N/A'}):")
    print(f"    Min: {ts['min']:.2f} | Max: {ts['max']:.2f} | Mean: {ts['mean']:.2f} | Std: {ts['std']:.2f}")
    print(f"    Negative: {ts['negative_count']} | Zero: {ts['zero_count']}")

    if dq['outlier_cols']:
        for col, stats in dq['outlier_cols'].items():
            print(
                f"    ⚠️  Extreme Ausreißer: 99.9% = {stats['ratio']:.1f}x Mean ({stats['p999']:.0f} vs {stats['mean']:.0f})")

    if dq['has_issues']:
        print(f"  ❌ {len(dq['issues'])} Datenqualitäts-Probleme gefunden!")
    else:
        print(f"  ✅ Keine kritischen Probleme")

    # Warnungen sammeln
    warnings = []

    # Datenqualitäts-Warnungen
    if dq['has_issues']:
        warnings.append("🚨 DATENQUALITÄT-PROBLEME (verhindern Training!):")
        for issue in dq['issues']:
            warnings.append(f"  → {issue}")
        warnings.append("  → Empfehlung: Bereinige Daten in dataset_tft.py (clip/filter)")

    # Split-Warnungen
    if not sv['is_valid']:
        for issue in sv['issues']:
            warnings.append(issue)
        for rec in sv['recommendations']:
            warnings.append(f"→ {rec}")

    # Zu kurze Zeitreihen
    gs = analysis['group_stats']
    short_groups = gs['distribution'].get('<40', 0)
    if gs['total_groups'] > 0 and short_groups / gs['total_groups'] > 0.2:
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


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Fragt den User nach Y/N Eingabe."""
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        response = input(f"{prompt} {suffix}: ").strip().lower()
        if response == "":
            return default
        return response in ["y", "yes", "ja", "j"]
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def ask_for_id_columns(integer_candidates: List[Tuple[str, int]], interactive: bool) -> List[str]:
    """
    Fragt interaktiv nach, welche Integer-Spalten Gruppen-Spalten sind.

    Returns:
        Liste der bestätigten ID-Spalten
    """
    if not integer_candidates:
        return []

    if not interactive:
        # Im nicht-interaktiven Modus: alle Kandidaten als ID annehmen
        return [col for col, _ in integer_candidates]

    print("\n  Mögliche Gruppen-Spalten gefunden:")
    for col, n_unique in integer_candidates:
        print(f"    - {col}: {n_unique} verschiedene Werte")
    print()

    confirmed_cols = []
    for col, n_unique in integer_candidates:
        if ask_yes_no(f"  Ist '{col}' eine Gruppen-Spalte (z.B. Store-ID, Produkt-ID)?", default=True):
            confirmed_cols.append(col)

    if confirmed_cols:
        print(f"\n  ✓ Gruppen-Spalten: {', '.join(confirmed_cols)}")
    else:
        print("\n  ✓ Keine Gruppen-Spalten ausgewählt (gesamter Datensatz = 1 Gruppe)")

    return confirmed_cols


def analyze_and_propose(path: Path, name: Optional[str] = None, interactive: bool = True) -> None:
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
    id_cols, integer_candidates = detect_id_columns(df, time_col)

    # Header ausgeben (vor interaktiver Abfrage)
    print("\n" + "=" * 70)
    print(f"Dataset-Analyse: {name}")
    print(f"Datei: {path}")
    print(
        f"Zeilen: {len(df):,} | Spalten: {len(df.columns)} | Memory: {df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

    print("\nZeitreihen-Eigenschaften:")
    print(
        f"  time_col: {time_col} ({df[time_col].min().strftime('%Y-%m-%d')} bis {df[time_col].max().strftime('%Y-%m-%d')})")
    print(f"  Frequenz: {freq_name} (erkannt)")
    print(f"  Zeitschritte: {len(df[time_col].unique())} {freq_name.split()[0]}")

    print("\nGruppen (schema.id_cols):")
    if id_cols:
        print(f"  Erkannt (String/Category): {', '.join(id_cols)}")
        for col in id_cols:
            print(f"    - {col}: {df[col].nunique()} unique Werte")
    else:
        print("  Keine String/Category-Spalten als ID erkannt.")

    # Interaktive Abfrage für Integer-Kandidaten
    confirmed_integer_cols = []
    if integer_candidates:
        confirmed_integer_cols = ask_for_id_columns(integer_candidates, interactive)

    # Finale ID-Spalten: String-IDs + bestätigte Integer-IDs
    final_id_cols = id_cols + confirmed_integer_cols

    # Target-Vorschläge aktualisieren mit finalen ID-Spalten
    target_suggestions = suggest_target(df, time_col, final_id_cols)

    # TFT-Parameter (brauchen wir für min_group_length)
    tft_params = calculate_tft_params({}, freq_days)

    # Gruppen-Analyse mit finalen ID-Spalten
    group_stats = analyze_group_lengths(df, time_col, final_id_cols)

    # Zähle Gruppen unter min_group_length
    min_gl = tft_params["min_group_length"]
    if final_id_cols:
        lengths = df.groupby(final_id_cols).size()
        groups_too_short = int((lengths < min_gl).sum())
        group_stats["groups_below_min"] = groups_too_short
        group_stats["min_group_length_threshold"] = min_gl

    # Split-Validierung
    # Nutze die GEFILTERTE Mindestlänge (nach Ausschluss zu kurzer Gruppen)
    # Die kürzeste Gruppe nach Filterung hat mindestens min_group_length Zeitschritte
    default_split_ratios = [0.80, 0.10, 0.10]

    # Erstelle eine Kopie der group_stats mit gefilterter Mindestlänge
    filtered_group_stats = group_stats.copy()
    min_gl = tft_params["min_group_length"]
    if group_stats["min"] < min_gl:
        # Nach Filterung ist die kürzeste Gruppe mindestens min_gl lang
        filtered_group_stats["min"] = min_gl

    split_validation = validate_split_lengths(filtered_group_stats, tft_params, default_split_ratios)

    # NaN-Analyse
    nan_stats = analyze_nan_patterns(df)

    # Datenqualitäts-Analyse (KRITISCH für TFT!)
    target_col = target_suggestions[0] if target_suggestions else None
    if target_col:
        data_quality = analyze_data_quality(df, target_col)
    else:
        data_quality = {"has_issues": False, "issues": [], "target_stats": {}, "inf_cols": {}, "outlier_cols": {}}

    # Zusammenfassung
    analysis = {
        "rows": len(df),
        "cols": len(df.columns),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024 ** 2,
        "time_col": time_col,
        "date_range": (df[time_col].min().strftime("%Y-%m-%d"), df[time_col].max().strftime("%Y-%m-%d")),
        "frequency": (freq_name, freq_days),
        "timesteps": len(df[time_col].unique()),
        "id_cols": final_id_cols,
        "integer_candidates": integer_candidates,
        "id_col_stats": {col: df[col].nunique() for col in final_id_cols},
        "target_suggestions": target_suggestions,
        "group_stats": group_stats,
        "tft_params": tft_params,
        "split_validation": split_validation,
        "nan_stats": nan_stats,
        "data_quality": data_quality,
    }

    # Rest des Reports ausgeben (ohne Header, der wurde schon ausgegeben)
    print_report_continued(name, analysis, path)

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
    parser.add_argument("--no-interactive", dest="interactive", action="store_false",
                        help="Keine interaktiven Rückfragen (für Automatisierung)")
    parser.set_defaults(interactive=True)

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")

    analyze_and_propose(path, args.name, interactive=args.interactive)


if __name__ == "__main__":
    main()

# Aufruf:
# Interaktiv (default)
#   python -m src.data.analyze_dataset --path data/raw/walmart/train.csv
#   python -m src.data.analyze_dataset --path data/raw/booksales/train.csv
#
# Ohne Rückfragen (alle Integer-Kandidaten als ID annehmen)
# python -m src.data.analyze_dataset --path data/raw/walmart/train.csv --no-interactive
