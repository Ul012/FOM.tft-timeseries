"""
src/data/analyze_seasonal_peaks_detailed.py

Automatische Erkennung von saisonalen Peaks in Zeitreihendaten.
Funktioniert datensatzübergreifend (booksales, walmart, etc.)

Aufruf:
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.analyze_seasonal_peaks_detailed
    $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.analyze_seasonal_peaks_detailed

    # Mit angepassten Schwellwerten:
    python -m src.data.analyze_seasonal_peaks --elevated-threshold 10 --peak-threshold 25
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema


# -----------------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------------

@dataclass
class PeakConfig:
    """Schwellwerte für Peak-Erkennung."""
    elevated_threshold: float = 15.0
    peak_threshold: float = 30.0
    min_consecutive_days: int = 2


MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mär", 4: "Apr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dez"
}

DAYS_IN_MONTH = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
    7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
}


# -----------------------------------------------------------------------------
# Daten laden
# -----------------------------------------------------------------------------

def load_data(dataset_config: dict) -> pd.DataFrame:
    """Lädt Rohdaten basierend auf Dataset-Config."""
    dataset_name = dataset_config["name"]
    schema = get_schema(dataset_config)

    possible_paths = [
        BASE_DIR / "data" / "raw" / dataset_name / "train.csv",
        BASE_DIR / "data" / "raw" / dataset_name / "sales_train.csv",
    ]

    raw_data = dataset_config.get("raw_data", {})
    files = raw_data.get("files", [])
    if files:
        for f in files:
            path = f.get("path", "")
            if path:
                possible_paths.append(BASE_DIR / path)

    for path in possible_paths:
        if path and Path(path).exists():
            df = pd.read_csv(path, parse_dates=[schema["time_col"]])
            print(f"✓ Daten geladen: {path}")
            print(f"  Zeilen: {len(df):,}")
            return df

    raise FileNotFoundError(f"Keine Rohdaten gefunden für {dataset_name}")


def detect_frequency(df: pd.DataFrame, date_col: str) -> str:
    """Erkennt ob täglich oder wöchentlich."""
    dates = df[date_col].drop_duplicates().sort_values()
    if len(dates) < 2:
        return "daily"
    diffs = dates.diff().dropna().dt.days
    median_diff = diffs.median()
    return "weekly" if median_diff >= 6 else "daily"


# -----------------------------------------------------------------------------
# Analyse
# -----------------------------------------------------------------------------

def analyze_daily_averages(df: pd.DataFrame, date_col: str, target_col: str) -> Tuple[pd.DataFrame, float]:
    """Berechnet durchschnittliche Verkäufe pro Kalendertag."""
    df_daily = df.groupby(date_col)[target_col].sum().reset_index()
    df_daily["month"] = df_daily[date_col].dt.month
    df_daily["day"] = df_daily[date_col].dt.day
    df_daily["year"] = df_daily[date_col].dt.year

    df_avg = df_daily.groupby(["month", "day"])[target_col].agg(["mean", "std", "count"]).reset_index()
    df_avg.columns = ["month", "day", "avg_sales", "std_sales", "n_years"]

    overall_avg = df_daily[target_col].mean()
    df_avg["pct_vs_avg"] = ((df_avg["avg_sales"] / overall_avg) - 1) * 100

    # Sortierung: Jahreswechsel berücksichtigen (Nov, Dez, Jan, Feb, ...)
    # Trick: Dezember = -1, November = -2, sonst normal
    df_avg["sort_key"] = df_avg["month"].apply(lambda m: m if m < 11 else m - 14) * 100 + df_avg["day"]
    df_avg = df_avg.sort_values("sort_key").reset_index(drop=True)

    return df_avg, overall_avg


def analyze_weekly_averages(df: pd.DataFrame, date_col: str, target_col: str) -> Tuple[pd.DataFrame, float]:
    """Berechnet durchschnittliche Verkäufe pro Kalenderwoche."""
    df_weekly = df.groupby(date_col)[target_col].sum().reset_index()
    df_weekly["week"] = df_weekly[date_col].dt.isocalendar().week
    df_weekly["year"] = df_weekly[date_col].dt.year

    df_avg = df_weekly.groupby("week")[target_col].agg(["mean", "std", "count"]).reset_index()
    df_avg.columns = ["week", "avg_sales", "std_sales", "n_years"]

    overall_avg = df_weekly[target_col].mean()
    df_avg["pct_vs_avg"] = ((df_avg["avg_sales"] / overall_avg) - 1) * 100
    df_avg = df_avg.sort_values("week").reset_index(drop=True)

    return df_avg, overall_avg


def find_peak_periods_daily(df_avg: pd.DataFrame, config: PeakConfig) -> List[Tuple[int, int, int, int, float]]:
    """Findet zusammenhängende Peak-Perioden (täglich)."""
    df_avg = df_avg.copy()
    df_avg["is_peak"] = df_avg["pct_vs_avg"] >= config.peak_threshold

    periods = []
    current_period = None

    for idx, row in df_avg.iterrows():
        if row["is_peak"]:
            curr_month = int(row["month"])
            curr_day = int(row["day"])

            if current_period is None:
                current_period = {
                    "start_month": curr_month,
                    "start_day": curr_day,
                    "end_month": curr_month,
                    "end_day": curr_day,
                    "pct_values": [row["pct_vs_avg"]],
                }
            else:
                prev_month = current_period["end_month"]
                prev_day = current_period["end_day"]

                # Prüfe Konsekutivität (inkl. Jahreswechsel Dez→Jan)
                is_consecutive = (
                        (curr_month == prev_month and curr_day == prev_day + 1) or
                        (curr_month == prev_month + 1 and prev_day >= 28 and curr_day == 1) or
                        (prev_month == 12 and curr_month == 1 and prev_day == 31 and curr_day <= 7) or
                        (prev_month == 12 and curr_month == 1 and prev_day >= 28 and curr_day == 1)
                )

                if is_consecutive:
                    current_period["end_month"] = curr_month
                    current_period["end_day"] = curr_day
                    current_period["pct_values"].append(row["pct_vs_avg"])
                else:
                    # Speichere alte Periode
                    if len(current_period["pct_values"]) >= config.min_consecutive_days:
                        avg_pct = np.mean(current_period["pct_values"])
                        periods.append((
                            current_period["start_month"],
                            current_period["start_day"],
                            current_period["end_month"],
                            current_period["end_day"],
                            avg_pct,
                        ))
                    # Starte neue Periode
                    current_period = {
                        "start_month": curr_month,
                        "start_day": curr_day,
                        "end_month": curr_month,
                        "end_day": curr_day,
                        "pct_values": [row["pct_vs_avg"]],
                    }
        else:
            if current_period is not None:
                if len(current_period["pct_values"]) >= config.min_consecutive_days:
                    avg_pct = np.mean(current_period["pct_values"])
                    periods.append((
                        current_period["start_month"],
                        current_period["start_day"],
                        current_period["end_month"],
                        current_period["end_day"],
                        avg_pct,
                    ))
                current_period = None

    # Letzte Periode abschließen
    if current_period is not None and len(current_period["pct_values"]) >= config.min_consecutive_days:
        avg_pct = np.mean(current_period["pct_values"])
        periods.append((
            current_period["start_month"],
            current_period["start_day"],
            current_period["end_month"],
            current_period["end_day"],
            avg_pct,
        ))

    return periods


def find_peak_weeks(df_avg: pd.DataFrame, config: PeakConfig) -> List[Tuple[int, float]]:
    """Findet Peak-Wochen (wöchentlich)."""
    peaks = []
    for _, row in df_avg.iterrows():
        if row["pct_vs_avg"] >= config.peak_threshold:
            peaks.append((int(row["week"]), row["pct_vs_avg"]))
    return peaks


def calc_duration(start_m: int, start_d: int, end_m: int, end_d: int) -> int:
    """Berechnet Dauer in Tagen."""
    if start_m == end_m:
        return end_d - start_d + 1

    # Jahreswechsel
    if start_m == 12 and end_m == 1:
        return (31 - start_d + 1) + end_d

    days = DAYS_IN_MONTH[start_m] - start_d + 1
    m = start_m + 1
    while m != end_m:
        if m > 12:
            m = 1
        days += DAYS_IN_MONTH[m]
        m += 1
        if m > 12:
            m = 1
    days += end_d
    return days


# -----------------------------------------------------------------------------
# Ausgabe
# -----------------------------------------------------------------------------

def print_monthly_summary(df_avg: pd.DataFrame, overall_avg: float, target_col: str) -> None:
    """Druckt monatliche Zusammenfassung."""
    print("\n" + "=" * 70)
    print("MONATLICHE ÜBERSICHT")
    print("=" * 70)

    monthly = df_avg.groupby("month").agg({"avg_sales": "mean", "pct_vs_avg": "mean"}).reset_index()
    monthly = monthly.sort_values("month")

    print(f"\nGesamtdurchschnitt (täglich): {overall_avg:,.0f} {target_col}")
    print(f"\n{'Monat':<10} {'Ø Verkäufe':>12} {'vs. Gesamt-Ø':>12}")
    print("-" * 40)

    for _, row in monthly.iterrows():
        month_name = MONTH_NAMES.get(int(row["month"]), str(row["month"]))
        if row["pct_vs_avg"] > 15:
            marker = " ⭐"
        elif row["pct_vs_avg"] > 5:
            marker = " 📈"
        elif row["pct_vs_avg"] < -10:
            marker = " 📉"
        else:
            marker = ""
        print(f"{month_name:<10} {row['avg_sales']:>12,.0f} {row['pct_vs_avg']:>+11.1f}%{marker}")


def print_weekly_summary(df_avg: pd.DataFrame, overall_avg: float, target_col: str) -> None:
    """Druckt wöchentliche Zusammenfassung."""
    print("\n" + "=" * 70)
    print("WÖCHENTLICHE ÜBERSICHT (Top 10)")
    print("=" * 70)

    print(f"\nGesamtdurchschnitt (wöchentlich): {overall_avg:,.0f} {target_col}")
    print(f"\n{'KW':<6} {'Ø Verkäufe':>14} {'vs. Ø':>10}")
    print("-" * 35)

    top = df_avg.nlargest(10, "pct_vs_avg")
    for _, row in top.iterrows():
        marker = " ⭐⭐" if row["pct_vs_avg"] >= 30 else (" ⭐" if row["pct_vs_avg"] >= 15 else "")
        print(f"KW {int(row['week']):<3} {row['avg_sales']:>14,.0f} {row['pct_vs_avg']:>+9.0f}%{marker}")


def print_daily_peaks(df_avg: pd.DataFrame, config: PeakConfig) -> None:
    """Druckt tägliche Peak-Details mit Sternchen."""
    print("\n" + "=" * 70)
    print(f"PEAK-TAGE (>{config.elevated_threshold}% über Durchschnitt)")
    print("=" * 70)

    df_filtered = df_avg[df_avg["pct_vs_avg"] >= config.elevated_threshold].copy()
    # Sortiere nach Monat, Tag (normal, nicht Jahreswechsel)
    df_filtered = df_filtered.sort_values(["month", "day"])

    if len(df_filtered) == 0:
        print(f"\nKeine Tage mit >{config.elevated_threshold}% gefunden.")
        return

    print(f"\n{'Datum':<12} {'Ø Verkäufe':>12} {'vs. Ø':>10} {'Stufe':>10}")
    print("-" * 50)

    for _, row in df_filtered.iterrows():
        month_name = MONTH_NAMES.get(int(row["month"]), str(row["month"]))
        date_str = f"{int(row['day']):02d}. {month_name}"

        if row["pct_vs_avg"] >= config.peak_threshold:
            marker = "⭐⭐ (2)"
        else:
            marker = "⭐ (1)"

        print(f"{date_str:<12} {row['avg_sales']:>12,.0f} {row['pct_vs_avg']:>+9.0f}% {marker:>10}")


def print_peak_periods(periods: List, config: PeakConfig) -> None:
    """Druckt erkannte Peak-Perioden."""
    print("\n" + "=" * 70)
    print("ERKANNTE PEAK-PERIODEN")
    print("=" * 70)

    if not periods:
        print(f"\nKeine zusammenhängenden Perioden mit >{config.peak_threshold}% gefunden.")
        return

    print(f"\nSchwellwert: >{config.peak_threshold}%\n")

    for start_m, start_d, end_m, end_d, avg_pct in periods:
        start_str = f"{start_d:02d}. {MONTH_NAMES[start_m]}"
        end_str = f"{end_d:02d}. {MONTH_NAMES[end_m]}"
        duration = calc_duration(start_m, start_d, end_m, end_d)
        print(f"  {start_str} - {end_str} ({duration} Tage, Ø {avg_pct:+.0f}%)")


def determine_flag_name(periods: List) -> str:
    """Bestimmt passenden Flag-Namen basierend auf Zeitraum."""
    if not periods:
        return "is_seasonal_peak"

    main_peak = max(periods, key=lambda x: x[4])
    start_m, _, end_m, _, _ = main_peak

    if start_m == 12 or end_m == 1:
        return "is_newyear"
    elif start_m == 11 and end_m == 11:
        return "is_thanksgiving"
    elif start_m in [6, 7, 8]:
        return "is_summer_peak"
    else:
        return "is_seasonal_peak"


def print_final_recommendation_daily(dataset_name: str, periods: List, config: PeakConfig) -> None:
    """Druckt finale Empfehlung für tägliche Daten."""
    print("\n")
    print("=" * 70)
    print("█ EMPFEHLUNG")
    print("=" * 70)

    if not periods:
        print(f"""
Keine signifikanten Peaks (>{config.peak_threshold}%) gefunden.

➜ Kein zusätzliches date_flag nötig.
""")
        return

    flag_name = determine_flag_name(periods)

    # Generiere YAML date_flags Format
    yaml_periods = []
    for start_m, start_d, end_m, end_d, _ in periods:
        if start_m == end_m:
            yaml_periods.append(f"        - {{month: {start_m}, day_start: {start_d}, day_end: {end_d}}}")
        elif start_m == 12 and end_m == 1:
            # Jahreswechsel: zwei Einträge
            yaml_periods.append(f"        - {{month: 12, day_start: {start_d}, day_end: 31}}")
            yaml_periods.append(f"        - {{month: 1, day_start: 1, day_end: {end_d}}}")
        else:
            yaml_periods.append(
                f"        - {{month: {start_m}, day_start: {start_d}, day_end: {DAYS_IN_MONTH[start_m]}}}")
            yaml_periods.append(f"        - {{month: {end_m}, day_start: 1, day_end: {end_d}}}")

    yaml_block = "\n".join(yaml_periods)

    print(f"""
────────────────────────────────────────────────────────────────────────
SCHRITT 1: {dataset_name}.yaml anpassen
────────────────────────────────────────────────────────────────────────
Datei: configs/datasets/{dataset_name}.yaml

In preprocessing > feature_engineering > params hinzufügen:

    date_flags:
      {flag_name}:
{yaml_block}

In tft > flag_cols hinzufügen:

    flag_cols: ["{flag_name}"]

────────────────────────────────────────────────────────────────────────
SCHRITT 2: Preprocessing neu durchführen
────────────────────────────────────────────────────────────────────────
$env:DATASET_CONFIG="configs/datasets/{dataset_name}.yaml"
python -m src.pipeline --dataset configs/datasets/{dataset_name}.yaml --steps preprocessing,model_dataset,dataset_tft
""")


def print_final_recommendation_weekly(dataset_name: str, peak_weeks: List, config: PeakConfig,
                                      has_isholiday: bool) -> None:
    """Druckt finale Empfehlung für wöchentliche Daten."""
    print("\n")
    print("=" * 70)
    print("█ EMPFEHLUNG")
    print("=" * 70)

    if has_isholiday:
        print(f"""
Das Dataset hat bereits ein "IsHoliday" Feature in den Rohdaten.

➜ Kein zusätzliches date_flag nötig.

────────────────────────────────────────────────────────────────────────
NUR PRÜFEN: {dataset_name}.yaml
────────────────────────────────────────────────────────────────────────
Datei: configs/datasets/{dataset_name}.yaml

In tft > flag_cols sicherstellen:

    flag_cols: ["IsHoliday"]
""")
    else:
        weeks_str = ", ".join(str(w) for w, _ in sorted(peak_weeks, key=lambda x: x[1], reverse=True)[:5])
        print(f"""
Peak-Wochen gefunden: KW {weeks_str}

➜ Wöchentliche date_flags werden aktuell nicht unterstützt.
   Empfehlung: Manuell "is_holiday_week" Feature hinzufügen oder
   prüfen ob IsHoliday in Rohdaten verfügbar ist.
""")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Saisonale Peak-Analyse")
    parser.add_argument("--elevated-threshold", type=float, default=15.0)
    parser.add_argument("--peak-threshold", type=float, default=30.0)
    parser.add_argument("--min-days", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PeakConfig(
        elevated_threshold=args.elevated_threshold,
        peak_threshold=args.peak_threshold,
        min_consecutive_days=args.min_days,
    )

    dataset_config = load_dataset_config()
    dataset_name = dataset_config["name"]
    schema = get_schema(dataset_config)
    target_col = schema["target_col"]
    date_col = schema["time_col"]

    print("=" * 70)
    print(f"SAISONALE PEAK-ANALYSE: {dataset_name.upper()}")
    print("=" * 70)

    df = load_data(dataset_config)
    frequency = detect_frequency(df, date_col)
    print(f"  Frequenz: {frequency}")

    has_isholiday = "IsHoliday" in df.columns
    if has_isholiday:
        print(f"  IsHoliday Feature: ✓ vorhanden")

    if frequency == "weekly":
        df_avg, overall_avg = analyze_weekly_averages(df, date_col, target_col)
        print_weekly_summary(df_avg, overall_avg, target_col)
        peak_weeks = find_peak_weeks(df_avg, config)
        print_final_recommendation_weekly(dataset_name, peak_weeks, config, has_isholiday)
    else:
        df_avg, overall_avg = analyze_daily_averages(df, date_col, target_col)
        print_monthly_summary(df_avg, overall_avg, target_col)
        print_daily_peaks(df_avg, config)
        periods = find_peak_periods_daily(df_avg, config)
        print_peak_periods(periods, config)
        print_final_recommendation_daily(dataset_name, periods, config)


if __name__ == "__main__":
    main()

# Aufruf:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.analyze_seasonal_peaks_detailed
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.analyze_seasonal_peaks_detailed