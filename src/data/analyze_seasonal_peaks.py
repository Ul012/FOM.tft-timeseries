"""
src/data/analyze_seasonal_peaks.py

Zeigt saisonale Muster in Zeitreihendaten.
Der Nutzer entscheidet selbst, welche Tage als Flag markiert werden.

Aufruf:
    $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.analyze_seasonal_peaks
    $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.analyze_seasonal_peaks

    # Mehr/weniger Tage anzeigen:
    python -m src.data.analyze_seasonal_peaks --top 30
"""

import argparse
from pathlib import Path
import pandas as pd

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mär", 4: "Apr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Dez"
}


def load_data(dataset_config: dict) -> pd.DataFrame:
    """Lädt Rohdaten."""
    dataset_name = dataset_config["name"]
    schema = get_schema(dataset_config)

    possible_paths = [
        BASE_DIR / "data" / "raw" / dataset_name / "train.csv",
        BASE_DIR / "data" / "raw" / dataset_name / "sales_train.csv",
    ]

    for f in dataset_config.get("raw_data", {}).get("files", []):
        if f.get("path"):
            possible_paths.append(BASE_DIR / f["path"])

    for path in possible_paths:
        if path and Path(path).exists():
            df = pd.read_csv(path, parse_dates=[schema["time_col"]])
            print(f"✓ Daten geladen: {path} ({len(df):,} Zeilen)")
            return df

    raise FileNotFoundError(f"Keine Rohdaten gefunden")


def detect_frequency(df: pd.DataFrame, date_col: str) -> str:
    """Erkennt ob täglich oder wöchentlich."""
    dates = df[date_col].drop_duplicates().sort_values()
    if len(dates) < 2:
        return "daily"
    median_diff = dates.diff().dropna().dt.days.median()
    return "weekly" if median_diff >= 6 else "daily"


def main() -> None:
    parser = argparse.ArgumentParser(description="Saisonale Peak-Analyse")
    parser.add_argument("--top", type=int, default=20, help="Anzahl Top/Bottom Tage")
    args = parser.parse_args()

    dataset_config = load_dataset_config()
    dataset_name = dataset_config["name"]
    schema = get_schema(dataset_config)
    target_col = schema["target_col"]
    date_col = schema["time_col"]

    print("=" * 60)
    print(f"SAISONALE ANALYSE: {dataset_name.upper()}")
    print("=" * 60)

    df = load_data(dataset_config)
    frequency = detect_frequency(df, date_col)
    print(f"Frequenz: {frequency}")

    if "IsHoliday" in df.columns:
        print("IsHoliday: ✓ bereits in Rohdaten vorhanden")

    # Aggregiere pro Datum
    df_daily = df.groupby(date_col)[target_col].sum().reset_index()
    df_daily["month"] = df_daily[date_col].dt.month
    df_daily["day"] = df_daily[date_col].dt.day

    if frequency == "weekly":
        df_daily["week"] = df_daily[date_col].dt.isocalendar().week

    overall_avg = df_daily[target_col].mean()

    # === MONATLICHE ÜBERSICHT ===
    print("\n" + "=" * 60)
    print("MONATLICHE ÜBERSICHT")
    print("=" * 60)

    monthly = df_daily.groupby("month")[target_col].mean().reset_index()
    monthly["pct"] = ((monthly[target_col] / overall_avg) - 1) * 100
    monthly = monthly.sort_values("month")

    print(f"\nØ Gesamt: {overall_avg:,.0f}\n")
    print(f"{'Monat':<8} {'Ø Wert':>12} {'vs. Ø':>10}")
    print("-" * 35)

    for _, row in monthly.iterrows():
        name = MONTH_NAMES[int(row["month"])]
        pct = row["pct"]
        marker = " ⭐⭐" if pct >= 30 else (" ⭐" if pct >= 15 else (" 📈" if pct >= 5 else ""))
        print(f"{name:<8} {row[target_col]:>12,.0f} {pct:>+9.1f}%{marker}")

    # === TOP TAGE/WOCHEN ===
    if frequency == "weekly":
        # Wöchentlich: nach KW gruppieren
        weekly = df_daily.groupby("week")[target_col].mean().reset_index()
        weekly["pct"] = ((weekly[target_col] / overall_avg) - 1) * 100

        print("\n" + "=" * 60)
        print(f"TOP {args.top} WOCHEN")
        print("=" * 60)

        top = weekly.nlargest(args.top, "pct")
        print(f"\n{'KW':<6} {'Ø Wert':>12} {'vs. Ø':>10}")
        print("-" * 35)

        for _, row in top.iterrows():
            pct = row["pct"]
            marker = " ⭐⭐" if pct >= 30 else (" ⭐" if pct >= 15 else "")
            print(f"KW {int(row['week']):<3} {row[target_col]:>12,.0f} {pct:>+9.1f}%{marker}")
    else:
        # Täglich: nach Monat+Tag gruppieren
        daily_avg = df_daily.groupby(["month", "day"])[target_col].mean().reset_index()
        daily_avg["pct"] = ((daily_avg[target_col] / overall_avg) - 1) * 100

        print("\n" + "=" * 60)
        print(f"TOP {args.top} TAGE")
        print("=" * 60)

        top = daily_avg.nlargest(args.top, "pct")
        print(f"\n{'Datum':<10} {'Ø Wert':>12} {'vs. Ø':>10}")
        print("-" * 38)

        for _, row in top.iterrows():
            date_str = f"{int(row['day']):02d}. {MONTH_NAMES[int(row['month'])]}"
            pct = row["pct"]
            marker = " ⭐⭐" if pct >= 30 else (" ⭐" if pct >= 15 else "")
            print(f"{date_str:<10} {row[target_col]:>12,.0f} {pct:>+9.1f}%{marker}")

        print("\n" + "=" * 60)
        print(f"BOTTOM {min(args.top, 10)} TAGE")
        print("=" * 60)

        bottom = daily_avg.nsmallest(min(args.top, 10), "pct")
        print(f"\n{'Datum':<10} {'Ø Wert':>12} {'vs. Ø':>10}")
        print("-" * 38)

        for _, row in bottom.iterrows():
            date_str = f"{int(row['day']):02d}. {MONTH_NAMES[int(row['month'])]}"
            pct = row["pct"]
            print(f"{date_str:<10} {row[target_col]:>12,.0f} {pct:>+9.1f}%")

    print("\n" + "=" * 60)
    print("LEGENDE: ⭐⭐ = >30% | ⭐ = >15% | 📈 = >5%")
    print("=" * 60)


if __name__ == "__main__":
    main()

# Aufruf:
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.analyze_seasonal_peaks_v2
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.analyze_seasonal_peaks_v2