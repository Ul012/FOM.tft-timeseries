# src/data/data_alignment.py
# Zweck: Jahresmittel je (country, year) berechnen, 2017–2019 auf 2020-Niveau skalieren, als Parquet speichern.

from pathlib import Path
import numpy as np
import pandas as pd


# Projektroot automatisch bestimmen
BASE_DIR = Path(__file__).resolve().parents[2]
RAW = BASE_DIR / "data" / "raw" / "tabular-playground-series-sep-2022" / "train.csv"
OUT = BASE_DIR / "data" / "interim" / "train_aligned.parquet"


def align_yearly_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Skaliert num_sold pro (country, year) auf das 2020-Mittel.
    """
    out = df.copy()

    # Datums- & Jahresspalte
    if not pd.api.types.is_datetime64_any_dtype(out["date"]):
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["year"] = out["date"].dt.year

    # Jahresmittel je Land
    means = (
        out.groupby(["country", "year"], as_index=False)["num_sold"]
        .mean()
        .rename(columns={"num_sold": "mean_year"})
    )

    # 2020-Referenz je Land
    ref2020 = (
        means[means["year"] == 2020][["country", "mean_year"]]
        .rename(columns={"mean_year": "mean_2020"})
    )

    # Faktor = mean_2020 / mean_year  (2020 selbst → 1.0; ungültig/0 → 1.0)
    factors = means.merge(ref2020, on="country", how="left")
    factors["factor"] = factors["mean_2020"] / factors["mean_year"]

    # Bereinigen: Divisionen durch 0/NaN/Inf → 1.0; 2020 nie skalieren
    factors.loc[factors["year"] == 2020, "factor"] = 1.0
    invalid = (
        factors["mean_year"].isna()
        | factors["mean_2020"].isna()
        | (factors["mean_year"] == 0)
        | (factors["mean_2020"] == 0)
        | ~np.isfinite(factors["factor"])
    )
    factors.loc[invalid, "factor"] = 1.0

    # Faktor zurück zum Zeilenlevel
    out = out.merge(
        factors[["country", "year", "factor"]],
        on=["country", "year"],
        how="left",
        validate="many_to_one",
    )
    out["factor"] = out["factor"].fillna(1.0)

    # Skalieren
    out["num_sold"] = out["num_sold"] * out["factor"]
    out = out.drop(columns=["factor"])

    return out


def _print_sanity(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Kleine Plausibilisierung: Verhältnis (2017–2019) zu 2020 vor/nach Skalierung."""
    def _yr(df):
        d = df.copy()
        d["year"] = pd.to_datetime(d["date"]).dt.year
        return (
            d.groupby(["country", "year"])["num_sold"]
            .mean()
            .unstack("year")
            .filter([2017, 2018, 2019, 2020], axis=1)
        )

    b = _yr(before)
    a = _yr(after)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_before = b[[2017, 2018, 2019]].div(b[2020], axis=0)
        ratio_after = a[[2017, 2018, 2019]].div(a[2020], axis=0)

    print("\nSanity-Check (Durchschnitt num_sold Verhältnis zu 2020):")
    print("Vorher (soll ≠ 1.0 sein, typ. abweichend):")
    print(ratio_before.round(3).head())
    print("\nNachher (soll ≈ 1.0 sein):")
    print(ratio_after.round(3).head())


if __name__ == "__main__":
    # Einfache, pragmatische Pfade (kein Over-Engineering)
    RAW = Path("data/raw/tabular-playground-series-sep-2022/train.csv")
    OUT = Path("data/interim/train_aligned.parquet")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(RAW)
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")

    df_aligned = align_yearly_sales(df_raw)

    # Mini-Sanity (optional, schnell & hilfreich)
    _print_sanity(df_raw, df_aligned)

    # Parquet speichern
    df_aligned.to_parquet(OUT, index=False)
    print(f"\n✓ Gespeichert: {OUT}  (Zeilen: {len(df_aligned):,})")


# python -m src.data.data_alignment