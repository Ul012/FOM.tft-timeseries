# src/data/data_alignment.py
# Zweck: Jahresmittel je (country, year) berechnen, 2017–2019 auf 2020-Niveau skalieren, als Parquet speichern.

import numpy as np
import pandas as pd

from src.config import RAW_DIR, INTERIM_DIR, BASE_DIR
from src.utils.load_dataset_config import load_dataset_config

_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]

# Rohdaten-Input und Output nach zentraler Config
RAW = BASE_DIR / "data" / "raw" / _dataset_name / "train.csv"
OUT = BASE_DIR / "data" / "interim" / _dataset_name / "train_aligned.parquet"


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

    # Faktor = mean_2020 / mean_year  (2020 selbst → 1.0; unbekannt → 1.0)
    means = means.merge(ref2020, on="country", how="left", validate="many_to_one")
    means["factor"] = np.where(
        means["year"] == 2020,
        1.0,
        means["mean_2020"] / means["mean_year"],
    )

    # Faktoren wieder an Originaldaten mergen
    factors = means[["country", "year", "factor"]]
    out = out.merge(
        factors,
        on=["country", "year"],
        how="left",
        validate="many_to_one",
    )
    out["factor"] = out["factor"].fillna(1.0)

    # Skalieren
    out["num_sold"] = out["num_sold"] * out["factor"]
    out = out.drop(columns=["factor"])

    return out


def _print_sanity(df_raw: pd.DataFrame, df_aligned: pd.DataFrame) -> None:
    """Kleine Sanity-Checks: Jahresmittel vorher/nachher."""
    def _year_means(df: pd.DataFrame, label: str) -> pd.DataFrame:
        tmp = (
            df.assign(year=df["date"].dt.year)
            .groupby(["country", "year"], as_index=False)["num_sold"]
            .mean()
        )
        tmp["variant"] = label
        return tmp

    raw_means = _year_means(df_raw, "raw")
    aligned_means = _year_means(df_aligned, "aligned")

    combined = (
        pd.concat([raw_means, aligned_means], ignore_index=True)
        .pivot_table(
            index=["country", "year"],
            columns="variant",
            values="num_sold",
        )
        .reset_index()
    )

    print("\n--- Jahresmittel num_sold pro Land/Jahr (raw vs. aligned) ---")
    print(combined.head(12).to_string(index=False))


def main() -> None:
    print(f"[data_alignment] Lade Rohdaten: {RAW}")
    df_raw = pd.read_csv(RAW)
    df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")

    df_aligned = align_yearly_sales(df_raw)

    # Mini-Sanity
    _print_sanity(df_raw, df_aligned)

    # Parquet speichern
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_aligned.to_parquet(OUT, index=False)
    print(f"\n✓ Gespeichert: {OUT}  (Zeilen: {len(df_aligned):,})")


if __name__ == "__main__":
    main()

# Aufruf einzeln:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.data_alignment
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.data_alignment
#
# Via Pipeline:
#   python -m src.pipeline --dataset configs/datasets/walmart.yaml --steps preprocessing
#   python -m src.pipeline --dataset configs/datasets/booksales.yaml --steps preprocessing