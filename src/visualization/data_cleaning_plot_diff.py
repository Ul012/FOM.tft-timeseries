# src/visualization/data_cleaning_plot_diff.py
# Visualisiert die Differenz (cleaned - aligned) für eine Serie mit sichtbarer Änderung

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import INTERIM_DIR


def main() -> None:
    aligned_path = INTERIM_DIR / "train_aligned.parquet"
    cleaned_path = INTERIM_DIR / "train_cleaned.parquet"

    df_aligned = pd.read_parquet(aligned_path)
    df_cleaned = pd.read_parquet(cleaned_path)

    # Datum normalisieren und auf 2020 filtern
    for df in (df_aligned, df_cleaned):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df_aligned_2020 = df_aligned[df_aligned["date"].dt.year == 2020]
    df_cleaned_2020 = df_cleaned[df_cleaned["date"].dt.year == 2020]

    # Innerer Merge auf Schlüsselspalten
    keys = ["date", "country", "store", "product"]
    merged = df_aligned_2020.merge(
        df_cleaned_2020,
        on=keys,
        suffixes=("_aligned", "_cleaned"),
        how="inner",
    )

    if merged.empty:
        raise RuntimeError("Keine gemeinsamen Datenpunkte für 2020 gefunden.")

    # Differenz berechnen
    merged["diff"] = merged["num_sold_cleaned"] - merged["num_sold_aligned"]

    # Gruppe mit größter absoluter Änderung finden
    grp_cols = ["country", "store", "product"]
    grp_stats = (
        merged.groupby(grp_cols)["diff"]
        .apply(lambda s: s.abs().sum())
        .reset_index(name="abs_change")
    )
    top = grp_stats.sort_values("abs_change", ascending=False).iloc[0]

    country = top["country"]
    store = top["store"]
    product = top["product"]

    print(f"[data_cleaning_diff] Zeige Serie mit größter Änderung: "
          f"{country}, store={store}, product={product}")

    # Nur diese Serie plotten
    sel = merged[
        (merged["country"] == country)
        & (merged["store"] == store)
        & (merged["product"] == product)
    ].sort_values("date")

    plt.figure(figsize=(18, 6))
    sns.lineplot(x=sel["date"], y=sel["diff"])
    plt.axhline(0, color="black", linewidth=1)
    plt.title(
        f"Differenz (cleaned - aligned) für {country}, store {store}, product {product}"
    )
    plt.ylabel("Differenz num_sold")
    plt.xlabel("Datum (2020)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # python -m src.visualization.data_cleaning_plot_diff
    main()
