from src.config import INTERIM_DIR
import pandas as pd

aligned = pd.read_parquet(INTERIM_DIR / "train_aligned.parquet")
cleaned = pd.read_parquet(INTERIM_DIR / "train_cleaned.parquet")

aligned["date"] = pd.to_datetime(aligned["date"])
cleaned["date"] = pd.to_datetime(cleaned["date"])

keys = ["date", "country", "store", "product"]
merged = aligned.merge(cleaned, on=keys, suffixes=("_aligned","_cleaned"))

mask_lockdown = (
    (merged["date"].dt.year == 2020)
    & (merged["date"].dt.month.isin([3,4,5]))
)

changed_lockdown = (
    mask_lockdown
    & merged["num_sold_cleaned"].notna()
    & (merged["num_sold_cleaned"] != merged["num_sold_aligned"])
)

print("Geänderte Zeilen im Lockdown:", changed_lockdown.sum())
