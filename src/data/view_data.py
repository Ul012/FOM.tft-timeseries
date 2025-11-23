# src/data/view_data.py
# 👇 Lies einfach die CSV-Dateien und zeige einen Überblick.
from pathlib import Path
import pandas as pd

from src.config import BASE_DIR
from src.utils.load_dataset_config import load_dataset_config, get_schema

_dataset_config = load_dataset_config()
_dataset_name = _dataset_config["name"]
_schema = get_schema(_dataset_config)
TIME_COL = _schema["time_col"]

DATA_DIR = BASE_DIR / "data" / "raw" / _dataset_name

train_path = DATA_DIR / "train.csv"
test_path  = DATA_DIR / "test.csv"

for p in (train_path, test_path):
    if not p.exists():
        raise FileNotFoundError(f"❌ Datei nicht gefunden: {p}")

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

# Zeitkolumne in datetime konvertieren (falls vorhanden)
if TIME_COL in train_df.columns:
    train_df[TIME_COL] = pd.to_datetime(train_df[TIME_COL], errors="coerce")

if TIME_COL in test_df.columns:
    test_df[TIME_COL] = pd.to_datetime(test_df[TIME_COL], errors="coerce")

print("✅ Dateien geladen.")
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("\nSpaltennamen:", list(train_df.columns[:10]))

print("\nHead (5 Zeilen):")
with pd.option_context("display.max_columns", 20, "display.width", 200):
    print(train_df.head(5))

# Aufruf:
#   $env:DATASET_CONFIG="configs/datasets/walmart.yaml"; python -m src.data.view_data
#   $env:DATASET_CONFIG="configs/datasets/booksales.yaml"; python -m src.data.view_data