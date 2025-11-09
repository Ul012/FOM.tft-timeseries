# src/data/view_data.py
# 👇 Lies einfach die CSV-Dateien und zeige einen Überblick.
from pathlib import Path
import pandas as pd

# Pfad **relativ zu dieser Datei**, damit er immer stimmt
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "tabular-playground-series-sep-2022"

train_path = DATA_DIR / "train.csv"
test_path  = DATA_DIR / "test.csv"

for p in (train_path, test_path):
    if not p.exists():
        raise FileNotFoundError(f"❌ Datei nicht gefunden: {p}")

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

# Falls keine Spalte 'date' existiert, einfach ignorieren
if "date" in train_df.columns:
    train_df["date"] = pd.to_datetime(train_df["date"], errors="coerce")

if "date" in test_df.columns:
    test_df["date"] = pd.to_datetime(test_df["date"], errors="coerce")

print("✅ Dateien geladen.")
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("\nSpaltennamen:", list(train_df.columns[:10]))

print("\nHead (5 Zeilen):")
with pd.option_context("display.max_columns", 20, "display.width", 200):
    print(train_df.head(5))

# python -m src.data.view_data