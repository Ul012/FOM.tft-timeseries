# src/data/view_data_plot.py
"""
Visualisiert tägliche Verkaufszahlen pro Buchprodukt (Book Sales Dataset).
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Datenpfad wie bisher (Jan-2022 Dataset!)
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "tabular-playground-series-sep-2022"

train_df = pd.read_csv(DATA_DIR / "train.csv")
train_df["date"] = pd.to_datetime(train_df["date"])

# Gruppieren
daily_sales_product = train_df.groupby(["date", "product"], as_index=False)["num_sold"].sum()
daily_sales_store = train_df.groupby(["date", "store"], as_index=False)["num_sold"].sum()
daily_sales_country = train_df.groupby(["date", "country"], as_index=False)["num_sold"].sum()

# --- Product ---
fig, ax = plt.subplots(figsize=(18, 6))
sns.lineplot(x="date", y="num_sold", hue="product", data=daily_sales_product, ax=ax)
ax.set_title("Daily total sales per product")
plt.tight_layout()
plt.show()

# --- Store ---
fig, ax = plt.subplots(figsize=(18, 6))
sns.lineplot(x="date", y="num_sold", hue="store", data=daily_sales_store, ax=ax)
ax.set_title("Daily total sales per store")
plt.tight_layout()
plt.show()

# --- Country ---
fig, ax = plt.subplots(figsize=(18, 6))
sns.lineplot(x="date", y="num_sold", hue="country", data=daily_sales_country, ax=ax)
ax.set_title("Daily total sales per country")
plt.tight_layout()
plt.show()
