import pandas as pd
import numpy as np

# Loading data and analyzing features
df = pd.read_csv("global_gaming_esports_2010_2025.csv")
print("First 5 rows:")
print(df.head())
print("\nColumns:")
print(df.columns)

target = "Gaming_Revenue_BillionUSD"

nFeatures = [
    "Esports_Revenue_MillionUSD",
    "Active_Players_Million",
    "Esports_Viewers_Million",
    "Avg_Spending_USD",
    "Esports_Tournaments_Count",
    "Pro_Players_Count",
    "Internet_Penetration_Percent",
    "Avg_Latency_ms",
    "AR_VR_Adoption_Index",
    "Streaming_Influence_Index",
    "Covid_Impact_Index",
    "Female_Gamer_Percent",
    "Mobile_Gaming_Share",
    "Esports_PrizePool_MillionUSD",
    "Gaming_Companies_Count",
]

cFeats = ["Country", "Region", "Top_Genre", "Top_Platform"]
year = "Year"

# Relevant Categories
rColumns = [year, target] + nFeatures + cFeats
df = df[rColumns].copy()

for c in [target] + nFeatures:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop records with missing values
df = df.dropna(subset = [target] + nFeatures)
print("Cleaned data (first 5 rows):")
print(df.head())
print("\nShape after cleaning:", df.shape)

# Feature Engineering - Sort by Country/Year, get previous year revenue, find growth rate, drop countries w insufficient data
df = df.sort(["Country", year])
df["Lag_Revenue"] = df.groupby("Country")[target].shift(1)
df["YoY_Growth"] = df.groupby("Country")[target].pct_change()
df = df.dropna(subset = ["Lag_Revenue", "YoY_Growth"])


