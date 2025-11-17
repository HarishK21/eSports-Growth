import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

# Feature Engineering
df = df.sort_values(["Country", year]) # Sort by country + year
df["Lag_Revenue"] = df.groupby("Country")[target].shift(1) # Previous Year Data
df["YoY_Growth"] = df.groupby("Country")[target].pct_change() # Calculate percentage change
df = df.dropna(subset = ["Lag_Revenue", "YoY_Growth"]) # Drop countries with no growth

print("\nLag + Year over year growth: ")
print(df[[year, "Country", target, "Lag_Revenue", "YoY_Growth"]].head())

# Preparing Model + Train/Test
columnModels = (nFeatures + ["Lag_Revenue", "YoY_Growth", year] + cFeats)
X = df[columnModels].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) # Split: Training 80% | Testing 20%
print("\nTrain Shape: ", X_train.shape, " Test Shape: ", X_test.shape)