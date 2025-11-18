import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

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

# Sort by country + year
df = df.sort_values(["Country", year])
# Previous Year Data
df["Lag_Revenue"] = df.groupby("Country")[target].shift(1)
# Calculate percentage change 
df["YoY_Growth"] = df.groupby("Country")[target].pct_change()
# Drop countries with no growth
df = df.dropna(subset = ["Lag_Revenue", "YoY_Growth"]) 

print("\nLag + Year over year growth: ")
print(df[[year, "Country", target, "Lag_Revenue", "YoY_Growth"]].head())

# Preparing Model + Train/Test
columnModels = (nFeatures + ["Lag_Revenue", "YoY_Growth", year] + cFeats)
X = df[columnModels].copy()
y = df[target].copy()

# Split: Training 80% | Testing 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) 
print("\nTrain Shape: ", X_train.shape, " Test Shape: ", X_test.shape)

# Preprocessing

# Numeric features that need to be scaled
nFeaturesModel = nFeatures + ["Lag_Revenue", "YoY_Growth", year] 
cFeaturesModel = cFeats

# One step pipeline that standardizes numeric features (mean = 0, std = 1)
nTransformer = Pipeline(steps = [("scalar", StandardScaler())])

# Converting categorical features to numeric format 
cTransformer = OneHotEncoder(handle_unknown = "ignore") 

# Applying transformations to features
preprocessor  = ColumnTransformer(transformers = [("numeric", nTransformer, nFeaturesModel), ("categorical", cTransformer, cFeaturesModel)]) #

# Setting up different models
models = {"Linear Regression": LinearRegression(), "Ridge Regression": Ridge(alpha = 1.0), "Random Forest": RandomForestRegressor(n_estimators = 200, max_depth = 10, random_state = 42)}

def eval(name, model):
    pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred) # R2 Score (Close to 1 = better)
    mae = mean_absolute_error(y_test, y_pred) # MAE Score 
    rmse = mean_squared_error(y_test, y_pred, squared = False)
    print(f"\n{name}")
    print(f"R^2: {r2:.5f}")
    print(f"MAE: {mae:.5f}")
    print(f"RMSE: {rmse:.5f}")
    return {"name": name, "pipeline" : pipe, "r2": r2}

results = []
for name, model in models.items():
    results.append(eval(name, model))
best = max(results, key=lambda d: d["r2"])
bestPipeline = best["pipeline"]
print(f"\nBest Model: {best['name']} (R2 = {best['r2']:.4f})")

# Best model for test set
y_predBest = bestPipeline.predict(X_test)

# Predicted vs Actual

plt.figure(figsize = (6, 6))
plt.scatter(y_test, y_predBest, alpha = 0.7)
minNum = min(y_test.min(), y_predBest.min())
maxNum = max(y_test.max(), y_predBest.max())

# Plotting Chart

plt.plot([minNum, maxNum], [minNum, maxNum], "r--")
plt.xlabel("Actual: " + target)
plt.ylabel("Predicted " + target)
plt.title(f"{best['name']}: Predicted vs Actual")
plt.tight_layout()
plt.show()
