import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

df = pd.read_csv('survey_results_slim.csv')

#Dropping uneccesary columns
missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending = False)
cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
cols_to_drop.extend([
    'LearnCodeOnline', 'OpSysPersonal use', 'ICorPM', 'ProfessionalTech', 'TBranch', 'AISearchHaveWorkedWith', 'MiscTechHaveWorkedWith',
    'OfficeStackAsyncHaveWorkedWith', 'OfficeStackSyncHaveWorkedWith', 'NEWCollabToolsHaveWorkedWith'])

cols_to_drop = list(set(cols_to_drop))
df_cleaned = df.drop(columns = cols_to_drop)
print(f"Dropped {len(cols_to_drop)} columns")
print (f"Remaining Colmns: {df_cleaned.shape[1]}")
print(f"\nRemaining Columns:\n{df_cleaned.columns}")

print("\nData Types:")
print(df_cleaned.dtypes.value_counts())

print("\nUnique Values In Key Columns:")
for col in ['Country', 'DevType', 'EdLevel', 'Employment']:
    if col in df_cleaned.columns:
        print(f"{col}: {df_cleaned[col].nunique()} unique values")
        
df_model = df_cleaned.dropna().copy()
print(f"Final Dataset Shape: {df_model.shape}")
print(f"Rows Retained: {len(df_model)} ({len(df_model)/48019*100:.1f}% of filtered data.)")

print("\nSalary Statistics: ")
print(df_model['ConvertedCompYearly'].describe())
print(f"\nSalaries > $500k: {(df_model['ConvertedCompYearly'] > 500000).sum()}")
print(f"\nSalaries < $10k: {(df_model['ConvertedCompYearly'] < 10000).sum()}")