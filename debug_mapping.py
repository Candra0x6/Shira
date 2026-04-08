import pandas as pd
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from shariah_features import ShariaFinancialFeatures

features = ShariaFinancialFeatures()
df = pd.read_csv("data/processed/companies_processed.csv")

for acc in features.FEATURE_ORDER[:8]:
    col = features.find_account_column(df, acc)
    print(f"Account: {acc:30} -> Column: {col}")
    if col and col in df.columns:
        val = df[col].iloc[0]
        print(f"  Sample value: {val}")
    else:
        print("  NOT FOUND")
