import pandas as pd
import numpy as np
import pickle
import os

project_root = os.path.dirname(os.path.abspath(__file__))
model_path = f"{project_root}/models/final_trained_model.pkl"
scaler_path = f"{project_root}/models/feature_scaler.pkl"
features_csv = f"{project_root}/data/processed/companies_with_features.csv"

# Load
with open(model_path, "rb") as f:
    model = pickle.load(f)
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv(features_csv)
feature_cols = [
    "total_assets", "total_liabilities", "total_equity", "net_revenue",
    "nonhalal_revenue_percent", "net_income", "operating_cash_flow", "interest_expense",
    "debt_to_equity", "debt_to_assets", "roe", "roa", "profit_margin",
    "interest_coverage", "cash_flow_to_debt", "f_riba", "f_nonhalal",
    "riba_intensity", "sector_encoded"
]

X = df[feature_cols].fillna(0)
X_scaled = scaler.transform(X)

# Predict
probs = model.predict_proba(X_scaled)[:, 1]
preds = model.predict(X_scaled)

print(f"Total samples: {len(probs)}")
print(f"Probabilities Min: {probs.min():.4f}")
print(f"Probabilities Max: {probs.max():.4f}")
print(f"Probabilities Mean: {probs.mean():.4f}")
print(f"Predictions (0: Compliant, 1: Non-compliant):")
print(pd.Series(preds).value_counts())

# Check top 10 probs
print("\nTop 10 Probabilities:")
print(np.sort(probs)[-10:])

# Check bottom 10 probs
print("\nBottom 10 Probabilities:")
print(np.sort(probs)[:10])
