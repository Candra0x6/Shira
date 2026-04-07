#!/bin/bash

################################################################################
# Quick Test Script - Run Model with Sample Data & Show Results
# 
# Usage: ./quick_test.sh
#
################################################################################

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}\n"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 not found"
    exit 1
fi

print_header "🚀 Shariah Compliance Model - Quick Test"

log_step "Checking required data files..."
if [ ! -f "$PROJECT_DIR/data/processed/companies_with_features.csv" ]; then
    log_error "Data file not found! Run './run.sh' first to generate data."
    exit 1
fi
log_success "Data file found"

log_step "Checking trained model..."
if [ ! -f "$PROJECT_DIR/models/xgb_shariah_model.pkl" ]; then
    log_error "Model file not found! Run './run.sh' first to train model."
    exit 1
fi
log_success "Model file found"

log_step "Running quick predictions on sample data..."

python3 << ENDPYTHON
import sys
import os
sys.path.insert(0, '$PROJECT_DIR/src')
os.chdir('$PROJECT_DIR')

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load features and labels
print("\n📊 Loading data...")
features_df = pd.read_csv('data/processed/companies_with_features.csv')
labels_df = pd.read_csv('data/processed/companies_with_labels.csv', usecols=['symbol', 'shariah_compliant'])
df = features_df.merge(labels_df, on='symbol', how='left')
print(f"   Loaded {len(df)} companies with labels")

# Load model and scaler
print("\n🤖 Loading trained model...")
model = joblib.load('models/xgb_shariah_model.pkl')
scaler = joblib.load('models/xgb_scaler.pkl')
print("   Model loaded successfully")

# Feature columns
feature_cols = [col for col in features_df.columns if col != 'symbol']

# Get predictions for first 20 companies
print("\n🎯 Making predictions...")
sample_df = df.head(20).copy()
X = sample_df[feature_cols].values
X_scaled = scaler.transform(X)

predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)
confidence = np.max(probabilities, axis=1)

sample_df['prediction'] = predictions
sample_df['confidence'] = confidence

# Calculate accuracy
actual = sample_df['shariah_compliant'].values
accuracy = (predictions == actual).mean()

print(f"\n\033[1;34m📈 Quick Test Results (20 Sample Companies)\033[0m")
print("=" * 80)
print(f"Sample Accuracy:        {accuracy:.2%}")
print(f"Compliant (Sample):     {(predictions == 1).sum()} companies")
print(f"Non-Compliant (Sample): {(predictions == 0).sum()} companies")
print(f"Avg Confidence:         {confidence.mean():.2%}")

print(f"\n\033[1;34m📋 Predictions Table\033[0m")
print("-" * 80)
print(f"{'#':<3} {'Company':<12} {'Actual':<10} {'Predict':<10} {'Confidence':<15} {'Match':<8}")
print("-" * 80)

for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
    actual_val = '✓ Yes' if row['shariah_compliant'] == 1 else '✗ No'
    predict = '✓ Yes' if row['prediction'] == 1 else '✗ No'
    match = '✓' if row['prediction'] == row['shariah_compliant'] else '✗'
    
    print(f"{idx:<3} {row['symbol']:<12} {actual_val:<10} {predict:<10} {row['confidence']:.2%}          {match:<8}")

print("-" * 80)

# Show model feature importance
print(f"\n\033[1;34m🔍 Top 5 Important Features\033[0m")
print("-" * 80)
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

for idx, (_, row) in enumerate(importance_df.head(5).iterrows(), 1):
    bar_width = int(row['importance'] * 50)
    bar = '█' * bar_width
    print(f"{idx}. {row['feature']:<30} {row['importance']:.2%} {bar}")

print("\n" + "=" * 80)
print(f"\n\033[1;32m✅ Quick test completed successfully!\033[0m")
print(f"\n💡 Next steps:")
print(f"   • Run './run.sh' for full pipeline execution")
print(f"   • Run './view_results.sh' to see all predictions")
print(f"   • Check 'reports/model_predictions_explanations.csv' for full results")

ENDPYTHON

log_success "Quick test complete!"
echo -e "\n${YELLOW}💡 Tips:${NC}"
echo "   • Run './run.sh' to execute the complete pipeline"
echo "   • Run './view_results.sh' to see detailed results"
echo "   • Run './run.sh --help' for all available options"
