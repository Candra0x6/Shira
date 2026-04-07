#!/bin/bash

################################################################################
# Show Model Results in Console
# 
# Usage: ./view_results.sh
#
################################################################################

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPORTS_DIR="$PROJECT_DIR/reports"

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

if [ ! -f "$REPORTS_DIR/model_predictions_explanations.csv" ]; then
    echo -e "${RED}Error: Results file not found.${NC}"
    echo -e "${BLUE}Run './run.sh' first to generate predictions.${NC}"
    exit 1
fi

print_header "📊 Shariah Compliance Model Results"

python3 << ENDPYTHON
import pandas as pd
import os

os.chdir('$PROJECT_DIR')
df = pd.read_csv('reports/model_predictions_explanations.csv')

# === OVERVIEW ===
print(f"\n\033[1;34m🔍 Overview\033[0m")
print("-" * 70)
print(f"Total Companies Analyzed:       {len(df):>6}")
print(f"Shariah Compliant:              {(df['predicted_compliant'] == 1).sum():>6} ({(df['predicted_compliant'] == 1).sum()/len(df)*100:>5.1f}%)")
print(f"Non-Compliant:                  {(df['predicted_compliant'] == 0).sum():>6} ({(df['predicted_compliant'] == 0).sum()/len(df)*100:>5.1f}%)")
print(f"Average Confidence Score:       {df['confidence'].mean():>6.2%}")
print(f"Min Confidence:                 {df['confidence'].min():>6.2%}")
print(f"Max Confidence:                 {df['confidence'].max():>6.2%}")

# === ACCURACY ===
if 'prediction_correct' in df.columns:
    accuracy = (df['prediction_correct'] == 1).sum() / len(df)
    print(f"\n\033[1;34m✅ Model Accuracy\033[0m")
    print("-" * 70)
    print(f"Test Set Accuracy:              {accuracy:>6.2%}")
    
    # Confusion matrix
    tn = ((df['predicted_compliant'] == 0) & (df['shariah_compliant'] == 0)).sum()
    tp = ((df['predicted_compliant'] == 1) & (df['shariah_compliant'] == 1)).sum()
    fn = ((df['predicted_compliant'] == 0) & (df['shariah_compliant'] == 1)).sum()
    fp = ((df['predicted_compliant'] == 1) & (df['shariah_compliant'] == 0)).sum()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"Sensitivity (Recall):           {sensitivity:>6.2%}")
    print(f"Specificity:                    {specificity:>6.2%}")
    print(f"Precision:                      {precision:>6.2%}")

# === TOP COMPLIANT COMPANIES ===
print(f"\n\033[1;32m🏆 Top 15 Compliant Companies (Highest Confidence)\033[0m")
print("-" * 70)
compliant = df[df['predicted_compliant'] == 1].nlargest(15, 'confidence')[['symbol', 'confidence']]
print(f"{'Rank':<6} {'Company Symbol':<20} {'Confidence Score':<20}")
print("-" * 70)
for rank, (idx, row) in enumerate(compliant.iterrows(), 1):
    bar_width = int(row['confidence'] * 30)
    bar = '█' * bar_width + '░' * (30 - bar_width)
    print(f"{rank:<6} {row['symbol']:<20} {row['confidence']:.2%} [{bar}]")

# === BOTTOM NON-COMPLIANT COMPANIES ===
print(f"\n\033[1;31m⚠️  Top 15 Non-Compliant Companies (Lowest Compliance Score)\033[0m")
print("-" * 70)
non_compliant = df[df['predicted_compliant'] == 0].nsmallest(15, 'confidence')[['symbol', 'confidence']]
print(f"{'Rank':<6} {'Company Symbol':<20} {'Risk Score':<20}")
print("-" * 70)
for rank, (idx, row) in enumerate(non_compliant.iterrows(), 1):
    risk_score = 1 - row['confidence']
    bar_width = int(risk_score * 30)
    bar = '█' * bar_width + '░' * (30 - bar_width)
    print(f"{rank:<6} {row['symbol']:<20} {risk_score:.2%} [{bar}]")

# === CONFIDENCE DISTRIBUTION ===
print(f"\n\033[1;34m📈 Confidence Score Distribution\033[0m")
print("-" * 70)
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
counts = pd.cut(df['confidence'], bins=bins, labels=labels).value_counts().sort_index()
for label, count in counts.items():
    pct = count / len(df) * 100
    bar_width = int(pct / 2)
    bar = '█' * bar_width
    print(f"{label:>8}: {count:>4} companies ({pct:>5.1f}%) {bar}")

# === SAMPLE PREDICTIONS ===
print(f"\n\033[1;34m📋 Sample Predictions (First 10 Companies)\033[0m")
print("-" * 70)
print(f"{'#':<3} {'Symbol':<15} {'Prediction':<15} {'Confidence':<12} {'Status':<12}")
print("-" * 70)
for idx, row in df.head(10).iterrows():
    pred_text = '✅ Compliant' if row['predicted_compliant'] == 1 else '❌ Non-Compl.'
    status = '✓ High' if row['confidence'] > 0.85 else '⚠ Medium' if row['confidence'] > 0.70 else '✗ Low'
    print(f"{idx+1:<3} {row['symbol']:<15} {pred_text:<15} {row['confidence']:.2%}        {status:<12}")

print("\n" + "=" * 70)

ENDPYTHON

echo -e "\n${GREEN}✅ Results displayed above.${NC}"
echo -e "${BLUE}📁 Full results saved in: reports/model_predictions_explanations.csv${NC}\n"
