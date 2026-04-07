# ⚙️ Setup & Installation Guide

> Complete setup instructions for both business analysts and data engineers

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Quick Start (5 min)](#quick-start-5-min)
3. [For Business Analysts](#for-business-analysts)
4. [For Data Engineers](#for-data-engineers)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements
- **OS:** Linux, macOS, or Windows (with WSL2)
- **Python:** 3.8 or later
- **RAM:** 4 GB (8 GB recommended)
- **Disk:** 2 GB free space
- **Network:** Internet access for initial setup only

### Recommended Setup
- **OS:** Linux (Ubuntu 20.04+) or macOS
- **Python:** 3.10 or later
- **RAM:** 8+ GB
- **CPU:** Multi-core processor
- **GPU:** CUDA 11+ (optional, for faster training)

### Dependency Versions
```
Python:                3.8+
pandas:               2.1.0
numpy:                1.24.0
scikit-learn:         1.3.0
xgboost:              2.0.0
shap:                 0.43.0
matplotlib:           3.8.0
seaborn:              0.13.0
jupyter:              1.0.0 (optional)
```

---

## Quick Start (5 min)

### Option 1: For Viewing Results Only (No Setup Needed)
If you just want to see predictions and reports:

```bash
# 1. View the technical report (PDF)
open reports/Technical_Report_Shariah_Compliance_Model.pdf

# 2. View predictions CSV
cat reports/predictions_with_explanations.csv | head -20

# 3. View visualizations
open reports/feature_importance_shap.png
open reports/shap_summary_plot.png
open reports/prediction_distribution.png
```

**Files to Review:**
- ✅ `README.md` - Project overview
- ✅ `reports/Technical_Report_*.pdf` - Full technical report
- ✅ `reports/predictions_with_explanations.csv` - All 496 predictions
- ✅ `reports/*.png` - Visualizations

---

## For Business Analysts

### Goal
Understand model predictions, review results, and generate compliance reports.

### Setup (5 minutes)

#### Step 1: Verify Python Installation
```bash
python3 --version  # Should be 3.8+
# Output example: Python 3.10.12
```

If Python is not installed, [download it](https://www.python.org/downloads/).

#### Step 2: Clone Repository (if needed)
```bash
cd /home/cn/projects/competition/model
# Already in the right location for this project
```

#### Step 3: View Pre-Generated Results
```bash
# Check what's already available
ls -lh reports/
ls -lh models/

# You should see:
# ✓ Technical_Report_Shariah_Compliance_Model.pdf (17 KB)
# ✓ predictions_with_explanations.csv (35 KB)
# ✓ feature_importance_shap.png (246 KB)
# ✓ shap_summary_plot.png (171 KB)
# ✓ prediction_distribution.png (161 KB)
```

#### Step 4: Open Prediction Results
```bash
# View predictions in spreadsheet application
open reports/predictions_with_explanations.csv

# Or view in terminal
head -20 reports/predictions_with_explanations.csv
```

### Using Results

#### Understanding Predictions
Each row in `predictions_with_explanations.csv` contains:

| Column | Meaning | Example |
|--------|---------|---------|
| `ticker` | Company ID | AALI_0 |
| `company_name` | Company Name | AALI Corp 0 |
| `sector` | Industry | Palm Oil |
| `predicted_compliant` | Prediction (0=No, 1=Yes) | 1 |
| `prob_compliant` | Confidence % | 0.765 (76.5%) |
| `actual_compliant` | True label (if known) | 1 |
| `correct_prediction` | Was model right? | True |

#### Reading the Technical Report
```
1. Skip to "Executive Summary" (Page 1)
   → See key metrics and findings

2. Read "Feature Importance" section (Page 3)
   → Understand what drives compliance decisions

3. Review "Error Analysis" (Page 4)
   → Know when model is uncertain

4. Check "Production Status" (This README)
   → Confirm model is ready to deploy
```

#### Interpreting Confidence Scores
```
prob_compliant > 0.80  → High confidence (safe to auto-decide)
prob_compliant 0.45-0.80 → Borderline (recommend manual review)
prob_compliant < 0.45  → Low confidence (investigate manually)
```

### Common Tasks for Analysts

#### Task 1: Find Companies with Low Compliance
```bash
# Using Python
python3 << 'EOF'
import pandas as pd

df = pd.read_csv('reports/predictions_with_explanations.csv')

# Show non-compliant companies
non_compliant = df[df['predicted_compliant'] == 0]
print(f"Found {len(non_compliant)} non-compliant companies:\n")
print(non_compliant[['ticker', 'company_name', 'sector', 'prob_compliant']])
EOF
```

#### Task 2: Export Results to Excel
```bash
# Using Python
python3 << 'EOF'
import pandas as pd

df = pd.read_csv('reports/predictions_with_explanations.csv')
df.to_excel('compliance_predictions.xlsx', index=False)
print("✓ Exported to compliance_predictions.xlsx")
EOF
```

#### Task 3: Filter by Sector
```bash
# Show all Palm Oil companies
python3 << 'EOF'
import pandas as pd

df = pd.read_csv('reports/predictions_with_explanations.csv')
palm_oil = df[df['sector'] == 'Palm Oil']
print(f"Palm Oil companies: {len(palm_oil)}")
print(f"Compliant: {(palm_oil['predicted_compliant'] == 1).sum()}")
print(f"Non-compliant: {(palm_oil['predicted_compliant'] == 0).sum()}")
EOF
```

### Next Steps
- ✅ Review the PDF technical report
- ✅ Check prediction CSV for specific companies
- ✅ Share results with stakeholders
- → For more details, see "For Data Engineers" section

---

## For Data Engineers

### Goal
Reproduce model predictions, run tests, integrate into pipelines.

### Setup (10 minutes)

#### Step 1: Create Python Virtual Environment
```bash
cd /home/cn/projects/competition/model

# Create virtual environment
python3 -m venv venv_ml

# Activate it
source venv_ml/bin/activate

# Upgrade pip
pip install --upgrade pip
```

#### Step 2: Install Dependencies
```bash
# Option A: Quick install (pre-tested versions)
pip install -q pandas==2.1.0 numpy==1.24.0 scikit-learn==1.3.0 \
  xgboost==2.0.0 shap==0.43.0 matplotlib==3.8.0 seaborn==0.13.0

# Option B: From requirements file (if it exists)
pip install -r requirements.txt

# Verify installations
python3 << 'EOF'
import pandas, numpy, sklearn, xgboost, shap
print("✓ All packages installed successfully!")
print(f"  - XGBoost version: {xgboost.__version__}")
print(f"  - SHAP version: {shap.__version__}")
EOF
```

#### Step 3: Verify Model Artifacts
```bash
# Check that all model files exist
ls -lh models/final_trained_model.pkl
ls -lh models/feature_scaler.pkl
ls -lh models/model_metadata.json

# Output should show files > 0 bytes
# ✓ final_trained_model.pkl (173 KB)
# ✓ feature_scaler.pkl (864 B)
# ✓ model_metadata.json (1.6 KB)
```

#### Step 4: Load and Test Model
```bash
# Quick test to ensure model works
python3 << 'EOF'
import pickle
import json

# Load model
with open('models/final_trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load metadata
with open('models/model_metadata.json', 'r') as f:
    meta = json.load(f)

print("✓ Model loaded successfully")
print(f"  - Estimators: {model.n_estimators}")
print(f"  - Max depth: {model.max_depth}")
print(f"  - Test accuracy: {meta['performance']['test_accuracy']*100:.1f}%")
print(f"  - Test F1: {meta['performance']['test_f1']*100:.1f}%")
EOF
```

#### Step 5: Run Interactive Notebook
```bash
# Run the full prediction notebook with SHAP explanations
python3 notebooks/Model_Predictions_and_Explanations.py

# This will:
# ✓ Load trained XGBoost model
# ✓ Load feature scaler
# ✓ Prepare engineered features (19 total)
# ✓ Make predictions on 496 companies
# ✓ Calculate SHAP values
# ✓ Generate 3 visualizations
# ✓ Export predictions to CSV
# ✓ Show 3 detailed prediction examples

# Output:
# ✓ Saved: reports/feature_importance_shap.png
# ✓ Saved: reports/shap_summary_plot.png
# ✓ Saved: reports/prediction_distribution.png
# ✓ Saved: reports/predictions_with_explanations.csv
```

### Development Setup

#### Setting Up IDE
```bash
# For VS Code
code .

# For PyCharm
# Open project directory in PyCharm

# For Jupyter Notebook
pip install jupyter
jupyter notebook

# Then open:
# http://localhost:8888
# And navigate to notebooks/
```

#### Project Structure Navigation
```
models/                    ← Trained model + scaler
├── final_trained_model.pkl (Load with pickle)
├── feature_scaler.pkl     (Load with pickle)
└── model_metadata.json    (Load with json)

data/
├── raw/idx_real_kaggle.csv (Original data - 496 rows)
└── engineered_features.csv (19 features for prediction)

reports/
├── Technical_Report_*.pdf  (7-page report)
├── predictions_with_explanations.csv (All 496 predictions)
└── *.png                   (3 visualizations)

notebooks/
└── Model_Predictions_and_Explanations.py (Main notebook - 588 lines)

tests/
├── test_features.py        (Feature engineering tests)
├── test_model.py           (Model prediction tests)
└── test_shap.py           (SHAP explanation tests)
```

### Running Code Examples

#### Example 1: Load Model and Make Predictions
```python
import pickle
import pandas as pd
import numpy as np

# Load model and scaler
model = pickle.load(open('models/final_trained_model.pkl', 'rb'))
scaler = pickle.load(open('models/feature_scaler.pkl', 'rb'))

# Load data
X = pd.read_csv('data/engineered_features.csv')

# Scale features
X_scaled = scaler.transform(X)

# Make predictions
predictions = model.predict(X_scaled)  # 0 or 1
probabilities = model.predict_proba(X_scaled)  # [prob_non_compliant, prob_compliant]

print(f"Total companies: {len(predictions)}")
print(f"Compliant: {(predictions == 1).sum()}")
print(f"Non-compliant: {(predictions == 0).sum()}")
```

#### Example 2: Get SHAP Explanations
```python
import pickle
import shap
import pandas as pd

# Load model and data
model = pickle.load(open('models/final_trained_model.pkl', 'rb'))
scaler = pickle.load(open('models/feature_scaler.pkl', 'rb'))
X = pd.read_csv('data/engineered_features.csv')
X_scaled = scaler.transform(X)

# Create SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)

# For company 0, get top 5 driving features
sample_shap = shap_values[1][0]  # Binary classification: class 1 (compliant)
features = X.columns
importance = pd.DataFrame({
    'feature': features,
    'shap_value': sample_shap
}).sort_values('shap_value', key=abs, ascending=False)

print("Top 5 features for company 0:")
print(importance.head(5))
```

#### Example 3: Integrate with Pandas
```python
import pickle
import pandas as pd

# Load data
df = pd.read_csv('reports/predictions_with_explanations.csv')

# Filter non-compliant companies
non_compliant = df[df['predicted_compliant'] == 0]

# Get companies with high confidence
high_conf = df[df['prediction_confidence'] > 0.85]

# Group by sector
by_sector = df.groupby('sector').agg({
    'predicted_compliant': ['count', 'sum', 'mean']
})

print(by_sector)
```

---

## Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'xgboost'"

**Solution:**
```bash
# Ensure virtual environment is activated
source venv_ml/bin/activate

# Reinstall XGBoost
pip install --force-reinstall xgboost==2.0.0

# Verify
python3 -c "import xgboost; print(xgboost.__version__)"
```

### Issue 2: "FileNotFoundError: models/final_trained_model.pkl"

**Solution:**
```bash
# Verify you're in the right directory
pwd
# Should output: /home/cn/projects/competition/model

# Check if files exist
ls -lh models/
# Should show: final_trained_model.pkl, feature_scaler.pkl, model_metadata.json
```

### Issue 3: "Cannot open data file: idx_real_kaggle.csv"

**Solution:**
```bash
# Prepare data if missing
python3 << 'EOF'
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load raw data (assuming it exists)
df = pd.read_csv('data/raw/idx_real_kaggle.csv')

# Engineer features
df['debt_to_equity'] = (df['total_liabilities'] / df['total_equity']).fillna(0)
df['debt_to_assets'] = (df['total_liabilities'] / df['total_assets']).fillna(0)
df['roe'] = (df['net_income'] / df['total_equity']).fillna(0)
df['roa'] = (df['net_income'] / df['total_assets']).fillna(0)
df['profit_margin'] = (df['net_income'] / df['net_revenue']).fillna(0)
df['interest_coverage'] = (df['net_income'] / (df['interest_expense'] + 0.001)).fillna(0)
df['cash_flow_to_debt'] = (df['operating_cash_flow'] / (df['total_liabilities'] + 0.001)).fillna(0)
df['f_riba'] = df['interest_expense'] / (df['net_revenue'] + 0.001)
df['f_nonhalal'] = df['nonhalal_revenue_percent']
df['riba_intensity'] = df['interest_expense'] / (df['total_assets'] + 0.001)

le = LabelEncoder()
df['sector_encoded'] = le.fit_transform(df['sector'])

# Select features
feature_cols = [
    'total_assets', 'total_liabilities', 'total_equity', 'net_revenue',
    'nonhalal_revenue_percent', 'net_income', 'operating_cash_flow',
    'interest_expense', 'debt_to_equity', 'debt_to_assets', 'roe', 'roa',
    'profit_margin', 'interest_coverage', 'cash_flow_to_debt',
    'f_riba', 'f_nonhalal', 'riba_intensity', 'sector_encoded'
]

X = df[feature_cols]
X.to_csv('data/engineered_features.csv', index=False)
print("✓ Engineered features saved")
EOF
```

### Issue 4: "Cannot allocate memory" when running SHAP

**Solution:**
```bash
# SHAP can be memory-intensive. Use smaller sample:
python3 << 'EOF'
import pickle
import shap
import pandas as pd

model = pickle.load(open('models/final_trained_model.pkl', 'rb'))
scaler = pickle.load(open('models/feature_scaler.pkl', 'rb'))
X = pd.read_csv('data/engineered_features.csv')
X_scaled = scaler.transform(X)

# Use only first 100 samples
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled[:100])
print("✓ SHAP values calculated for 100 samples")
EOF
```

### Issue 5: "SHAP values shape mismatch"

**Solution:**
```bash
# For binary classification, SHAP returns values for each class
python3 << 'EOF'
import pickle
import shap

model = pickle.load(open('models/final_trained_model.pkl', 'rb'))

# Create dummy data
import numpy as np
X_dummy = np.random.randn(10, 19)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_dummy)

# For binary classification:
print(f"Type of shap_values: {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"  - Class 0 shape: {shap_values[0].shape}")
    print(f"  - Class 1 shape: {shap_values[1].shape}")
    shap_values_class1 = shap_values[1]  # Use class 1 (compliant)
else:
    print(f"  - Direct shape: {shap_values.shape}")
    shap_values_class1 = shap_values
EOF
```

---

## Verification Checklist

After setup, verify everything works:

```bash
# 1. Check Python version
python3 --version  # Should be 3.8+

# 2. Check virtual environment
which python  # Should point to venv_ml/bin/python

# 3. Check dependencies
pip list | grep -E "xgboost|shap|pandas"

# 4. Check model artifacts
ls -lh models/final_trained_model.pkl
ls -lh models/feature_scaler.pkl
ls -lh models/model_metadata.json

# 5. Test model loading
python3 -c "import pickle; pickle.load(open('models/final_trained_model.pkl', 'rb')); print('✓ Model loads')"

# 6. Run automated tests (see TESTING.md)
bash tests/run_all_tests.sh
```

---

## Next Steps

### For Business Analysts
→ [View the Technical Report](../reports/Technical_Report_Shariah_Compliance_Model.pdf)  
→ [Check Prediction Results](../reports/predictions_with_explanations.csv)  
→ [Read DEPLOYMENT_AND_USAGE_GUIDE.md](./DEPLOYMENT_AND_USAGE_GUIDE.md)

### For Data Engineers
→ [Run Automated Tests](./TESTING.md)  
→ [Explore Model Artifacts](../models/)  
→ [Review Model Code](../notebooks/Model_Predictions_and_Explanations.py)

### For DevOps
→ [See DEPLOYMENT_AND_USAGE_GUIDE.md](./DEPLOYMENT_AND_USAGE_GUIDE.md)  
→ [Production Integration](./DEPLOYMENT_AND_USAGE_GUIDE.md#production-integration)  
→ [Monitoring & Maintenance](./DEPLOYMENT_AND_USAGE_GUIDE.md#monitoring--maintenance)

---

**Last Updated:** April 7, 2026  
**Version:** 1.0 (Stable)
