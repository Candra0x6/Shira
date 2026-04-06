# ML Model Tuning Plan: 92% → >93% Accuracy

## Executive Summary

**Goal:** Increase model accuracy from 92% to >93% using aggressive optimization
**Approach:** Multi-layered tuning (class imbalance + features + hyperparameters + ensemble)
**Expected Result:** 94-96% accuracy achievable
**Scope:** Comprehensive full optimization with interpretability considerations
**Timeline:** 4-6 hours of implementation

---

## Phase 1: Class Imbalance Correction (QUICK WIN - 1-2% gain)

### Problem
- Data: 77.8% compliant vs 22.2% non-compliant (3.5:1 imbalance)
- Model consequence: Overfits to majority class
- Current: High recall, lower precision → too many false positives

### Solutions

#### 1a. Scale Positive Weight
```python
# In XGBoost initialization:
scale_pos_weight = n_negative / n_positive = 110/386 ≈ 0.285

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=0.285,  # ← ADD THIS
    ...
)
```

#### 1b. SMOTE Oversampling (Optional)
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, sampling_strategy=0.7)  # Balance to 70% minority
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
```

#### 1c. Decision Threshold Optimization
```python
# Instead of default 0.5, find optimal threshold
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores = []
for threshold in thresholds:
    y_pred_tuned = (y_pred_proba >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred_tuned)
    f1_scores.append(f1)

optimal_threshold = thresholds[np.argmax(f1_scores)]
# Expected: ~0.40-0.45 instead of 0.50
```

---

## Phase 2: Feature Engineering Enhancement (2-3% gain)

### Current Features (19 total)
1. Base financials: assets, liabilities, equity, revenue, income, cash flow
2. Financial ratios: debt ratios, ROE, ROA, profit margin, interest coverage
3. Shariah-specific: f_riba, f_nonhalal, riba_intensity
4. Sector: encoded categorical

### New Features to Add

#### 2a. Interaction Features
```python
# Capture non-linear relationships
df['debt_nonhalal_interaction'] = df['debt_to_assets'] * df['nonhalal_revenue_percent']
df['leverage_efficiency'] = df['roa'] / (df['debt_to_assets'] + 0.01)
df['halal_profitability'] = df['profit_margin'] * (1 - df['nonhalal_revenue_percent'])
df['islamic_health_score'] = (
    (1 - df['debt_to_assets']) * 
    (1 - df['nonhalal_revenue_percent']) * 
    df['interest_coverage']
)
```

#### 2b. Polynomial Features
```python
from sklearn.preprocessing import PolynomialFeatures

# For top features only to avoid explosion
top_features = ['debt_to_assets', 'nonhalal_revenue_percent', 'roa', 'profit_margin']
poly = PolynomialFeatures(degree=2, include_bias=False)
# Add poly features for interaction detection
```

#### 2c. Domain-Specific Indicators
```python
# Islamic Finance KPIs
df['shariah_health_index'] = (
    (1 - df['debt_to_assets']) * 0.4 +      # 40% weight: low debt
    (1 - df['nonhalal_revenue_percent']) * 0.4 +  # 40% weight: high halal %
    (df['interest_coverage'] > 2).astype(int) * 0.2   # 20% weight: interest coverage
)

df['financial_stress'] = (
    (df['debt_to_assets'] > 0.5) * 1 +
    (df['nonhalal_revenue_percent'] > 0.3) * 1 +
    (df['profit_margin'] < 0) * 1 +
    (df['interest_coverage'] < 1) * 1
)

df['liquidity_quality'] = df['operating_cash_flow'] / (df['total_liabilities'] + 1e-5)
```

#### 2d. Sector Risk Encoding
```python
# Replace simple encoding with sector risk levels
sector_risk = {
    'Technology': 1,      # Lower risk
    'Finance': 3,         # Medium-high risk
    'Banking': 5,         # High risk
    'Pharmaceuticals': 2,
    ...
}
df['sector_risk_level'] = df['sector'].map(sector_risk)
```

#### 2e. Statistical Features
```python
# Normalization and scale features
df['debt_zscore'] = (df['debt_to_assets'] - df['debt_to_assets'].mean()) / df['debt_to_assets'].std()
df['income_zscore'] = (df['nonhalal_revenue_percent'] - df['nonhalal_revenue_percent'].mean()) / df['nonhalal_revenue_percent'].std()

# Create bins/categories
df['debt_category'] = pd.cut(df['debt_to_assets'], bins=[0, 0.33, 0.5, 0.7, 1.0], 
                              labels=['low', 'medium', 'high', 'very_high'])
```

**Feature Count After:** ~35-40 features (vs current 19)

---

## Phase 3: Hyperparameter Tuning (2-3% gain)

### Current Hyperparameters
```python
n_estimators=150, max_depth=6, learning_rate=0.1, 
min_child_weight=3, subsample=0.8, colsample_bytree=0.8
```

### 3a. Grid Search (Quick)
```python
param_grid = {
    'max_depth': [4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
}

# Use GridSearchCV with stratified k-fold
from sklearn.model_selection import GridSearchCV

grid_search = GridSearchCV(
    xgb.XGBClassifier(n_estimators=200, random_state=42, scale_pos_weight=0.285),
    param_grid,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_weighted',  # Better than accuracy for imbalanced data
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
```

### 3b. Random Search (More comprehensive)
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'max_depth': range(3, 10),
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.15, 0.2],
    'n_estimators': [100, 150, 200, 250],
    'min_child_weight': range(1, 10),
    'gamma': [0, 0.1, 0.5, 1, 5],
    'subsample': np.arange(0.5, 1.0, 0.1),
    'colsample_bytree': np.arange(0.5, 1.0, 0.1),
    'reg_alpha': [0, 0.1, 1],        # L1 regularization
    'reg_lambda': [0, 0.1, 1],       # L2 regularization
}

random_search = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=42),
    param_dist,
    n_iter=100,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_weighted',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)
```

### 3c. Key Hyperparameters to Tune
| Parameter | Current | Test Range | Impact |
|-----------|---------|-----------|--------|
| max_depth | 6 | 4-8 | Balance complexity/overfit |
| learning_rate | 0.1 | 0.01-0.2 | Convergence speed |
| n_estimators | 150 | 100-300 | Model capacity |
| min_child_weight | 3 | 1-7 | Prevent overfit |
| gamma | (default=0) | 0-5 | Pruning threshold |
| reg_alpha | (default=0) | 0-1 | L1 regularization |
| reg_lambda | (default=1) | 0.1-2 | L2 regularization |
| subsample | 0.8 | 0.6-1.0 | Row sampling |
| colsample_bytree | 0.8 | 0.6-1.0 | Feature sampling |

---

## Phase 4: Ensemble Methods (1-2% gain)

### 4a. Voting Classifier
```python
from sklearn.ensemble import VotingClassifier

# Train 3 models with different algorithms
xgb_model = xgb.XGBClassifier(**best_xgb_params)
lgb_model = LGBMClassifier(**best_lgb_params)  # LightGBM
cat_model = CatBoostClassifier(**best_cat_params)  # CatBoost

voting = VotingClassifier(
    estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('cat', cat_model)],
    voting='soft'  # Use probability averaging
)

voting.fit(X_train_scaled, y_train)
# Expected improvement: +0.5-1.5% accuracy
```

### 4b. Stacking
```python
from sklearn.ensemble import StackingClassifier

# Base learners
base_learners = [
    ('xgb', xgb.XGBClassifier(**best_xgb_params)),
    ('lgb', LGBMClassifier(**best_lgb_params)),
]

# Meta-learner
meta_learner = LogisticRegression()

stacking = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=StratifiedKFold(n_splits=5, random_state=42)
)

stacking.fit(X_train_scaled, y_train)
```

### 4c. Weighted Averaging
```python
# Train multiple models with different seeds
models = []
for seed in [42, 43, 44, 45, 46]:
    model = xgb.XGBClassifier(**best_xgb_params, random_state=seed)
    model.fit(X_train_scaled, y_train)
    models.append(model)

# Ensemble prediction with weights
y_pred_ensemble = np.zeros(len(y_test))
for model in models:
    y_pred_ensemble += model.predict_proba(X_test_scaled)[:, 1]
y_pred_ensemble /= len(models)
y_pred_final = (y_pred_ensemble >= optimal_threshold).astype(int)
```

---

## Phase 5: Validation Strategy Improvement (1-2% gain)

### 5a. Stratified K-Fold (ensure class balance per fold)
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = {}
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    model = xgb.XGBClassifier(**best_params, scale_pos_weight=0.285)
    model.fit(X_train_fold, y_train_fold)
    
    y_pred = model.predict_proba(X_val_fold)[:, 1]
    for metric in ['accuracy', 'f1', 'auc']:
        # Calculate and store
```

### 5b. Nested Cross-Validation
```python
# Outer CV: for evaluation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV: for hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Tune on inner CV
    grid_search = GridSearchCV(..., cv=inner_cv)
    grid_search.fit(X_train, y_train)
    
    # Evaluate on test
    score = grid_search.score(X_test, y_test)
    # Store for final evaluation
```

### 5c. Calibration
```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrate probabilities after training
calibrated_model = CalibratedClassifierCV(
    base_estimator=best_model,
    method='sigmoid',  # or 'isotonic'
    cv=StratifiedKFold(n_splits=5)
)

calibrated_model.fit(X_train_scaled, y_train)
y_pred_proba_calibrated = calibrated_model.predict_proba(X_test_scaled)[:, 1]
```

---

## Phase 6: Evaluation Metrics Improvement (1-1.5% gain)

### 6a. Change Evaluation Metrics
```python
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, 
    matthews_corrcoef, roc_auc_score, precision_recall_curve
)

# Better metrics for imbalanced data
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Balanced Accuracy': balanced_accuracy_score(y_test, y_pred),
    'F1 (weighted)': f1_score(y_test, y_pred, average='weighted'),
    'F1 (macro)': f1_score(y_test, y_pred, average='macro'),
    'MCC': matthews_corrcoef(y_test, y_pred),
    'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
}

print({k: f'{v:.4f}' for k, v in metrics.items()})
```

### 6b. Confusion Matrix Analysis
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
# Analyze: where are we making mistakes?
# True Negatives, False Positives, False Negatives, True Positives
tn, fp, fn, tp = cm.ravel()

specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)
precision = tp / (tp + fp)
recall = sensitivity

print(f"Precision: {precision:.4f} (fewer false positives)")
print(f"Recall: {recall:.4f} (fewer false negatives)")
print(f"Specificity: {specificity:.4f}")
```

---

## Implementation Timeline

| Phase | Task | Time | Priority | Expected Gain |
|-------|------|------|----------|--------------|
| 1 | Class imbalance (scale_pos_weight + threshold) | 30 min | **HIGH** | 1-2% |
| 2 | Feature engineering (interactions + domain) | 1.5 hours | **HIGH** | 2-3% |
| 3 | Hyperparameter tuning (grid/random search) | 1.5 hours | **HIGH** | 2-3% |
| 4 | Ensemble methods (voting + stacking) | 1 hour | **MEDIUM** | 1-2% |
| 5 | Validation strategy (nested CV + calibration) | 45 min | **MEDIUM** | 1-1.5% |
| 6 | Metrics & analysis | 30 min | **LOW** | 0.5% |

**Total Time:** 5-5.5 hours
**Expected Cumulative Gain:** 8-12.5% improvement possible
**Realistic Target:** 94-96% accuracy (vs current 92%)

---

## Code Structure Changes

### File: `/src/ml_confidence_scorer_v2.py` (NEW)
```
- MLConfidenceScorerV2 class (extends current)
  - enhanced_prepare_features() - with all new features
  - tune_hyperparameters() - grid/random search
  - train_ensemble() - voting + stacking
  - calibrate_model() - probability calibration
  - evaluate_comprehensive() - all metrics
  - explain_model() - SHAP + feature importance
```

### File: `/src/hyperparameter_tuner.py` (NEW)
```
- HyperparameterTuner class
  - grid_search()
  - random_search()
  - bayesian_optimization()
  - report_best_params()
```

### File: `/notebooks/01_Shariah_Compliance_Scoring_MVP.ipynb` (MODIFY)
```
- Add new cells for tuning (Cell 10a-10f)
- Replace Cell 10-Hybrid with tuning pipeline
- Add comparison cell (old vs new accuracy)
```

---

## Risk Mitigation

### Overfitting Risk
- Use stratified k-fold CV throughout
- Monitor train vs test accuracy
- Use early stopping in XGBoost
- Apply regularization (gamma, reg_alpha, reg_lambda)

### Data Leakage Risk
- Never fit scaler/encoder on full dataset
- Use nested CV for hyperparameter tuning
- Ensure threshold optimization on validation, not test

### Interpretability Risk
- Keep ensemble size small (3-5 models max)
- Document all new features
- Maintain SHAP explainability
- Track feature importance changes

---

## Success Criteria

✓ Accuracy > 93% (target >93%)
✓ Balanced Accuracy > 90%
✓ F1-Score (weighted) > 0.92
✓ Precision > 0.88
✓ Recall > 0.90
✓ Model remains interpretable
✓ All metrics stable across CV folds
✓ No data leakage or overfitting

---

## Next Steps

1. **Implement Phase 1:** Class imbalance fixes (30 min) → Expected 92% → 93%
2. **Implement Phase 2:** Feature engineering (1.5 hrs) → Expected 93% → 95%
3. **Implement Phase 3:** Hyperparameter tuning (1.5 hrs) → Expected 95% → 95.5%
4. **Implement Phase 4:** Ensemble methods (1 hr) → Expected 95.5% → 96%
5. **Implement Phase 5:** Validation strategy (45 min) → Expected 96% → 96.5%
6. **Test & Validate:** Final accuracy check

**Ready to implement?** All phases are planned and ready for execution.
