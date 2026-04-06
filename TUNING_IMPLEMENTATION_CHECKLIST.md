# ML Tuning Implementation Checklist

## Summary
- **Current Accuracy:** 92%
- **Target Accuracy:** >93%
- **Realistic Target:** 94-96% achievable
- **Approach:** Comprehensive 6-phase tuning
- **Estimated Time:** 5-5.5 hours
- **Status:** PLAN READY - Awaiting Implementation Approval

---

## Pre-Implementation Analysis ✓

- [x] Current baseline identified (92% accuracy)
- [x] Class imbalance analyzed (3.5:1 ratio)
- [x] Data quality verified (496 records, 0 missing values)
- [x] Feature analysis completed (19 current features)
- [x] Hyperparameter review done (150 estimators, max_depth=6)
- [x] Bottlenecks identified
- [x] 6-phase plan created
- [x] Resource requirements estimated

---

## Phase 1: Class Imbalance Correction ⏳

### Tasks
- [ ] Add `scale_pos_weight=0.285` to XGBoost
- [ ] Implement threshold optimization loop
- [ ] Optional: Add SMOTE oversampling
- [ ] Test accuracy improvement

### Files to Create/Modify
- [ ] `/src/ml_confidence_scorer.py` - Update train() method
- [ ] Create threshold optimization utility function
- [ ] Update notebook cells with new params

### Success Metrics
- [ ] Accuracy: 92% → 93% (+1%)
- [ ] Balanced Accuracy: improved
- [ ] F1-Score: maintained or improved

**Estimated Time:** 30 minutes
**Expected Gain:** 1-2%

---

## Phase 2: Feature Engineering ⏳

### New Features to Create

#### 2a. Interaction Features
- [ ] `debt_nonhalal_interaction` = debt × non-halal
- [ ] `leverage_efficiency` = ROA / debt_ratio
- [ ] `halal_profitability` = profit_margin × (1 - non-halal%)
- [ ] `islamic_health_score` = composite of health indicators

#### 2b. Polynomial Features
- [ ] Apply PolynomialFeatures to top 4 features
- [ ] Limit degree to 2 to avoid explosion
- [ ] Test feature count (expect ~25-35 features)

#### 2c. Domain-Specific Indicators
- [ ] `shariah_health_index` = weighted composite
- [ ] `financial_stress` = count of violation indicators
- [ ] `liquidity_quality` = cash_flow / liabilities

#### 2d. Sector Risk Encoding
- [ ] Create sector_risk_level mapping
- [ ] Replace simple categorical with risk scoring
- [ ] Test impact on model

#### 2e. Statistical Features
- [ ] Z-score normalization for debt & income
- [ ] Binning/categorization of ratios
- [ ] Create feature_importance tracking

### Files to Create/Modify
- [ ] `/src/feature_engineer.py` (NEW)
  - [ ] FeatureEngineer class
  - [ ] Method: create_interaction_features()
  - [ ] Method: create_polynomial_features()
  - [ ] Method: create_domain_features()
  - [ ] Method: create_sector_risk_features()
  - [ ] Method: create_statistical_features()

- [ ] `/src/ml_confidence_scorer.py`
  - [ ] Update prepare_features() to use FeatureEngineer

### Success Metrics
- [ ] Feature count: 19 → 35-40 features
- [ ] Accuracy: 93% → 95% (+2%)
- [ ] No multicollinearity issues
- [ ] Feature importance identified

**Estimated Time:** 1.5 hours
**Expected Gain:** 2-3%

---

## Phase 3: Hyperparameter Tuning ⏳

### Tuning Strategy

#### 3a. Grid Search (Fast)
- [ ] Define param_grid (5 × 4 × 3 × 3 × 3 = 540 combinations)
- [ ] Use StratifiedKFold (5 splits)
- [ ] Scoring: 'f1_weighted' (not accuracy)
- [ ] Parallelization: n_jobs=-1
- [ ] Save best_params

#### 3b. Random Search (Comprehensive)
- [ ] Define param_dist (10 hyperparameters)
- [ ] n_iter=100 random combinations
- [ ] Use StratifiedKFold (5 splits)
- [ ] Save best_params & comparison report

#### 3c. Key Parameters
- [ ] max_depth: [4,5,6,7,8]
- [ ] learning_rate: [0.01, 0.05, 0.1, 0.15]
- [ ] n_estimators: [100, 150, 200, 250]
- [ ] min_child_weight: [1, 3, 5, 7]
- [ ] gamma: [0, 0.1, 0.5, 1, 5]
- [ ] reg_alpha & reg_lambda: [0, 0.1, 1]
- [ ] subsample & colsample_bytree: [0.6, 0.7, 0.8, 0.9]

### Files to Create/Modify
- [ ] `/src/hyperparameter_tuner.py` (NEW)
  - [ ] HyperparameterTuner class
  - [ ] Method: grid_search()
  - [ ] Method: random_search()
  - [ ] Method: report_results()

- [ ] `/src/ml_confidence_scorer.py`
  - [ ] Update train() to use HyperparameterTuner

### Success Metrics
- [ ] Best hyperparameters identified
- [ ] Accuracy: 95% → 95.5% (+0.5%)
- [ ] All CV folds stable
- [ ] No overfitting detected

**Estimated Time:** 1.5 hours
**Expected Gain:** 2-3%

---

## Phase 4: Ensemble Methods ⏳

### Ensemble Strategies

#### 4a. Voting Classifier
- [ ] Train XGBoost with best params
- [ ] Train LightGBM with best params
- [ ] Train CatBoost with best params
- [ ] VotingClassifier with 'soft' voting
- [ ] Test accuracy

#### 4b. Stacking
- [ ] XGBoost & LightGBM as base learners
- [ ] LogisticRegression as meta-learner
- [ ] 5-fold CV for stacking
- [ ] Test accuracy

#### 4c. Model Blending
- [ ] Train 5 models with different random seeds
- [ ] Weighted averaging of probabilities
- [ ] Optimal weights optimization
- [ ] Test accuracy

### Files to Create/Modify
- [ ] `/src/ensemble_models.py` (NEW)
  - [ ] EnsembleBuilder class
  - [ ] Method: voting_classifier()
  - [ ] Method: stacking_classifier()
  - [ ] Method: model_blending()
  - [ ] Method: predict_ensemble()

- [ ] Requirements
  - [ ] Add lightgbm to dependencies
  - [ ] Add catboost to dependencies

### Success Metrics
- [ ] Voting accuracy: > 95.5%
- [ ] Stacking accuracy: > 95.7%
- [ ] Blending accuracy: > 95.5%
- [ ] Final ensemble accuracy: 96% +

**Estimated Time:** 1 hour
**Expected Gain:** 1-2%

---

## Phase 5: Validation Strategy ⏳

### Validation Improvements

#### 5a. Stratified K-Fold
- [ ] StratifiedKFold with 5 splits
- [ ] Ensure class balance per fold
- [ ] Track metrics per fold
- [ ] Check for stability

#### 5b. Nested Cross-Validation
- [ ] Outer CV: 5-fold for evaluation
- [ ] Inner CV: 3-fold for tuning
- [ ] Prevent data leakage
- [ ] Report final accuracy

#### 5c. Model Calibration
- [ ] CalibratedClassifierCV with sigmoid
- [ ] Improve probability estimates
- [ ] Better confidence scores
- [ ] Test calibration metrics

### Files to Create/Modify
- [ ] `/src/validation_utils.py` (NEW)
  - [ ] Function: stratified_kfold_eval()
  - [ ] Function: nested_cross_validation()
  - [ ] Function: calibrate_model()

- [ ] `/src/ml_confidence_scorer.py`
  - [ ] Update train() to use nested CV
  - [ ] Add calibration option

### Success Metrics
- [ ] CV folds stable (std < 2%)
- [ ] No data leakage detected
- [ ] Accuracy: ~96% across folds
- [ ] Calibration: improved probability estimates

**Estimated Time:** 45 minutes
**Expected Gain:** 1-1.5%

---

## Phase 6: Evaluation & Analysis ⏳

### Comprehensive Metrics

#### 6a. Multiple Metrics
- [ ] Accuracy (overall)
- [ ] Balanced Accuracy (weighted)
- [ ] F1-Score (weighted & macro)
- [ ] Matthews Correlation Coefficient
- [ ] ROC-AUC
- [ ] Precision-Recall AUC

#### 6b. Confusion Matrix Analysis
- [ ] True Positives, True Negatives
- [ ] False Positives, False Negatives
- [ ] Specificity, Sensitivity, Precision, Recall
- [ ] Identify remaining error patterns

#### 6c. Feature Importance
- [ ] Rank all features by importance
- [ ] SHAP values for interpretability
- [ ] Identify key drivers
- [ ] Document findings

### Files to Create/Modify
- [ ] `/src/evaluation_metrics.py` (NEW)
  - [ ] Function: comprehensive_metrics()
  - [ ] Function: plot_confusion_matrix()
  - [ ] Function: plot_roc_curve()
  - [ ] Function: plot_feature_importance()

- [ ] Notebook update
  - [ ] New cell for metrics comparison
  - [ ] Visualizations (4+ plots)
  - [ ] Results summary table

### Success Metrics
- [ ] Accuracy: >93% ✓
- [ ] Balanced Accuracy: >90%
- [ ] F1-Score: >0.92
- [ ] MCC: >0.80
- [ ] AUC-ROC: >0.95
- [ ] All metrics stable
- [ ] Results documented

**Estimated Time:** 30 minutes
**Expected Gain:** 0.5%

---

## Final Integration ⏳

### Notebook Updates
- [ ] Update Cell 10-Hybrid to use MLConfidenceScorerV2
- [ ] Add hyperparameter tuning cell
- [ ] Add ensemble training cell
- [ ] Add metrics comparison cell
- [ ] Add visualizations (6+ plots)
- [ ] Update results summary

### Documentation
- [ ] Update HYBRID_MODEL_SUMMARY.md
- [ ] Update QUICK_START_HYBRID_MODEL.md
- [ ] Create TUNING_RESULTS.md
- [ ] Create FEATURE_IMPORTANCE.md
- [ ] Add final accuracy comparison

### Testing
- [ ] Test notebook end-to-end
- [ ] Verify all cells run without errors
- [ ] Check accuracy > 93%
- [ ] Verify reproducibility

### Version Control
- [ ] Commit all new files
- [ ] Commit all modified files
- [ ] Push to repository
- [ ] Create release notes

---

## Success Criteria Checklist

### Accuracy Targets
- [ ] Baseline accuracy improved: 92% → 93%+
- [ ] Phase 1 gains: +1-2%
- [ ] Phase 2 gains: +2-3%
- [ ] Phase 3 gains: +2-3%
- [ ] Phase 4 gains: +1-2%
- [ ] Phase 5 gains: +1-1.5%
- [ ] **Final accuracy: 94-96%**

### Model Quality
- [ ] No overfitting (train-test gap < 2%)
- [ ] No data leakage (proper CV usage)
- [ ] Stable across CV folds (std < 2%)
- [ ] Balanced metrics (precision & recall)
- [ ] Interpretable (feature importance clear)

### Code Quality
- [ ] All new classes well-documented
- [ ] All methods have docstrings
- [ ] Error handling implemented
- [ ] No deprecated functions used
- [ ] Type hints included
- [ ] DRY principle followed

### Documentation
- [ ] Tuning plan documented
- [ ] Results documented with metrics
- [ ] Feature engineering explained
- [ ] Hyperparameters justified
- [ ] Ensemble strategy documented
- [ ] Usage instructions clear

---

## Risk Mitigation Checklist

### Data Leakage Prevention
- [ ] Scaler fit only on training data
- [ ] Encoders fit only on training data
- [ ] Threshold tuning on validation, not test
- [ ] Nested CV for hyperparameters
- [ ] No label peeking in feature engineering

### Overfitting Prevention
- [ ] Regularization applied (gamma, alpha, lambda)
- [ ] Early stopping used
- [ ] Cross-validation with 5+ folds
- [ ] Monitor train vs test metrics
- [ ] Feature selection if needed

### Interpretability Maintenance
- [ ] Ensemble size limited (3-5 models)
- [ ] Feature importance tracked
- [ ] SHAP explainability maintained
- [ ] New features documented
- [ ] Decision logic clear

---

## Implementation Order

**Recommended Sequence:**
1. **Phase 1** (30 min) - Quick win for 1-2% gain
2. **Phase 2** (1.5 hrs) - Feature engineering
3. **Phase 3** (1.5 hrs) - Hyperparameter tuning
4. **Phase 4** (1 hr) - Ensemble methods
5. **Phase 5** (45 min) - Validation strategy
6. **Phase 6** (30 min) - Final evaluation

**Total: 5.5 hours**
**Break points:** After phases 1, 3, and 5

---

## Resource Requirements

### Hardware
- [ ] GPU preferred (for faster tuning) - optional
- [ ] 8+ GB RAM (for grid search)
- [ ] 10+ GB disk space (for models & logs)

### Dependencies to Add
```python
# Already installed:
- xgboost
- sklearn
- pandas
- numpy

# Need to install:
- lightgbm (for ensemble)
- catboost (for ensemble)
- imbalanced-learn (optional, for SMOTE)
```

### Time Allocation
- Phase 1: 30 min (quick win)
- Phase 2: 1.5 hrs (feature engineering)
- Phase 3: 1.5 hrs (grid/random search)
- Phase 4: 1 hr (ensemble)
- Phase 5: 45 min (validation)
- Phase 6: 30 min (evaluation)
- Integration: 30 min (notebook + docs)

---

## Expected Results Summary

| Metric | Current | Target | Expected |
|--------|---------|--------|----------|
| Accuracy | 92.0% | >93% | 94-96% |
| Balanced Accuracy | ~88% | >90% | 91-93% |
| F1-Score | 95% | >92% | 93-95% |
| Precision | ~91% | >88% | 90-92% |
| Recall | ~96% | >90% | 93-95% |
| AUC-ROC | ~0.95 | >0.93 | 0.96-0.98 |
| MCC | ~0.76 | >0.80 | 0.82-0.88 |

---

## Notes

- All code follows existing patterns in codebase
- Backward compatibility maintained
- Original models preserved as fallback
- Documentation prioritized for maintenance
- Testing recommended after each phase
- Consider parallelization for grid search
- Monitor GPU memory during ensemble training

---

## Ready for Implementation?

**Current Status:** PLAN COMPLETE ✓

**Approval Required For:**
- [x] Phase 1: Class imbalance fixes
- [x] Phase 2: Feature engineering
- [x] Phase 3: Hyperparameter tuning
- [x] Phase 4: Ensemble methods
- [x] Phase 5: Validation strategy
- [x] Phase 6: Evaluation & analysis

**Next Step:** Execute implementation plan phase-by-phase

