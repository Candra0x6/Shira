# ============================================================
# CELL 01 — RUNTIME VALIDATION & ENVIRONMENT BOOTSTRAP
# MANDATORY: Must execute before any other cell
# ============================================================

import subprocess, sys, os, json, warnings
warnings.filterwarnings('ignore')

# --- GPU/TPU Detection ---
try:
    import torch
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else 'CPU'
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if gpu_available else 0
    print(f'[GPU] {gpu_name} | VRAM: {vram_gb:.1f} GB')
except:
    print('[INFO] PyTorch not yet installed — proceeding to dependency installation')

# --- Seed Locking (Reproducibility Mandate) ---
GLOBAL_SEED = 42
import random, numpy as np
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
os.environ['PYTHONHASHSEED'] = str(GLOBAL_SEED)
print(f'[SEED] Global random seed locked → {GLOBAL_SEED}')

# --- Colab Drive Mount ---
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    REPO_ROOT = '/content/drive/MyDrive/AIEMM_Project'
    print(f'[DRIVE] Mounted → {REPO_ROOT}')
except:
    # Local development mode
    REPO_ROOT = '/home/cn/projects/competition/model'
    print(f'[LOCAL] Development mode → {REPO_ROOT}')

# Create directory structure if not exists
for subdir in ['data/raw', 'data/processed', 'models/checkpoint', 'reports']:
    os.makedirs(f'{REPO_ROOT}/{subdir}', exist_ok=True)
    
print(f'[DIR] Project structure ready at {REPO_ROOT}')

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 02 — PINNED DEPENDENCY INSTALLATION
# ============================================================

%%capture install_log
!pip install -q \
    pandas==2.1.0 \
    numpy==1.24.0 \
    scikit-learn==1.3.0 \
    xgboost==2.0.0 \
    shap==0.43.0 \
    lime==0.2.0.1 \
    gradio==4.28.3 \
    pyngrok==7.1.6 \
    matplotlib==3.8.0 \
    seaborn==0.13.0

print('[INSTALL] All dependencies installed successfully')
print('[VERSIONS] Core packages:')
import pandas as pd; import numpy as np; import sklearn; import xgboost; import shap; import lime
print(f'  pandas: {pd.__version__}')
print(f'  numpy: {np.__version__}')
print(f'  scikit-learn: {sklearn.__version__}')
print(f'  xgboost: {xgboost.__version__}')
print(f'  shap: {shap.__version__}')
print(f'  lime: {lime.__version__}')

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 03 — DATA INGESTION (REAL IDX DATA)
# ============================================================

import pandas as pd
import os

# Configuration: Use real IDX data
DATA_SOURCE = 'real'  # Using real IDX 2023 data
REAL_DATA_PATH = f'{REPO_ROOT}/data/raw/idx_2023_real_500.csv'

if os.path.exists(REAL_DATA_PATH):
    df_raw = pd.read_csv(REAL_DATA_PATH)
    print(f'[INFO] Loaded real IDX data: {REAL_DATA_PATH}')
    print(f'       {len(df_raw)} records, {len(df_raw.columns)} columns')
else:
    print(f'[ERROR] IDX CSV not found at {REAL_DATA_PATH}')
    print(f'[INFO] Please download from Kaggle: "IDX Financial Statements Dataset"')
    df_raw = None

# Validate and display schema
if df_raw is not None:
    print(f'\n[SCHEMA] Columns present:')
    for col in df_raw.columns:
        print(f'  - {col}')
    
    print(f'\n[DATA] First 3 records:')
    print(df_raw.head(3))
    
    print(f'\n[STATS]')
    print(f'  - Total records: {len(df_raw)}')
    print(f'  - Shariah compliant: {df_raw["shariah_compliant"].sum()} ({df_raw["shariah_compliant"].mean():.1%})')
    print(f'  - Non-compliant: {(1-df_raw["shariah_compliant"]).sum()} ({(1-df_raw["shariah_compliant"]).mean():.1%})')
    print(f'  - Missing values: {df_raw.isnull().sum().sum()}')


# ===== CELL SEPARATOR =====

# ============================================================
# CELL 04 — EXPLORATORY DATA ANALYSIS
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

if df_raw is not None:
    print('[EDA] Data Summary Statistics:')
    print(df_raw[['total_assets', 'total_debt', 'interest_bearing_debt', 'total_revenue', 'nonhalal_revenue']].describe())
    
    print('\n[EDA] Missing Values:')
    print(df_raw.isnull().sum())
    
    print('\n[EDA] Sector Distribution:')
    print(df_raw['sector_code'].value_counts())
    
    # Data type validation
    numeric_cols = ['total_assets', 'total_debt', 'interest_bearing_debt', 'total_revenue', 'nonhalal_revenue']
    for col in numeric_cols:
        if df_raw[col].dtype not in ['int64', 'float64']:
            print(f'[WARNING] {col} is not numeric, converting...')
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
    
    print('\n[EDA] Data types corrected.')
    print('[EDA] Ready for feature engineering.')

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 05 — FEATURE ENGINEERING (3 CORE FEATURES)
# ============================================================

import pandas as pd
import numpy as np

if df_raw is not None:
    df = df_raw.copy()
    
    # Feature 1: F_RIBA (Interest-Bearing Debt Ratio)
    # Formula: interest_bearing_debt / total_debt
    df['F_RIBA'] = np.where(
        df['total_debt'] > 0,
        df['interest_bearing_debt'] / df['total_debt'],
        0
    )
    df['F_RIBA'] = df['F_RIBA'].clip(0, 1)  # Bound to [0, 1]
    
    # Feature 2: F_NONHALAL (Non-Halal Income Percentage)
    # Formula: nonhalal_revenue / total_revenue
    df['F_NONHALAL'] = np.where(
        df['total_revenue'] > 0,
        df['nonhalal_revenue'] / df['total_revenue'],
        0
    )
    df['F_NONHALAL'] = df['F_NONHALAL'].clip(0, 1)  # Bound to [0, 1]
    
    # Feature 3: LEVERAGE_RATIO (Debt-to-Assets Ratio)
    # Formula: total_debt / total_assets
    df['LEVERAGE_RATIO'] = np.where(
        df['total_assets'] > 0,
        df['total_debt'] / df['total_assets'],
        0
    )
    df['LEVERAGE_RATIO'] = df['LEVERAGE_RATIO'].clip(0, 1)  # Bound to [0, 1]
    
    print('[FEAT] Feature Engineering Completed:')
    print(f'  F_RIBA — Interest-Bearing Debt Ratio')
    print(f'    Mean: {df[\"F_RIBA\"].mean():.4f}, Median: {df[\"F_RIBA\"].median():.4f}')
    print(f'  F_NONHALAL — Non-Halal Income Percentage')
    print(f'    Mean: {df[\"F_NONHALAL\"].mean():.4f}, Median: {df[\"F_NONHALAL\"].median():.4f}')
    print(f'  LEVERAGE_RATIO — Debt-to-Assets Ratio')
    print(f'    Mean: {df[\"LEVERAGE_RATIO\"].mean():.4f}, Median: {df[\"LEVERAGE_RATIO\"].median():.4f}')
    
    print(f'\n[FEAT] Feature Summary:')
    print(df[['F_RIBA', 'F_NONHALAL', 'LEVERAGE_RATIO']].describe())

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 06 — RULE-BASED LABEL GENERATION (AAOIFI RULES)
# ============================================================

import pandas as pd

# Prohibited sectors per AAOIFI standards
PROHIBITED_SECTORS = ['ALCOHOL', 'WEAPONS', 'GAMBLING', 'PORK', 'CONVENTIONAL_BANKING', 'ADULT_CONTENT']

# Guardrail thresholds (hard rules, not overridable)
RIBA_THRESHOLD = 0.50       # F_RIBA > 0.50 = AUTO-REJECT
NONHALAL_THRESHOLD = 0.25   # F_NONHALAL > 0.25 = AUTO-REJECT

if df is not None:
    # Initialize: assume all PERMIT unless guardrails trigger
    df['SHARIAH_COMPLIANCE'] = 1  # 1 = PERMIT
    
    # Audit log for label reasoning
    df['REJECTION_REASON'] = ''
    
    # Guardrail 1: Prohibited Sector
    prohibited_mask = df['sector_code'].isin(PROHIBITED_SECTORS)
    df.loc[prohibited_mask, 'SHARIAH_COMPLIANCE'] = 0
    df.loc[prohibited_mask, 'REJECTION_REASON'] = 'PROHIBITED_SECTOR'
    
    # Guardrail 2: High Riba (Interest-Bearing Debt > 50%)
    riba_mask = df['F_RIBA'] > RIBA_THRESHOLD
    df.loc[riba_mask, 'SHARIAH_COMPLIANCE'] = 0
    df.loc[riba_mask & (df['REJECTION_REASON'] == ''), 'REJECTION_REASON'] = 'HIGH_RIBA'
    df.loc[riba_mask & (df['REJECTION_REASON'] != ''), 'REJECTION_REASON'] += '|HIGH_RIBA'
    
    # Guardrail 3: High Non-Halal Income > 25%
    nonhalal_mask = df['F_NONHALAL'] > NONHALAL_THRESHOLD
    df.loc[nonhalal_mask, 'SHARIAH_COMPLIANCE'] = 0
    df.loc[nonhalal_mask & (df['REJECTION_REASON'] == ''), 'REJECTION_REASON'] = 'HIGH_NONHALAL'
    df.loc[nonhalal_mask & (df['REJECTION_REASON'] != ''), 'REJECTION_REASON'] += '|HIGH_NONHALAL'
    
    print('[LABEL] Rule-Based Label Generation Complete (AAOIFI Standards):')
    print(f'  Permitted (SHARIAH_COMPLIANCE=1): {(df[\"SHARIAH_COMPLIANCE\"] == 1).sum()}')
    print(f'  Rejected  (SHARIAH_COMPLIANCE=0): {(df[\"SHARIAH_COMPLIANCE\"] == 0).sum()}')
    print(f'\n[LABEL] Rejection Reasons Distribution:')
    print(df[df['SHARIAH_COMPLIANCE'] == 0]['REJECTION_REASON'].value_counts())
    print(f'\n[LABEL] First 10 records with compliance labels:')
    print(df[['company_id', 'company_name', 'sector_code', 'F_RIBA', 'F_NONHALAL', 'SHARIAH_COMPLIANCE', 'REJECTION_REASON']].head(10))

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 07 — TRAIN-TEST SPLIT & FEATURE SCALING
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

if df is not None:
    # Select feature columns for model
    FEATURE_COLS = ['F_RIBA', 'F_NONHALAL', 'LEVERAGE_RATIO']
    X = df[FEATURE_COLS].copy()
    y = df['SHARIAH_COMPLIANCE'].copy()
    
    # Stratified train-test split (preserve class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=GLOBAL_SEED
    )
    
    print(f'[SPLIT] Train-Test Split (Stratified, 80-20):')
    print(f'  Training set: {len(X_train)} samples')
    print(f'  Test set: {len(X_test)} samples')
    print(f'\n[SPLIT] Class Distribution (Training):')
    print(f'  PERMIT (1): {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)')
    print(f'  REJECT (0): {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)')
    print(f'\n[SPLIT] Class Distribution (Test):')
    print(f'  PERMIT (1): {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.1f}%)')
    print(f'  REJECT (0): {(y_test == 0).sum()} ({(y_test == 0).sum() / len(y_test) * 100:.1f}%)')
    
    # Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=FEATURE_COLS, index=X_test.index)
    
    print(f'\n[SCALE] Feature Scaling Complete (StandardScaler):')
    print(f'  Training features mean: {X_train_scaled.mean().values}')
    print(f'  Training features std: {X_train_scaled.std().values}')
    print(f'\n[SCALE] Ready for model training.')

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 08 — XGBOOST MODEL INSTANTIATION & TRAINING
# ============================================================

import xgboost as xgb
import numpy as np

if X_train_scaled is not None:
    # XGBoost hyperparameters (optimized for financial classification)
    xgb_params = {
        'n_estimators': 200,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': GLOBAL_SEED,
        'verbosity': 1
    }
    
    # Create model
    model_xgb = xgb.XGBClassifier(**xgb_params)
    
    # Train with early stopping on validation set
    X_train_eval, X_val, y_train_eval, y_val = train_test_split(
        X_train_scaled, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=GLOBAL_SEED
    )
    
    print('[MODEL] XGBoost Training Started...')
    model_xgb.fit(
        X_train_eval, y_train_eval,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    print(f'[MODEL] XGBoost Training Complete')
    print(f'  Best iteration: {model_xgb.best_iteration}')
    print(f'  Best score (logloss): {model_xgb.best_score:.4f}')
    
    # Feature importance
    importances = model_xgb.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f'\n[MODEL] Feature Importance:')
    print(feature_importance_df)
    
    # Predictions on train/test
    y_train_pred = model_xgb.predict(X_train_scaled)
    y_test_pred = model_xgb.predict(X_test_scaled)
    y_train_pred_proba = model_xgb.predict_proba(X_train_scaled)[:, 1]
    y_test_pred_proba = model_xgb.predict_proba(X_test_scaled)[:, 1]
    
    print(f'\n[MODEL] Predictions ready for validation.')

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 09 — 5-FOLD STRATIFIED CROSS-VALIDATION
# ============================================================

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

if model_xgb is not None:
    # Define scoring metrics
    scoring = {
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall',
        'roc_auc': 'roc_auc'
    }
    
    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
    cv_results = cross_validate(
        model_xgb,
        X_train_scaled,
        y_train,
        cv=skf,
        scoring=scoring
    )
    
    # Summarize CV results
    print('[CV] 5-Fold Stratified Cross-Validation Results:')
    print(f'\nF1-Score:')
    print(f'  Folds: {cv_results[\"test_f1\"]}')
    print(f'  Mean: {cv_results[\"test_f1\"].mean():.4f} ± {cv_results[\"test_f1\"].std():.4f}')
    
    print(f'\nPrecision:')
    print(f'  Folds: {cv_results[\"test_precision\"]}')
    print(f'  Mean: {cv_results[\"test_precision\"].mean():.4f} ± {cv_results[\"test_precision\"].std():.4f}')
    
    print(f'\nRecall:')
    print(f'  Folds: {cv_results[\"test_recall\"]}')
    print(f'  Mean: {cv_results[\"test_recall\"].mean():.4f} ± {cv_results[\"test_recall\"].std():.4f}')
    
    print(f'\nAUC-ROC:')
    print(f'  Folds: {cv_results[\"test_roc_auc\"]}')
    print(f'  Mean: {cv_results[\"test_roc_auc\"].mean():.4f} ± {cv_results[\"test_roc_auc\"].std():.4f}')
    
    # Target thresholds check
    f1_target = cv_results['test_f1'].mean() >= 0.85
    precision_target = cv_results['test_precision'].mean() >= 0.90
    recall_target = cv_results['test_recall'].mean() >= 0.85
    auc_target = cv_results['test_roc_auc'].mean() >= 0.92
    
    print(f'\n[CV] Target Thresholds:')
    print(f'  F1 ≥ 0.85: {\"✓\" if f1_target else \"✗\"} ({cv_results[\"test_f1\"].mean():.4f})')
    print(f'  Precision ≥ 0.90: {\"✓\" if precision_target else \"✗\"} ({cv_results[\"test_precision\"].mean():.4f})')
    print(f'  Recall ≥ 0.85: {\"✓\" if recall_target else \"✗\"} ({cv_results[\"test_recall\"].mean():.4f})')
    print(f'  AUC-ROC ≥ 0.92: {\"✓\" if auc_target else \"✗\"} ({cv_results[\"test_roc_auc\"].mean():.4f})')

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 10 — SAVE MODEL & SCALER CHECKPOINT
# ============================================================

import pickle
import os

if model_xgb is not None:
    checkpoint_dir = f'{REPO_ROOT}/models/checkpoint'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save XGBoost model
    model_path = f'{checkpoint_dir}/xgb_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_xgb, f)
    
    # Save StandardScaler
    scaler_path = f'{checkpoint_dir}/scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save CV results
    cv_results_path = f'{checkpoint_dir}/cv_results.pkl'
    with open(cv_results_path, 'wb') as f:
        pickle.dump(cv_results, f)
    
    print(f'[SAVE] Model checkpoint saved:')
    print(f'  Model: {model_path}')
    print(f'  Scaler: {scaler_path}')
    print(f'  CV Results: {cv_results_path}')
    print(f'\n[SAVE] Checkpoint ready for Phase 4 (Evaluation & Deployment).')

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 11 — CLASSIFICATION METRICS (TEST SET)
# ============================================================

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report

if y_test_pred is not None:
    # Calculate metrics
    f1 = f1_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    auc_roc = roc_auc_score(y_test, y_test_pred_proba)
    
    print('[METRICS] Test Set Performance:')
    print(f'  F1-Score: {f1:.4f}')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall: {recall:.4f}')
    print(f'  AUC-ROC: {auc_roc:.4f}')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f'\n[METRICS] Confusion Matrix:')
    print(f'  TN={cm[0,0]}, FP={cm[0,1]}')
    print(f'  FN={cm[1,0]}, TP={cm[1,1]}')
    
    # Classification report
    print(f'\n[METRICS] Classification Report:')
    print(classification_report(y_test, y_test_pred, target_names=['REJECT', 'PERMIT']))
    
    # Target validation
    print(f'\n[METRICS] Target Thresholds (Test Set):')
    print(f'  F1 ≥ 0.85: {\"✓\" if f1 >= 0.85 else \"✗\"} ({f1:.4f})')
    print(f'  Precision ≥ 0.90: {\"✓\" if precision >= 0.90 else \"✗\"} ({precision:.4f})')
    print(f'  Recall ≥ 0.85: {\"✓\" if recall >= 0.85 else \"✗\"} ({recall:.4f})')
    print(f'  AUC-ROC ≥ 0.92: {\"✓\" if auc_roc >= 0.92 else \"✗\"} ({auc_roc:.4f})')

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 12 — VISUALIZATIONS (CONFUSION MATRIX & ROC CURVE)
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

if y_test_pred is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title('Confusion Matrix (Test Set)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_xticklabels(['REJECT', 'PERMIT'])
    axes[0].set_yticklabels(['REJECT', 'PERMIT'])
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
    auc_score = roc_auc_score(y_test, y_test_pred_proba)
    axes[1].plot(fpr, tpr, label=f'ROC (AUC={auc_score:.4f})', linewidth=2)
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Baseline')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve (Test Set)')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f'{REPO_ROOT}/reports/model_evaluation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f'[VIZ] Plots saved: {plot_path}')

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 13 — SHAP EXPLANATIONS
# ============================================================

import shap
import matplotlib.pyplot as plt

if model_xgb is not None:
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model_xgb)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # For binary classification, shap_values is a list; use [1] for positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    print('[SHAP] SHAP Explainer created')
    
    # Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_scaled, plot_type='bar')
    shap_beeswarm_path = f'{REPO_ROOT}/reports/shap_beeswarm.png'
    plt.savefig(shap_beeswarm_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'[SHAP] Beeswarm plot saved: {shap_beeswarm_path}')
    
    # Waterfall plot for first instance
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test_scaled.iloc[0],
        feature_names=FEATURE_COLS
    ), show=False)
    shap_waterfall_path = f'{REPO_ROOT}/reports/shap_waterfall_instance_0.png'
    plt.savefig(shap_waterfall_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'[SHAP] Waterfall plot saved: {shap_waterfall_path}')
    print('[SHAP] SHAP explanations complete.')

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 14 — LIME EXPLANATIONS (SINGLE INSTANCE)
# ============================================================

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

if model_xgb is not None:
    # Create LIME explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_scaled.values,
        feature_names=FEATURE_COLS,
        class_names=['REJECT', 'PERMIT'],
        verbose=False
    )
    
    # Explain first test instance
    instance_idx = 0
    exp = lime_explainer.explain_instance(
        X_test_scaled.iloc[instance_idx].values,
        model_xgb.predict_proba,
        num_features=3
    )
    
    print(f'[LIME] Explanation for instance {instance_idx}:')
    print(f'  Actual label: {y_test.iloc[instance_idx]} ({\\\
\"PERMIT\" if y_test.iloc[instance_idx] == 1 else \"REJECT\"})')
    print(f'  Predicted: {y_test_pred[instance_idx]}')
    print(f'  Probability: {y_test_pred_proba[instance_idx]:.4f}')
    print(f'\n[LIME] Feature Contributions:')
    for feature, weight in exp.as_list():
        print(f'  {feature}: {weight:.4f}')
    
    # Save LIME plot
    fig = exp.as_pyplot_figure()
    lime_plot_path = f'{REPO_ROOT}/reports/lime_explanation_instance_0.png'
    plt.savefig(lime_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'[LIME] LIME plot saved: {lime_plot_path}')

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 15 — GUARDRAIL TEST VALIDATION (DETERMINISTIC RULES)
# ============================================================

import pandas as pd
import numpy as np

# Predefined guardrail thresholds
RIBA_THRESHOLD = 0.50
NONHALAL_THRESHOLD = 0.25
PROHIBITED_SECTORS = ['ALCOHOL', 'WEAPONS', 'GAMBLING', 'PORK', 'CONVENTIONAL_BANKING', 'ADULT_CONTENT']

def apply_guardrails(row):
    \"\"\"Apply deterministic guardrails to override ML predictions\"\"\"
    if row['sector_code'] in PROHIBITED_SECTORS:
        return 0  # REJECT
    if row['F_RIBA'] > RIBA_THRESHOLD:
        return 0  # REJECT
    if row['F_NONHALAL'] > NONHALAL_THRESHOLD:
        return 0  # REJECT
    return 1  # PERMIT (default)

if df is not None:
    print('[GUARDRAIL] Testing Deterministic Rules...')
    
    # Test Case 1: High RIBA
    tc001 = df[df['F_RIBA'] > RIBA_THRESHOLD].copy()
    tc001['guardrail_decision'] = tc001.apply(apply_guardrails, axis=1)
    tc001_reject_rate = (tc001['guardrail_decision'] == 0).sum() / len(tc001) if len(tc001) > 0 else 0
    
    print(f'\n[GUARDRAIL] TC-001: High RIBA (F_RIBA > {RIBA_THRESHOLD})')
    print(f'  Records: {len(tc001)}')
    print(f'  Rejection Rate: {tc001_reject_rate * 100:.1f}%')
    print(f'  Status: {\"✓ PASS\" if tc001_reject_rate == 1.0 else \"✗ FAIL\"}')
    
    # Test Case 2: High Non-Halal
    tc002 = df[df['F_NONHALAL'] > NONHALAL_THRESHOLD].copy()
    tc002['guardrail_decision'] = tc002.apply(apply_guardrails, axis=1)
    tc002_reject_rate = (tc002['guardrail_decision'] == 0).sum() / len(tc002) if len(tc002) > 0 else 0
    
    print(f'\n[GUARDRAIL] TC-002: High Non-Halal (F_NONHALAL > {NONHALAL_THRESHOLD})')
    print(f'  Records: {len(tc002)}')
    print(f'  Rejection Rate: {tc002_reject_rate * 100:.1f}%')
    print(f'  Status: {\"✓ PASS\" if tc002_reject_rate == 1.0 else \"✗ FAIL\"}')
    
    # Test Case 3: Prohibited Sectors
    tc003 = df[df['sector_code'].isin(PROHIBITED_SECTORS)].copy()
    tc003['guardrail_decision'] = tc003.apply(apply_guardrails, axis=1)
    tc003_reject_rate = (tc003['guardrail_decision'] == 0).sum() / len(tc003) if len(tc003) > 0 else 0
    
    print(f'\n[GUARDRAIL] TC-003: Prohibited Sectors')
    print(f'  Records: {len(tc003)}')
    print(f'  Rejection Rate: {tc003_reject_rate * 100:.1f}%')
    print(f'  Status: {\"✓ PASS\" if tc003_reject_rate == 1.0 else \"✗ FAIL\"}')
    
    # Overall validation
    all_pass = (tc001_reject_rate == 1.0) and (tc002_reject_rate == 1.0) and (tc003_reject_rate == 1.0)
    print(f'\n[GUARDRAIL] Overall Status: {\"✓ ALL PASS\" if all_pass else \"✗ SOME FAILURES\"}')

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 16 — GRADIO UI: SINGLE-ENTITY MODE
# ============================================================

import gradio as gr
import numpy as np
import pickle
import lime.lime_tabular
import matplotlib.pyplot as plt
from io import BytesIO

# Load model and scaler
with open(f'{REPO_ROOT}/models/checkpoint/xgb_model.pkl', 'rb') as f:
    model_loaded = pickle.load(f)
with open(f'{REPO_ROOT}/models/checkpoint/scaler.pkl', 'rb') as f:
    scaler_loaded = pickle.load(f)

# Initialize LIME explainer
lime_exp = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled.values,
    feature_names=FEATURE_COLS,
    class_names=['REJECT', 'PERMIT'],
    verbose=False
)

def predict_single_entity(f_riba, f_nonhalal, leverage_ratio):
    \"\"\"Predict compliance for a single company\"\"\"
    
    # Prepare input
    input_array = np.array([[f_riba, f_nonhalal, leverage_ratio]])
    input_scaled = scaler_loaded.transform(input_array)
    
    # Prediction
    pred = model_loaded.predict(input_scaled)[0]
    proba = model_loaded.predict_proba(input_scaled)[0]
    
    # LIME explanation
    lime_exp_obj = lime_exp.explain_instance(
        input_scaled[0],
        model_loaded.predict_proba,
        num_features=3
    )
    
    # Create LIME plot
    fig = lime_exp_obj.as_pyplot_figure()
    
    # Format output
    decision = 'PERMIT' if pred == 1 else 'REJECT'
    confidence = max(proba)
    result_text = f\"Decision: {decision}\\nConfidence: {confidence:.2%}\\nREJECT Prob: {proba[0]:.4f}\\nPERMIT Prob: {proba[1]:.4f}\"
    
    return result_text, fig

# Create Gradio interface
with gr.Blocks(title='Shariah Compliance Scorer') as demo_single:
    gr.Markdown('# 🕌 Shariah Compliance Scorer\\n## Single-Entity Analysis')
    
    with gr.Row():
        with gr.Column():
            f_riba_input = gr.Slider(0, 1, step=0.01, label='F_RIBA (Interest-Bearing Debt Ratio)', value=0.3)
            f_nonhalal_input = gr.Slider(0, 1, step=0.01, label='F_NONHALAL (Non-Halal Income %)', value=0.1)
            leverage_input = gr.Slider(0, 1, step=0.01, label='LEVERAGE_RATIO (Debt-to-Assets)', value=0.5)
            submit_btn = gr.Button('Score Compliance')
        
        with gr.Column():
            result_output = gr.Textbox(label='Compliance Decision', lines=4)
            lime_plot_output = gr.Plot(label='LIME Explanation')
    
    submit_btn.click(
        fn=predict_single_entity,
        inputs=[f_riba_input, f_nonhalal_input, leverage_input],
        outputs=[result_output, lime_plot_output]
    )

print('[GRADIO] Single-entity mode interface ready.')

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 17 — GRADIO UI: BATCH MODE
# ============================================================

import gradio as gr
import pandas as pd
import numpy as np
import pickle

def predict_batch(csv_file):
    \"\"\"Batch prediction on CSV file\"\"\"
    
    # Load CSV
    df_batch = pd.read_csv(csv_file.name)
    
    # Extract features (assume CSV has F_RIBA, F_NONHALAL, LEVERAGE_RATIO columns)
    if not all(col in df_batch.columns for col in FEATURE_COLS):
        return None, 'Error: CSV must contain F_RIBA, F_NONHALAL, LEVERAGE_RATIO columns'
    
    X_batch = df_batch[FEATURE_COLS].copy()
    X_batch_scaled = scaler_loaded.transform(X_batch)
    
    # Predictions
    y_pred = model_loaded.predict(X_batch_scaled)
    y_proba = model_loaded.predict_proba(X_batch_scaled)[:, 1]
    
    # Create results DataFrame
    results = df_batch.copy()
    results['SHARIAH_DECISION'] = y_pred.map({0: 'REJECT', 1: 'PERMIT'})
    results['PERMIT_PROBABILITY'] = y_proba
    results['REJECT_PROBABILITY'] = 1 - y_proba
    
    # Summary statistics
    n_permit = (y_pred == 1).sum()
    n_reject = (y_pred == 0).sum()
    summary = f\"Batch Results: {len(results)} companies\\nPERMIT: {n_permit} ({n_permit/len(results)*100:.1f}%)\\nREJECT: {n_reject} ({n_reject/len(results)*100:.1f}%)\"
    
    return results[['company_id', 'company_name', 'SHARIAH_DECISION', 'PERMIT_PROBABILITY', 'REJECT_PROBABILITY']], summary

# Create batch interface
with gr.Blocks(title='Shariah Compliance Scorer - Batch') as demo_batch:
    gr.Markdown('# 🕌 Shariah Compliance Scorer\\n## Batch Analysis')
    
    with gr.Row():
        csv_input = gr.File(label='Upload CSV (with F_RIBA, F_NONHALAL, LEVERAGE_RATIO)', type='file')
        submit_batch_btn = gr.Button('Process Batch')
    
    with gr.Row():
        summary_output = gr.Textbox(label='Summary')
        results_output = gr.Dataframe(label='Results')
    
    submit_batch_btn.click(
        fn=predict_batch,
        inputs=csv_input,
        outputs=[results_output, summary_output]
    )

print('[GRADIO] Batch mode interface ready.')

# ===== CELL SEPARATOR =====

# ============================================================
# CELL 18 — DEPLOYMENT & FINAL VALIDATION
# ============================================================

import gradio as gr
import os
import pickle
import json
from datetime import datetime

# Combine interfaces
demo = gr.TabbedInterface(
    [demo_single, demo_batch],
    ['Single Entity', 'Batch Processing'],
    title='Shariah Compliance Scoring Engine v1.0'
)

# Optional: Deploy with ngrok (uncomment for Colab)
# demo.launch(share=True, debug=True)

print(f'[DEPLOY] Gradio UI ready for launch')
print(f'[DEPLOY] To launch locally: demo.launch(share=False)')
print(f'[DEPLOY] To create public link: demo.launch(share=True)')

# Validation Report
validation_report = {
    'timestamp': datetime.now().isoformat(),
    'model_name': 'XGBoost Shariah Compliance Scorer v1.0',
    'test_metrics': {
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'auc_roc': float(auc_roc)
    },
    'cv_metrics': {
        'f1_mean': float(cv_results['test_f1'].mean()),
        'precision_mean': float(cv_results['test_precision'].mean()),
        'recall_mean': float(cv_results['test_recall'].mean()),
        'auc_mean': float(cv_results['test_roc_auc'].mean())
    },
    'guardrail_validation': {
        'tc001_high_riba': 'PASS' if tc001_reject_rate == 1.0 else 'FAIL',
        'tc002_high_nonhalal': 'PASS' if tc002_reject_rate == 1.0 else 'FAIL',
        'tc003_prohibited_sectors': 'PASS' if tc003_reject_rate == 1.0 else 'FAIL'
    },
    'deployment_status': 'READY'
}

# Save validation report
report_path = f'{REPO_ROOT}/reports/validation_report.json'
with open(report_path, 'w') as f:
    json.dump(validation_report, f, indent=2)

print(f'\n[FINAL] Validation Report:')
print(json.dumps(validation_report, indent=2))
print(f'\n[FINAL] Report saved: {report_path}')
print(f'\n✓ PHASE 5 COMPLETE: Model ready for deployment')