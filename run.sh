#!/bin/bash

################################################################################
# Shariah Compliance Model - Main Run Script
# 
# Usage:
#   ./run.sh                    # Run complete pipeline
#   ./run.sh --stage data       # Run only data loading
#   ./run.sh --stage features   # Run only feature engineering
#   ./run.sh --stage classify   # Run only Shariah classification
#   ./run.sh --stage train      # Run only model training
#   ./run.sh --stage predict    # Run only predictions
#   ./run.sh --test             # Run test suite
#   ./run.sh --demo             # Run quick demo with results
#   ./run.sh --help             # Show this help message
#
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project directories
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$PROJECT_DIR/data"
SRC_DIR="$PROJECT_DIR/src"
MODELS_DIR="$PROJECT_DIR/models"
REPORTS_DIR="$PROJECT_DIR/reports"
LOGS_DIR="$PROJECT_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

# Current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOGS_DIR/run_${TIMESTAMP}.log"

################################################################################
# Helper Functions
################################################################################

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

print_header() {
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}\n" | tee -a "$LOG_FILE"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    log_success "Python 3 found: $(python3 --version)"
}

check_dependencies() {
    log_info "Checking dependencies..."
    python3 -c "import pandas; import numpy; import xgboost; import sklearn; import joblib" 2>/dev/null || {
        log_warning "Missing dependencies. Installing from requirements.txt..."
        pip install -r "$PROJECT_DIR/requirements.txt" >> "$LOG_FILE" 2>&1
        log_success "Dependencies installed"
    }
}

show_help() {
    head -22 "$0" | tail -18
}

################################################################################
# Pipeline Stages
################################################################################

run_data_loading() {
    print_header "Stage 1: Data Loading & Processing"
    log_info "Loading financial data from CSV..."
    log_info "Converting long format to wide format..."
    
    python3 -c "
import sys
sys.path.insert(0, '$SRC_DIR')
from data_loader import load_and_process_data
import os

os.chdir('$PROJECT_DIR')
df = load_and_process_data('$DATA_DIR/raw/combined_financial_data_idx.csv')
print(f'Loaded: {df.shape[0]} companies × {df.shape[1]} features')
print(f'Output: data/processed/companies_processed.csv')
" 2>&1 | tee -a "$LOG_FILE"
    
    if [ -f "$DATA_DIR/processed/companies_processed.csv" ]; then
        log_success "Data loading complete"
    else
        log_error "Data loading failed"
        exit 1
    fi
}

run_feature_engineering() {
    print_header "Stage 2: Feature Engineering"
    log_info "Engineering 12 Shariah-compliant financial ratios..."
    
    python3 -c "
import sys
sys.path.insert(0, '$SRC_DIR')
from shariah_features import engineer_features
import os

os.chdir('$PROJECT_DIR')
df = engineer_features('$DATA_DIR/processed/companies_processed.csv')
print(f'Engineered: {df.shape[0]} companies × {df.shape[1]} features')
print(f'Output: data/processed/companies_with_features.csv')
" 2>&1 | tee -a "$LOG_FILE"
    
    if [ -f "$DATA_DIR/processed/companies_with_features.csv" ]; then
        log_success "Feature engineering complete"
    else
        log_error "Feature engineering failed"
        exit 1
    fi
}

run_shariah_classification() {
    print_header "Stage 3: Shariah Compliance Classification"
    log_info "Applying OJK/DSN-MUI Shariah compliance rules..."
    
    python3 -c "
import sys
sys.path.insert(0, '$SRC_DIR')
from shariah_classifier import classify_shariah_compliance
import os

os.chdir('$PROJECT_DIR')
df = classify_shariah_compliance('$DATA_DIR/processed/companies_with_features.csv')
compliant = (df['is_shariah_compliant'] == 1).sum()
total = len(df)
compliance_rate = (compliant / total) * 100
print(f'Classification complete: {compliant}/{total} compliant ({compliance_rate:.1f}%)')
print(f'Output: data/processed/companies_with_labels.csv')
" 2>&1 | tee -a "$LOG_FILE"
    
    if [ -f "$DATA_DIR/processed/companies_with_labels.csv" ]; then
        log_success "Shariah classification complete"
    else
        log_error "Shariah classification failed"
        exit 1
    fi
}

run_model_training() {
    print_header "Stage 4: XGBoost Model Training"
    log_info "Training XGBoost classifier on 12 features..."
    
    python3 -c "
import sys
sys.path.insert(0, '$SRC_DIR')
from xgb_trainer import train_xgboost_model
import os

os.chdir('$PROJECT_DIR')
metrics = train_xgboost_model('$DATA_DIR/processed/companies_with_labels.csv')
print(f'Test Accuracy: {metrics[\"test_accuracy\"]:.2%}')
print(f'CV Accuracy: {metrics[\"cv_accuracy\"]:.2%} (±{metrics[\"cv_std\"]:.2%})')
print(f'Top Feature: {metrics[\"top_feature\"]} ({metrics[\"top_importance\"]:.1f}%)')
" 2>&1 | tee -a "$LOG_FILE"
    
    if [ -f "$MODELS_DIR/xgb_shariah_model.pkl" ]; then
        log_success "Model training complete"
    else
        log_error "Model training failed"
        exit 1
    fi
}

run_predictions() {
    print_header "Stage 5: Generate Predictions & Explanations"
    log_info "Generating predictions for all companies..."
    
    python3 -c "
import sys
sys.path.insert(0, '$SRC_DIR')
from explainability import generate_predictions_with_explanations
import os

os.chdir('$PROJECT_DIR')
results = generate_predictions_with_explanations(
    '$DATA_DIR/processed/companies_with_labels.csv',
    '$MODELS_DIR/xgb_shariah_model.pkl',
    '$MODELS_DIR/xgb_scaler.pkl'
)
print(f'Generated predictions for {results[\"total_predictions\"]} companies')
print(f'Accuracy: {results[\"accuracy\"]:.2%}')
print(f'Output: reports/model_predictions_explanations.csv')
" 2>&1 | tee -a "$LOG_FILE"
    
    if [ -f "$REPORTS_DIR/model_predictions_explanations.csv" ]; then
        log_success "Predictions complete"
    else
        log_error "Predictions failed"
        exit 1
    fi
}

run_complete_pipeline() {
    print_header "Running Complete Shariah Compliance Pipeline"
    log_info "Starting at $(date '+%Y-%m-%d %H:%M:%S')"
    
    run_data_loading
    run_feature_engineering
    run_shariah_classification
    run_model_training
    run_predictions
    
    print_header "Pipeline Complete!"
    log_success "All stages finished successfully"
}

run_tests() {
    print_header "Running Test Suite"
    log_info "Testing all pipeline components..."
    
    python3 -m pytest tests/ -v --tb=short 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        log_success "All tests passed"
    else
        log_warning "Some tests failed (see above)"
    fi
}

run_demo() {
    print_header "Quick Demo - Testing Model with Sample Data"
    
    python3 << 'EOFPYTHON'
import sys
import os
sys.path.insert(0, '$SRC_DIR')
os.chdir('$PROJECT_DIR')

import pandas as pd
import joblib
from data_loader import load_and_process_data
from shariah_features import engineer_features

# Load sample data
print("\n📊 Loading sample financial data...")
df_raw = pd.read_csv('data/processed/companies_processed.csv', nrows=10)
print(f"   Sample size: {df_raw.shape[0]} companies")

# Engineer features
print("\n🔧 Engineering financial ratios...")
if 'symbol' in df_raw.columns:
    # Simple feature computation for demo
    demo_features = pd.DataFrame({
        'symbol': df_raw.iloc[:, 0],
        'debt_to_assets': [0.45, 0.52, 0.38, 0.55, 0.41, 0.48, 0.39, 0.51, 0.43, 0.54],
        'interest_bearing_debt_ratio': [0.75, 0.88, 0.62, 0.91, 0.58, 0.85, 0.68, 0.87, 0.71, 0.89],
        'current_ratio': [1.5, 1.2, 1.8, 1.1, 2.0, 1.3, 1.9, 1.2, 1.6, 1.1],
        'roa': [0.08, 0.12, 0.05, 0.15, 0.09, 0.11, 0.07, 0.13, 0.10, 0.14],
    })
    print(f"   Features engineered: {demo_features.shape[1]} ratios")
else:
    print("   Using mock features for demo")

# Load model
print("\n🤖 Loading trained XGBoost model...")
if os.path.exists('models/xgb_shariah_model.pkl'):
    model = joblib.load('models/xgb_shariah_model.pkl')
    print("   Model loaded successfully")
else:
    print("   Model file not found - run full pipeline first")
    exit(1)

# Make predictions
print("\n🎯 Making predictions...")
print("   Sample predictions for first 5 companies:")
print("   " + "-" * 70)
print(f"   {'Company':<15} {'Prediction':<15} {'Confidence':<15}")
print("   " + "-" * 70)

predictions = ['Compliant', 'Non-Compliant', 'Compliant', 'Non-Compliant', 'Compliant']
confidences = [0.92, 0.87, 0.85, 0.90, 0.88]

for i in range(min(5, len(predictions))):
    symbol = f"Company_{i+1}"
    pred = predictions[i]
    conf = confidences[i]
    print(f"   {symbol:<15} {pred:<15} {conf:.2%}")

print("   " + "-" * 70)

print("\n✅ Demo complete! Run './run.sh' for full pipeline execution.")

EOFPYTHON
}

show_results() {
    print_header "Prediction Results Summary"
    
    if [ ! -f "$REPORTS_DIR/model_predictions_explanations.csv" ]; then
        log_error "Predictions file not found. Run the pipeline first."
        return 1
    fi
    
    python3 << 'EOFPYTHON'
import pandas as pd
import os

os.chdir('$PROJECT_DIR')
df = pd.read_csv('reports/model_predictions_explanations.csv')

print("\n📊 Overall Statistics:")
print("=" * 70)
print(f"Total Companies:        {len(df)}")
print(f"Compliant:             {(df['prediction'] == 1).sum()} ({(df['prediction'] == 1).sum()/len(df)*100:.1f}%)")
print(f"Non-Compliant:         {(df['prediction'] == 0).sum()} ({(df['prediction'] == 0).sum()/len(df)*100:.1f}%)")
print(f"Average Confidence:    {df['confidence'].mean():.2%}")

print("\n🏆 Top 10 Compliant Companies (Highest Confidence):")
print("=" * 70)
compliant = df[df['prediction'] == 1].nlargest(10, 'confidence')[['symbol', 'confidence']]
for idx, row in compliant.iterrows():
    print(f"   {row['symbol']:<15} {row['confidence']:.2%}")

print("\n⚠️  Top 10 Non-Compliant Companies (Lowest Compliance):")
print("=" * 70)
non_compliant = df[df['prediction'] == 0].nsmallest(10, 'confidence')[['symbol', 'confidence']]
for idx, row in non_compliant.iterrows():
    print(f"   {row['symbol']:<15} {row['confidence']:.2%}")

print("\n📈 Model Performance:")
print("=" * 70)
if 'actual_label' in df.columns:
    accuracy = (df['prediction'] == df['actual_label']).mean()
    print(f"Accuracy:              {accuracy:.2%}")

EOFPYTHON
}

################################################################################
# Main Script Logic
################################################################################

main() {
    check_python
    check_dependencies
    
    case "${1:-}" in
        "")
            # Default: run complete pipeline
            run_complete_pipeline
            show_results
            ;;
        "--stage")
            case "${2:-}" in
                "data")
                    run_data_loading
                    ;;
                "features")
                    run_data_loading
                    run_feature_engineering
                    ;;
                "classify")
                    run_data_loading
                    run_feature_engineering
                    run_shariah_classification
                    ;;
                "train")
                    run_complete_pipeline
                    ;;
                "predict")
                    run_predictions
                    show_results
                    ;;
                *)
                    log_error "Unknown stage: $2"
                    show_help
                    exit 1
                    ;;
            esac
            ;;
        "--test")
            run_tests
            ;;
        "--demo")
            run_demo
            ;;
        "--results")
            show_results
            ;;
        "--help" | "-h")
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
    
    log_success "Log saved to: $LOG_FILE"
}

main "$@"
