#!/bin/bash

################################################################################
# Status Check - Verify Project Setup & Data Files
# 
# Usage: ./check_status.sh
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

check_ok() {
    echo -e "${GREEN}[✓]${NC} $1"
}

check_fail() {
    echo -e "${RED}[✗]${NC} $1"
}

check_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_header() {
    echo -e "\n${CYAN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}\n"
}

echo -e "\n${BLUE}🔍 System Status Check${NC}\n"

# === PYTHON ===
print_header "Python Environment"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    check_ok "Python 3 installed: $PYTHON_VERSION"
else
    check_fail "Python 3 not found"
    exit 1
fi

# Check dependencies
echo ""
python3 << 'EOFPYTHON'
import sys

deps = {
    'pandas': 'Data processing',
    'numpy': 'Numerical operations',
    'scikit-learn': 'Machine learning',
    'xgboost': 'XGBoost classifier',
    'joblib': 'Model serialization',
}

missing = []
for pkg, desc in deps.items():
    try:
        __import__(pkg)
        print(f"\033[0;32m[✓]\033[0m {pkg:<15} ({desc})")
    except ImportError:
        print(f"\033[0;31m[✗]\033[0m {pkg:<15} ({desc})")
        missing.append(pkg)

if missing:
    print(f"\n\033[1;33m[!] Missing packages: {', '.join(missing)}\033[0m")
    print(f"    Run: pip install -r requirements.txt")

EOFPYTHON

# === DIRECTORIES ===
print_header "Project Structure"

dirs=(
    "data/raw"
    "data/processed"
    "src"
    "models"
    "reports"
    "notebooks"
    "tests"
)

for dir in "${dirs[@]}"; do
    if [ -d "$PROJECT_DIR/$dir" ]; then
        count=$(find "$PROJECT_DIR/$dir" -type f 2>/dev/null | wc -l)
        check_ok "$dir/ ($count files)"
    else
        check_fail "$dir/ (missing)"
    fi
done

# === DATA FILES ===
print_header "Data Files"

data_files=(
    "data/raw/combined_financial_data_idx.csv"
    "data/processed/companies_processed.csv"
    "data/processed/companies_with_features.csv"
    "data/processed/companies_with_labels.csv"
)

for file in "${data_files[@]}"; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        size=$(du -h "$PROJECT_DIR/$file" | cut -f1)
        rows=$(wc -l < "$PROJECT_DIR/$file" 2>/dev/null | tail -1)
        check_ok "$file ($rows rows, $size)"
    else
        check_warn "$file (not generated yet)"
    fi
done

# === MODEL FILES ===
print_header "Model Artifacts"

model_files=(
    "models/xgb_shariah_model.pkl"
    "models/xgb_scaler.pkl"
)

for file in "${model_files[@]}"; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        size=$(du -h "$PROJECT_DIR/$file" | cut -f1)
        check_ok "$file ($size)"
    else
        check_warn "$file (not trained yet)"
    fi
done

# === REPORTS ===
print_header "Report Files"

reports=(
    "reports/model_predictions_explanations.csv"
)

for file in "${reports[@]}"; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        size=$(du -h "$PROJECT_DIR/$file" | cut -f1)
        rows=$(wc -l < "$PROJECT_DIR/$file" 2>/dev/null | tail -1)
        check_ok "$file ($rows predictions, $size)"
    else
        check_warn "$file (not generated yet)"
    fi
done

# === SOURCE CODE ===
print_header "Source Code Modules"

modules=(
    "src/data_loader.py"
    "src/shariah_features.py"
    "src/shariah_classifier.py"
    "src/xgb_trainer.py"
    "src/explainability.py"
)

for file in "${modules[@]}"; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        lines=$(wc -l < "$PROJECT_DIR/$file")
        check_ok "$file ($lines lines)"
    else
        check_fail "$file (missing)"
    fi
done

# === SCRIPTS ===
print_header "Executable Scripts"

scripts=(
    "run.sh"
    "view_results.sh"
    "quick_test.sh"
)

for script in "${scripts[@]}"; do
    if [ -f "$PROJECT_DIR/$script" ]; then
        if [ -x "$PROJECT_DIR/$script" ]; then
            check_ok "$script (executable)"
        else
            check_warn "$script (not executable - run: chmod +x $script)"
        fi
    else
        check_fail "$script (missing)"
    fi
done

# === SUMMARY ===
print_header "Status Summary"

echo -e "${BLUE}Quick Start Commands:${NC}\n"
echo "  1. Check everything is working:"
echo -e "     ${CYAN}./quick_test.sh${NC}\n"
echo "  2. Run the complete pipeline:"
echo -e "     ${CYAN}./run.sh${NC}\n"
echo "  3. View results:"
echo -e "     ${CYAN}./view_results.sh${NC}\n"
echo "  4. Get help:"
echo -e "     ${CYAN}./run.sh --help${NC}\n"

if [ -f "$PROJECT_DIR/models/xgb_shariah_model.pkl" ] && [ -f "$PROJECT_DIR/reports/model_predictions_explanations.csv" ]; then
    echo -e "${GREEN}✅ Status: Ready to use!${NC}"
    echo "   All data, models, and reports are generated."
else
    echo -e "${YELLOW}⏳ Status: Setup required${NC}"
    echo "   Run './run.sh' to generate data and train model."
fi

echo ""
