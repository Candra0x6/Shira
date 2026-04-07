#!/bin/bash

##############################################################################
# Master Test Runner for Shariah Compliance Model
# Runs all test suites and generates comprehensive report
##############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TESTS_DIR="$SCRIPT_DIR"

echo ""
echo "================================================================================"
echo "  SHARIAH COMPLIANCE MODEL - COMPREHENSIVE TEST SUITE"
echo "================================================================================"
echo ""
echo "Test Directory: $TESTS_DIR"
echo "Project Root:   $PROJECT_ROOT"
echo ""

# Counters
TOTAL_PASSED=0
TOTAL_FAILED=0
TOTAL_TESTS=0

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${BLUE}Using: $PYTHON_VERSION${NC}"
echo ""

##############################################################################
# Function to run a test file and capture results
##############################################################################
run_test_file() {
    local test_file=$1
    local test_name=$2
    
    echo "================================================================================"
    echo -e "  ${BLUE}Running: $test_name${NC}"
    echo "================================================================================"
    echo ""
    
    # Run the test and capture output
    if python3 "$test_file" 2>&1 | tee /tmp/test_output.log; then
        # Extract counts from output (simple pattern matching)
        local output=$(cat /tmp/test_output.log)
        
        # Try to extract passed/failed counts
        if grep -q "Results:" /tmp/test_output.log; then
            local results=$(grep "Results:" /tmp/test_output.log | tail -1)
            local passed=$(echo "$results" | grep -oE "[0-9]+ passed" | grep -oE "[0-9]+")
            local failed=$(echo "$results" | grep -oE "[0-9]+ failed" | grep -oE "[0-9]+")
            
            echo ""
            echo -e "${GREEN}✓ $test_name COMPLETED${NC}"
            echo "  Passed: $passed"
            echo "  Failed: $failed"
            
            TOTAL_PASSED=$((TOTAL_PASSED + passed))
            TOTAL_FAILED=$((TOTAL_FAILED + failed))
        fi
    else
        echo ""
        echo -e "${RED}✗ $test_name FAILED${NC}"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi
    
    echo ""
}

##############################################################################
# Feature Engineering Tests
##############################################################################
if [ -f "$TESTS_DIR/test_features.py" ]; then
    run_test_file "$TESTS_DIR/test_features.py" "FEATURE ENGINEERING TESTS"
else
    echo -e "${YELLOW}⊘ test_features.py not found${NC}"
fi

##############################################################################
# Model Prediction Tests
##############################################################################
if [ -f "$TESTS_DIR/test_model.py" ]; then
    run_test_file "$TESTS_DIR/test_model.py" "MODEL PREDICTION TESTS"
else
    echo -e "${YELLOW}⊘ test_model.py not found${NC}"
fi

##############################################################################
# SHAP Explainability Tests
##############################################################################
if [ -f "$TESTS_DIR/test_shap.py" ]; then
    run_test_file "$TESTS_DIR/test_shap.py" "SHAP EXPLAINABILITY TESTS"
else
    echo -e "${YELLOW}⊘ test_shap.py not found${NC}"
fi

##############################################################################
# Integration Tests
##############################################################################
if [ -f "$TESTS_DIR/test_integration.py" ]; then
    run_test_file "$TESTS_DIR/test_integration.py" "INTEGRATION TESTS"
else
    echo -e "${YELLOW}⊘ test_integration.py not found${NC}"
fi

##############################################################################
# Final Report
##############################################################################
echo "================================================================================"
echo "  FINAL TEST REPORT"
echo "================================================================================"
echo ""

if [ $TOTAL_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED ($TOTAL_PASSED/$TOTAL_PASSED)${NC}"
    echo ""
    echo "The model is ready for:"
    echo "  ✓ Production deployment"
    echo "  ✓ Business use (compliance checking)"
    echo "  ✓ SHAP-based explainability"
    echo "  ✓ Batch processing"
    echo ""
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo "  Passed: $TOTAL_PASSED"
    echo "  Failed: $TOTAL_FAILED"
    echo ""
    echo "Please review the failures above and fix issues before deployment."
    echo ""
    exit 1
fi
