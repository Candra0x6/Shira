#!/bin/bash

################################################################################
# Generate Technical Report Script
# 
# Generates comprehensive technical documentation including:
# - Model architecture and hyperparameters
# - Performance metrics and statistics
# - Feature importance analysis
# - Data pipeline summary
# - Deployment recommendations
#
# Usage: ./generate_report.sh
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
    echo -e "${GREEN}[✓]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_header "📄 Technical Report Generator"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 not found"
    exit 1
fi
log_success "Python 3 found"

# Check if the script exists
if [ ! -f "$PROJECT_DIR/src/generate_technical_report.py" ]; then
    log_error "generate_technical_report.py not found"
    exit 1
fi
log_success "Report generator script found"

log_step "Generating technical reports..."
python3 "$PROJECT_DIR/src/generate_technical_report.py" "$PROJECT_DIR"

if [ $? -eq 0 ]; then
    print_header "✅ Report Generation Complete"
    echo -e "${GREEN}Technical documentation has been generated successfully!${NC}\n"
    
    echo -e "${YELLOW}📁 Generated Files:${NC}"
    echo "  • reports/TECHNICAL_REPORT.md (Comprehensive technical report)"
    echo "  • reports/technical_report.json (Structured report data)"
    
    echo -e "\n${YELLOW}💡 Tips:${NC}"
    echo "  • Read TECHNICAL_REPORT.md for detailed technical analysis"
    echo "  • Use technical_report.json for programmatic access"
    echo "  • Share reports with stakeholders and team members"
    
    echo -e "\n${YELLOW}🔗 Quick Links:${NC}"
    echo "  • Model Performance: Check Section 2 of TECHNICAL_REPORT.md"
    echo "  • Feature Analysis: Check Section 4 of TECHNICAL_REPORT.md"
    echo "  • Deployment Guide: Check Section 6 of TECHNICAL_REPORT.md"
    
else
    log_error "Report generation failed"
    exit 1
fi

log_success "Report generation completed!"
