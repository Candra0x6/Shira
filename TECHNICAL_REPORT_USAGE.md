# Technical Report Generation

## Overview

The Shariah Compliance Model includes an automated technical report generation system that creates comprehensive documentation of model architecture, performance, and deployment recommendations.

## Quick Start

### Generate Reports
```bash
# Option 1: Using bash wrapper (easiest)
./generate_report.sh

# Option 2: Direct Python execution
python3 src/generate_technical_report.py

# Option 3: As part of full pipeline
./run.sh
```

## Generated Reports

### 1. TECHNICAL_REPORT.md
Comprehensive markdown technical report containing:

- **Section 1: Model Architecture**
  - Model name, type, and framework
  - Input/output specifications
  - Hyperparameters (n_estimators, max_depth, learning_rate, etc.)
  - Training configuration and data statistics

- **Section 2: Performance Metrics**
  - Overall performance (Accuracy, Precision, Recall, F1, AUC)
  - Confusion matrix
  - Classification metrics (Sensitivity, Specificity)

- **Section 3: Prediction Statistics**
  - Total predictions and distribution
  - Confidence score statistics (mean, median, std dev, percentiles)
  - Confidence distribution breakdown

- **Section 4: Feature Importance**
  - Top 5 most important features (ranked)
  - Complete ranking of all 12 features used

- **Section 5: Data Pipeline**
  - Raw data overview (89,243 rows × 7 columns)
  - Processed data (495 companies × 21 features)
  - Data transformation steps

- **Section 6: Deployment Recommendations**
  - Performance assessment
  - Confidence level analysis
  - Data balance evaluation
  - Operational recommendations

- **Section 7: Technical Specifications**
  - Dependencies and requirements
  - Model file locations
  - Data file paths
  - Output file descriptions

- **Section 8: Conclusion**
  - Summary of model capabilities
  - Production readiness assessment

### 2. technical_report.json
Structured JSON file with all report data in programmatic format:

```json
{
  "report_metadata": {
    "generated_at": "2026-04-07 13:19:37",
    "model_version": "1.0.0",
    "report_type": "Technical Documentation"
  },
  "model_summary": { ... },
  "prediction_statistics": { ... },
  "feature_importance": { ... },
  "data_pipeline": { ... },
  "recommendations": { ... }
}
```

## Report Features

### Model Architecture Section
- Model type: XGBoost Classifier
- Input features: 12 (financial ratios)
- Output: Binary classification (Compliant / Non-Compliant)
- Framework: XGBoost
- Training date: Automatically captured
- Model file size: 100KB

### Performance Metrics
```
Training Accuracy:    94.95%
Test Accuracy:        92.00%
Precision:            92.68%
Recall:               97.44%
F1 Score:             95.00%
AUC Score:            92.83%
Matthews Correlation: 0.7565
Cohen's Kappa:        0.7506
```

### Feature Importance Rankings
```
1. total_liabilities              54.45%
2. nonhalal_revenue_percent        7.60%
3. roe                             5.62%
4. debt_to_equity                  4.62%
5. net_revenue                     4.32%
6-12. [Additional features...]
```

### Data Pipeline Documentation
- Raw Data: 89,243 rows (7.0 MB)
- Processed: 495 companies with 21 features
- Labeled: 495 companies with Shariah compliance labels
- Transformations: 8 automated steps documented

### Deployment Recommendations
- Production readiness: ✅ EXCELLENT
- Performance status: High test accuracy (92%)
- Confidence assessment: High AUC (92.83%)
- Data balance: Monitored
- Update frequency: Annual or as needed

## File Locations

```
/reports/
├── TECHNICAL_REPORT.md              # Main markdown report
├── technical_report.json            # Structured JSON data
├── model_predictions_explanations.csv
└── [other reports]

/src/
└── generate_technical_report.py     # Report generator script

/
└── generate_report.sh               # Bash wrapper script
```

## Python Script Details

### TechnicalReportGenerator Class

Main class for generating technical reports:

```python
from src.generate_technical_report import TechnicalReportGenerator

# Initialize generator
generator = TechnicalReportGenerator(project_dir="/path/to/project")

# Generate all reports
generator.generate_all_reports()

# Or generate specific formats
markdown = generator.generate_markdown_report()
json_report = generator.generate_json_report()
```

### Key Methods

- `load_model_artifacts()` - Loads model, scaler, and metadata
- `load_data()` - Loads processed data and predictions
- `generate_model_summary()` - Creates model architecture summary
- `generate_prediction_statistics()` - Calculates prediction stats
- `generate_feature_importance()` - Ranks features by importance
- `generate_data_pipeline_summary()` - Documents data transformation
- `generate_deployment_recommendations()` - Creates deployment guidance
- `generate_markdown_report()` - Builds markdown format
- `generate_json_report()` - Exports JSON format
- `generate_all_reports()` - Generates all formats

## Integration with Pipeline

The technical report generator is fully integrated into the main pipeline:

```bash
./run.sh
├── Stage 1: Data Loading & Processing
├── Stage 2: Feature Engineering
├── Stage 3: Shariah Compliance Classification
├── Stage 4: XGBoost Model Training
├── Stage 5: Generate Predictions & Explanations
└── Stage 6: Generate Technical Documentation ← NEW
```

## Report Updates

Reports are generated with a timestamp and overwrite previous versions. The generation process:

1. Loads trained model artifacts
2. Loads processed data and predictions
3. Calculates all statistics
4. Generates markdown document
5. Exports JSON data
6. Saves both files to `/reports/` directory

## Use Cases

### 1. Stakeholder Communication
Share `TECHNICAL_REPORT.md` with non-technical stakeholders to understand:
- Model performance
- Data sources
- Deployment status
- Recommendations

### 2. Documentation
Use as official technical documentation for:
- Model cards
- Technical specifications
- Deployment guides
- Training records

### 3. Programmatic Access
Use `technical_report.json` for:
- Automated monitoring dashboards
- Performance tracking systems
- Integration with other tools
- Data pipeline validation

### 4. Compliance & Audit
Reports provide:
- Training date and configuration
- Performance metrics on test data
- Data sources and transformations
- Deployment recommendations

## Customization

To customize the report generator:

1. Edit `src/generate_technical_report.py`
2. Modify report sections in `generate_markdown_report()`
3. Update deployment recommendations logic
4. Add custom metrics or analyses
5. Re-run `./generate_report.sh`

## Dependencies

- Python 3.x
- pandas
- numpy
- xgboost
- joblib

No additional dependencies required for report generation.

## Troubleshooting

### Report generation fails
```bash
# Check if all data files exist
./check_status.sh

# Verify Python environment
python3 -c "import pandas, numpy, xgboost, joblib"
```

### Missing report files
```bash
# Regenerate reports
./generate_report.sh

# Or as part of full pipeline
./run.sh
```

### JSON serialization errors
- Ensure model artifacts are valid
- Check if data files are accessible
- Verify JSON export doesn't contain incompatible types

## Next Steps

1. Review `TECHNICAL_REPORT.md` for full details
2. Share with stakeholders
3. Use JSON report for integration
4. Monitor model performance over time
5. Retrain as needed and regenerate reports

## Support

For issues or questions about the technical report generator:
1. Check the report content for accuracy
2. Review data pipeline in Section 5
3. Verify model artifacts exist
4. Run `./check_status.sh` to verify system state

---

**Last Updated:** 2026-04-07
**Report Version:** 1.0.0
**Generator Version:** 1.0.0
