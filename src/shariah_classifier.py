"""
Shariah Compliance Classifier Module
Apply OJK/DSN-MUI rules to classify companies as Shariah-compliant or non-compliant.

OJK/DSN-MUI Shariah Screening Rules:
1. Companies in prohibited sectors (tobacco, alcohol, gambling, pornography, weapons)
2. Companies with excessive debt (debt-to-assets > 30%)
3. Companies with excessive interest income (interest-income-ratio > 5%)
4. Companies with excessive interest-bearing debt (interest-bearing-debt-ratio > 30%)
5. Negative financial health indicators (negative equity, negative net income)

Compliance Label: 1 if meets all criteria, 0 otherwise
"""

import pandas as pd
import numpy as np
import json
import logging
import os
from typing import Tuple, Dict

logger = logging.getLogger(__name__)


class ShariaComplianceClassifier:
    """Classify companies as Shariah-compliant using OJK/DSN-MUI rules."""

    # Prohibited sectors per OJK/DSN-MUI guidelines
    PROHIBITED_SECTORS = [
        "Tobacco",
        "Alcohol",
        "Gambling",
        "Pornography",
        "Weapons",
        "Entertainment",
    ]

    # OJK/DSN-MUI Threshold Rules
    THRESHOLDS = {
        "debt_to_assets_max": 0.45,  # Max 45% debt (Source: DSN-MUI POJK Baseline)
        "interest_income_ratio_max": 0.10,  # Max 10% non-halal (Standard MUI)
        "interest_bearing_debt_ratio_max": 0.45,  # Max 45% interest-bearing debt (Consistent with riba cap)
        "roa_min": -0.10,  # Min -10% ROA
        "equity_ratio_min": -0.20,  # Min -20% equity ratio
    }

    def __init__(self, sector_mapping_path: str = None, verbose: bool = True):
        self.verbose = verbose
        self.sector_mapping = {}
        self.df_features = None
        self.labels = None

        # Load sector mapping if provided
        if sector_mapping_path and os.path.exists(sector_mapping_path):
            self._load_sector_mapping(sector_mapping_path)

    def _load_sector_mapping(self, path: str) -> None:
        """Load sector mapping from JSON file."""
        logger.info(f"[CLASSIFIER] Loading sector mapping from {path}")
        with open(path, "r") as f:
            self.sector_mapping = json.load(f)
        logger.info(
            f"[CLASSIFIER] Loaded mapping for {len(self.sector_mapping)} companies"
        )

    def load_features(self, csv_path: str) -> pd.DataFrame:
        """Load engineered features."""
        logger.info(f"[CLASSIFIER] Loading features from {csv_path}")
        self.df_features = pd.read_csv(csv_path)
        logger.info(
            f"[CLASSIFIER] Loaded {len(self.df_features)} companies with {len(self.df_features.columns) - 1} features"
        )
        return self.df_features

    def get_sector(self, symbol: str) -> str:
        """Get sector for a company symbol."""
        return self.sector_mapping.get(symbol, "Other")

    def apply_sector_rule(self, symbol: str) -> bool:
        """Check if company is in prohibited sector."""
        sector = self.get_sector(symbol)
        is_compliant = sector not in self.PROHIBITED_SECTORS
        return is_compliant

    def apply_financial_rules(self, row: pd.Series) -> Tuple[bool, Dict]:
        """
        Apply OJK/DSN-MUI financial rules to a company.

        Returns:
            (is_compliant, details_dict)
        """
        details = {
            "debt_to_assets_ok": True,
            "interest_income_ok": True,
            "interest_bearing_debt_ok": True,
            "profitability_ok": True,
            "equity_ok": True,
        }

        # Rule 1: Debt-to-Assets <= 45%
        debt_to_assets = row.get("debt_to_assets", row.get("f_riba"))
        if pd.notna(debt_to_assets):
            details["debt_to_assets_ok"] = (
                debt_to_assets <= self.THRESHOLDS["debt_to_assets_max"]
            )
            details["debt_to_assets_value"] = float(debt_to_assets)
        else:
            details["debt_to_assets_ok"] = False  # Fail if no debt data

        # Rule 2: Interest Income Ratio <= 10%
        interest_ratio = row.get("interest_income_ratio", row.get("f_nonhalal", row.get("nonhalal_revenue_percent")))
        if pd.notna(interest_ratio):
            details["interest_income_ok"] = (
                interest_ratio <= self.THRESHOLDS["interest_income_ratio_max"]
            )
            details["interest_income_value"] = float(interest_ratio)
        else:
            details["interest_income_ok"] = False

        # Rule 3: Interest-Bearing Debt Ratio <= 45% (Using f_riba as proxy)
        int_debt_ratio = row.get("interest_bearing_debt_ratio", row.get("f_riba"))
        if pd.notna(int_debt_ratio):
            details["interest_bearing_debt_ok"] = (
                int_debt_ratio <= self.THRESHOLDS["interest_bearing_debt_ratio_max"]
            )
            details["interest_bearing_debt_value"] = float(int_debt_ratio)
        else:
            details["interest_bearing_debt_ok"] = False

        # Rule 4: ROA >= -10%
        roa = row.get("roa")
        if pd.isna(roa) and pd.notna(row.get("net_income")) and pd.notna(row.get("total_assets")):
            roa = row["net_income"] / (row["total_assets"] + 1e-5)
            
        if pd.notna(roa):
            details["profitability_ok"] = roa >= self.THRESHOLDS["roa_min"]
            details["roa_value"] = float(roa)
        else:
            details["profitability_ok"] = False

        # Rule 5: Equity Ratio >= -20%
        equity_ratio = row.get("equity_ratio")
        if pd.isna(equity_ratio) and pd.notna(row.get("total_equity")) and pd.notna(row.get("total_assets")):
            equity_ratio = row["total_equity"] / (row["total_assets"] + 1e-5)
            
        if pd.notna(equity_ratio):
            details["equity_ok"] = equity_ratio >= self.THRESHOLDS["equity_ratio_min"]
            details["equity_ratio_value"] = float(equity_ratio)
        else:
            details["equity_ok"] = False

        # Overall financial compliance: all rules must be met (treat NaN as fail)
        financial_rules = [
            details["debt_to_assets_ok"],
            details["interest_income_ok"],
            details["interest_bearing_debt_ok"],
            details["profitability_ok"],
            details["equity_ok"],
        ]

        # Replace NaN with False (fail if missing data)
        financial_rules = [
            rule if isinstance(rule, bool) else False for rule in financial_rules
        ]

        is_compliant = all(financial_rules)

        return is_compliant, details

    def classify(self, df_features: pd.DataFrame = None) -> pd.DataFrame:
        """
        Classify all companies as Shariah-compliant or non-compliant.

        Returns:
            DataFrame with 'symbol', 'sector', 'shariah_compliant', and rule details
        """
        if df_features is not None:
            self.df_features = df_features
        elif self.df_features is None:
            raise ValueError("No features loaded. Call load_features() first.")

        logger.info(f"[CLASSIFIER] Classifying {len(self.df_features)} companies...")

        results = []

        for idx, row in self.df_features.iterrows():
            symbol = row["symbol"]

            # Apply sector rule
            sector = self.get_sector(symbol)
            sector_compliant = self.apply_sector_rule(symbol)

            # Apply financial rules
            financial_compliant, financial_details = self.apply_financial_rules(row)

            # Overall compliance: both sector AND financial rules must be met
            overall_compliant = sector_compliant and financial_compliant

            result = {
                "symbol": symbol,
                "sector": sector,
                "sector_compliant": int(sector_compliant),
                "financial_compliant": int(financial_compliant),
                "shariah_compliant": int(overall_compliant),
            }

            # Add financial details
            result.update(financial_details)

            results.append(result)

        self.labels = pd.DataFrame(results)

        # Summary statistics
        compliant_count = self.labels["shariah_compliant"].sum()
        non_compliant_count = len(self.labels) - compliant_count
        compliant_pct = (compliant_count / len(self.labels)) * 100

        logger.info(f"[CLASSIFIER] Classification Results:")
        logger.info(f"  Shariah Compliant: {compliant_count} ({compliant_pct:.1f}%)")
        logger.info(
            f"  Non-Compliant: {non_compliant_count} ({100 - compliant_pct:.1f}%)"
        )

        return self.labels

    def save_labels(self, output_path: str) -> None:
        """Save classification labels to CSV."""
        if self.labels is None:
            raise ValueError("No classifications generated. Call classify() first.")
        logger.info(f"[CLASSIFIER] Saving {len(self.labels)} classifications...")
        self.labels.to_csv(output_path, index=False)
        logger.info(f"[CLASSIFIER] Saved to {output_path}")

    def get_classification_summary(self) -> Dict:
        """Get summary statistics for classifications."""
        if self.labels is None:
            raise ValueError("No classifications generated. Call classify() first.")

        summary = {
            "total_companies": len(self.labels),
            "shariah_compliant": int(self.labels["shariah_compliant"].sum()),
            "non_compliant": int(
                len(self.labels) - self.labels["shariah_compliant"].sum()
            ),
            "compliant_pct": (
                self.labels["shariah_compliant"].sum() / len(self.labels) * 100
            ),
            "by_sector": self.labels.groupby("sector")["shariah_compliant"]
            .agg(["sum", "count"])
            .to_dict(),
        }

        return summary


def classify_shariah_compliance(
    features_csv: str,
    sector_mapping_json: str,
    output_csv: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to classify Shariah compliance.

    Args:
        features_csv: Path to companies_with_features.csv
        sector_mapping_json: Path to sector_mapping.json
        output_csv: Where to save companies_with_labels.csv
        verbose: Print debug info

    Returns:
        DataFrame with classification labels
    """
    classifier = ShariaComplianceClassifier(
        sector_mapping_path=sector_mapping_json, verbose=verbose
    )
    classifier.load_features(features_csv)
    labels_df = classifier.classify()
    classifier.save_labels(output_csv)
    return labels_df


if __name__ == "__main__":
    import os

    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    features_csv = f"{project_root}/data/processed/companies_with_features.csv"
    sector_mapping_json = f"{project_root}/src/sector_mapping.json"
    output_csv = f"{project_root}/data/processed/companies_with_labels.csv"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Classify compliance
    df_labels = classify_shariah_compliance(
        features_csv, sector_mapping_json, output_csv
    )

    print("\n[TEST] Classification Complete!")
    print(f"[TEST] Output shape: {df_labels.shape}")
    print(f"[TEST] Saved to: {output_csv}")

    # Print summary
    classifier = ShariaComplianceClassifier(sector_mapping_path=sector_mapping_json)
    classifier.labels = df_labels
    summary = classifier.get_classification_summary()

    print(f"\n[TEST] Summary:")
    print(f"  Total companies: {summary['total_companies']}")
    print(
        f"  Compliant: {summary['shariah_compliant']} ({summary['compliant_pct']:.1f}%)"
    )
    print(f"  Non-Compliant: {summary['non_compliant']}")
