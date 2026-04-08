"""
Shariah Financial Features Engineering Module
Engineer 12 Shariah-compliant financial ratios from raw financial statements.

OJK/DSN-MUI Guidelines:
1. Debt-to-Assets Ratio: Total debt / Total assets < 30%
2. Interest-Bearing Debt Ratio: (Interest-bearing debt / Total debt) < 30%
3. Interest Income Ratio: (Interest income / Total revenue) < 5%
4. Current Ratio: Current assets / Current liabilities > 1.0 (liquidity)
5. Return on Assets (ROA): Net income / Total assets (profitability)
6. Return on Equity (ROE): Net income / Total equity (shareholder returns)
7. Operating Cash Flow Ratio: Operating CF / Current liabilities (CF quality)
8. Asset Turnover: Total revenue / Total assets (efficiency)
9. Gross Margin: (Revenue - COGS) / Revenue (profitability)
10. Working Capital Ratio: (CA - CL) / Current assets (liquidity management)
11. Net Profit Margin: Net income / Total revenue (profitability)
12. Equity Ratio: Total equity / Total assets (leverage)
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class ShariaFinancialFeatures:
    """Engineer Shariah-compliant financial ratios from raw financial data."""

    # Define account name mappings (flexible matching)
    ACCOUNT_MAPPING = {
        "total_assets": ["Total_Assets", "Total Assets"],
        "total_liabilities": [
            "Total_Liabilities_Net_Minority_Interest",
            "Total_Liabilities",
            "Total_Debt",
            "Total Liabilities",
        ],
        "total_equity": [
            "Total_Equity_Gross_Minority_Interest",
            "Total_Equity",
            "Stockholders_Equity",
            "Common_Stock_Equity",
        ],
        "net_revenue": ["Total_Revenue", "Operating_Revenue", "Net_Revenue"],
        "nonhalal_revenue_percent": ["Nonhalal_Revenue_Percent", "nonhalal_revenue_percent"],
        "net_income": [
            "Net_Income",
            "Net_Income_Common_Stockholders",
            "Net_Income_Continuous_Operations",
        ],
        "operating_cash_flow": [
            "Operating_Cash_Flow",
            "Cash_Flowsfromusedin_Operating_Activities_Direct",
        ],
        "interest_expense": [
            "Interest_Expense",
            "Interest_Expense_Non_Operating",
            "Interest_Expense_Total",
        ],
    }

    # Fixed sector mapping to ensure encoding consistency with training
    SECTOR_MAPPING = {
        "Agriculture": 0,
        "Banking": 1,
        "Chemicals": 2,
        "Construction": 3,
        "Food & Beverage": 4,
        "Infrastructure": 5,
        "Insurance": 6,
        "Manufacturing": 7,
        "Manufacturing, Electronics": 8,
        "Manufacturing, Food & Beverage": 9,
        "Media": 10,
        "Mining": 11,
        "Oil & Gas": 12,
        "Other": 13,
        "Pharmaceuticals": 14,
        "Real Estate": 15,
        "Retail": 16,
        "Technology": 17,
        "Telecommunications": 18,
        "Tobacco": 19,
        "Utilities": 20,
    }

    # Canonical feature order for model input
    FEATURE_ORDER = [
        "total_assets",
        "total_liabilities",
        "total_equity",
        "net_revenue",
        "nonhalal_revenue_percent",
        "net_income",
        "operating_cash_flow",
        "interest_expense",
        "debt_to_equity",
        "debt_to_assets",
        "roe",
        "roa",
        "profit_margin",
        "interest_coverage",
        "cash_flow_to_debt",
        "f_riba",
        "f_nonhalal",
        "riba_intensity",
        "sector_encoded",
    ]

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.df = None
        self.features = None
        self.account_cols = {}  # Maps account type to actual column name

    def load_processed_data(self, csv_path: str) -> pd.DataFrame:
        """Load processed financial data."""
        if self.verbose:
            logger.info(f"[FEATURES] Loading processed data from {csv_path}")
        self.df = pd.read_csv(csv_path)
        if self.verbose:
            logger.info(
                f"[FEATURES] Loaded {len(self.df)} companies × {len(self.df.columns)} accounts"
            )
        return self.df

    def find_account_column(self, df: pd.DataFrame, account_type: str) -> Optional[str]:
        """Find best matching column for a financial account."""
        possible_cols = self.ACCOUNT_MAPPING.get(account_type, [])
        for col in possible_cols:
            if col in df.columns:
                return col
        # Fallback: flexible case-insensitive search
        account_lower = account_type.lower()
        for col in df.columns:
            if col.lower() == account_lower:
                return col
        return None

    def _safe_divide(
        self, numerator: pd.Series, denominator: pd.Series, default: float = 0.0
    ) -> pd.Series:
        """Safely divide two series, handling zeros and NaNs."""
        result = pd.Series(default, index=numerator.index)
        valid_mask = (denominator != 0) & denominator.notna() & numerator.notna()
        result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        return result

    def engineer_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Engineer all 19 features required by the production model."""
        if df is not None:
            self.df = df
        elif self.df is None:
            raise ValueError("No data loaded. Call load_processed_data() first.")

        if self.verbose:
            logger.info("[FEATURES] Engineering 19 features for production model...")

        # Initialize result with symbol and mandatory 19 features
        result_df = self.df[["symbol"]].copy()

        # 1. Base Financials (8 features)
        for acc in self.FEATURE_ORDER[:8]:
            col = self.find_account_column(self.df, acc)
            if col and col in self.df.columns:
                result_df[acc] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)
            else:
                result_df[acc] = 0.0

        # Fix specific missing indicators if they're in raw but not numeric
        if result_df["total_liabilities"].sum() == 0:
            logger.warning("[FEATURES] Total Liabilities is 0 - checking fallback columns...")
            # Some datasets use Total_Debt as proxy
            debt_col = self.find_account_column(self.df, "total_debt")
            if debt_col and debt_col in self.df.columns:
                result_df["total_liabilities"] = pd.to_numeric(self.df[debt_col], errors="coerce").fillna(0)

        # 2. Financial Ratios (7 features)
        result_df["debt_to_equity"] = self._safe_divide(
            result_df["total_liabilities"], result_df["total_equity"] + 1e-5
        )
        result_df["debt_to_assets"] = self._safe_divide(
            result_df["total_liabilities"], result_df["total_assets"] + 1e-5
        )
        result_df["roe"] = self._safe_divide(
            result_df["net_income"], result_df["total_equity"] + 1e-5
        )
        result_df["roa"] = self._safe_divide(
            result_df["net_income"], result_df["total_assets"] + 1e-5
        )
        result_df["profit_margin"] = self._safe_divide(
            result_df["net_income"], result_df["net_revenue"] + 1e-5
        )
        result_df["interest_coverage"] = self._safe_divide(
            result_df["net_income"], result_df["interest_expense"] + 1e-5
        )
        result_df["cash_flow_to_debt"] = self._safe_divide(
            result_df["operating_cash_flow"], result_df["total_liabilities"] + 1e-5
        )

        # 3. Shariah-Specific Indicators (3 features)
        result_df["f_riba"] = result_df["debt_to_assets"]
        result_df["f_nonhalal"] = result_df["nonhalal_revenue_percent"]
        result_df["riba_intensity"] = self._safe_divide(
            result_df["interest_expense"], result_df["net_revenue"] + 1e-5
        )

        # 4. Sector Encoding (1 feature)
        if "sector" in self.df.columns:
            result_df["sector"] = self.df["sector"]
        else:
            result_df["sector"] = "Other"

        result_df["sector_encoded"] = result_df["sector"].map(self.SECTOR_MAPPING).fillna(self.SECTOR_MAPPING["Other"])

        # Final order verification [symbol, sector] + 19 features
        cols = ["symbol", "sector"] + self.FEATURE_ORDER
        result_df = result_df[cols]

        if self.verbose:
            logger.info(
                f"[FEATURES] Engineered 19 features for {len(result_df)} companies"
            )
            missing_pct = result_df.iloc[:, 1:].isnull().sum().mean() * 100
            logger.info(f"[FEATURES] Average missing data: {missing_pct:.1f}%")

        self.features = result_df
        return result_df

    def save_features(self, output_path: str) -> None:
        """Save engineered features to CSV."""
        if self.features is None:
            raise ValueError("No features engineered. Call engineer_features() first.")
        logger.info(f"[FEATURES] Saving {len(self.features)} companies with 19 features")
        self.features.to_csv(output_path, index=False)
        logger.info(f"[FEATURES] Saved to {output_path}")

    def get_feature_summary(self) -> pd.DataFrame:
        """Get summary statistics for all engineered features."""
        if self.features is None:
            raise ValueError("No features engineered. Call engineer_features() first.")

        summary = self.features.iloc[:, 1:].describe().T
        return summary


def engineer_shariah_features(
    input_csv: str, output_csv: str, verbose: bool = True
) -> pd.DataFrame:
    """
    Convenience function to engineer Shariah financial features.

    Args:
        input_csv: Path to companies_processed.csv
        output_csv: Where to save companies_with_features.csv
        verbose: Print debug info

    Returns:
        DataFrame with 12 engineered financial ratios
    """
    engineer = ShariaFinancialFeatures(verbose=verbose)
    engineer.load_processed_data(input_csv)
    features_df = engineer.engineer_features()
    engineer.save_features(output_csv)
    return features_df


if __name__ == "__main__":
    import os

    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_csv = f"{project_root}/data/processed/companies_processed.csv"
    output_csv = f"{project_root}/data/processed/companies_with_features.csv"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Engineer features
    df_features = engineer_shariah_features(input_csv, output_csv)

    print("\n[TEST] Feature Engineering Complete!")
    print(f"[TEST] Output shape: {df_features.shape}")
    print(f"[TEST] Saved to: {output_csv}")
    print("\n[TEST] Feature Summary:")
    engineer = ShariaFinancialFeatures()
    engineer.features = df_features
    print(engineer.get_feature_summary())
