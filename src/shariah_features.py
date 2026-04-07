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
        "total_debt": ["Total_Debt", "Total Debt"],
        "interest_bearing_debt": [
            "Interest_Payable",
            "Current_Debt",
            "Long_Term_Debt",
        ],  # Sum of interest-bearing obligations
        "current_assets": ["Current_Assets", "Current Assets"],
        "current_liabilities": ["Current_Liabilities", "Current Liabilities"],
        "total_equity": [
            "Total_Equity_Gross_Minority_Interest",
            "Stockholders_Equity",
            "Common_Stock_Equity",
        ],
        "net_income": [
            "Net_Income",
            "Net_Income_Common_Stockholders",
            "Net_Income_Continuous_Operations",
        ],
        "total_revenue": ["Total_Revenue", "Operating_Revenue", "Total_Revenue"],
        "cost_of_revenue": [
            "Cost_Of_Revenue",
            "Reconciled_Cost_Of_Revenue",
            "Cost_Of_Revenue",
        ],
        "gross_profit": ["Gross_Profit"],
        "operating_cash_flow": [
            "Operating_Cash_Flow",
            "Cash_Flowsfromusedin_Operating_Activities_Direct",
        ],
        "interest_income": [
            "Interest_Income",
            "Interest_Income_Non_Operating",
            "Net_Interest_Income",
        ],
    }

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
        self, numerator: pd.Series, denominator: pd.Series, default: float = np.nan
    ) -> pd.Series:
        """Safely divide two series, handling zeros and NaNs."""
        result = pd.Series(default, index=numerator.index)
        valid_mask = (denominator != 0) & denominator.notna() & numerator.notna()
        result[valid_mask] = numerator[valid_mask] / denominator[valid_mask]
        return result

    def engineer_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Engineer all 12 Shariah financial ratios."""
        if df is not None:
            self.df = df
        elif self.df is None:
            raise ValueError("No data loaded. Call load_processed_data() first.")

        if self.verbose:
            logger.info("[FEATURES] Engineering 12 Shariah financial ratios...")

        result_df = self.df[["symbol"]].copy()

        # Find account columns
        accounts = {
            "total_assets": "Total_Assets",
            "total_debt": "Total_Debt",
            "current_assets": "Current_Assets",
            "current_liabilities": "Current_Liabilities",
            "total_equity": "Total_Equity_Gross_Minority_Interest",
            "net_income": "Net_Income",
            "total_revenue": "Total_Revenue",
            "cost_of_revenue": "Cost_Of_Revenue",
            "gross_profit": "Gross_Profit",
            "operating_cash_flow": "Operating_Cash_Flow",
            "interest_income": "Interest_Income",
        }

        # Try to find each account
        for account_type, default_col in accounts.items():
            col = self.find_account_column(self.df, account_type)
            if col is None:
                col = default_col  # Use default if found
                if col not in self.df.columns:
                    if self.verbose:
                        logger.warning(
                            f"[FEATURES] Could not find account: {account_type}"
                        )
                    col = None
            self.account_cols[account_type] = col

        # 1. Debt-to-Assets Ratio: Total Debt / Total Assets
        if (
            self.account_cols["total_debt"]
            and self.account_cols["total_assets"] in self.df.columns
        ):
            result_df["debt_to_assets"] = self._safe_divide(
                self.df[self.account_cols["total_debt"]],
                self.df[self.account_cols["total_assets"]],
            )
        else:
            result_df["debt_to_assets"] = np.nan

        # 2. Interest-Bearing Debt Ratio: (Interest-bearing debt / Total debt)
        # Estimate: Current_Debt + Long_Term_Debt (interest-bearing)
        interest_bearing = self.df.get(
            "Current_Debt", pd.Series(np.nan, index=self.df.index)
        ) + self.df.get("Long_Term_Debt", pd.Series(np.nan, index=self.df.index))
        if self.account_cols["total_debt"] in self.df.columns:
            result_df["interest_bearing_debt_ratio"] = self._safe_divide(
                interest_bearing, self.df[self.account_cols["total_debt"]]
            )
        else:
            result_df["interest_bearing_debt_ratio"] = np.nan

        # 3. Interest Income Ratio: Interest Income / Total Revenue
        if (
            self.account_cols["interest_income"]
            and self.account_cols["total_revenue"] in self.df.columns
        ):
            result_df["interest_income_ratio"] = self._safe_divide(
                self.df[self.account_cols["interest_income"]],
                self.df[self.account_cols["total_revenue"]],
            )
        else:
            result_df["interest_income_ratio"] = np.nan

        # 4. Current Ratio: Current Assets / Current Liabilities
        if (
            self.account_cols["current_assets"] in self.df.columns
            and self.account_cols["current_liabilities"] in self.df.columns
        ):
            result_df["current_ratio"] = self._safe_divide(
                self.df[self.account_cols["current_assets"]],
                self.df[self.account_cols["current_liabilities"]],
            )
        else:
            result_df["current_ratio"] = np.nan

        # 5. Return on Assets (ROA): Net Income / Total Assets
        if (
            self.account_cols["net_income"] in self.df.columns
            and self.account_cols["total_assets"] in self.df.columns
        ):
            result_df["roa"] = self._safe_divide(
                self.df[self.account_cols["net_income"]],
                self.df[self.account_cols["total_assets"]],
            )
        else:
            result_df["roa"] = np.nan

        # 6. Return on Equity (ROE): Net Income / Total Equity
        if (
            self.account_cols["net_income"] in self.df.columns
            and self.account_cols["total_equity"] in self.df.columns
        ):
            result_df["roe"] = self._safe_divide(
                self.df[self.account_cols["net_income"]],
                self.df[self.account_cols["total_equity"]],
            )
        else:
            result_df["roe"] = np.nan

        # 7. Operating Cash Flow Ratio: Operating CF / Current Liabilities
        if (
            self.account_cols["operating_cash_flow"] in self.df.columns
            and self.account_cols["current_liabilities"] in self.df.columns
        ):
            result_df["ocf_ratio"] = self._safe_divide(
                self.df[self.account_cols["operating_cash_flow"]],
                self.df[self.account_cols["current_liabilities"]],
            )
        else:
            result_df["ocf_ratio"] = np.nan

        # 8. Asset Turnover: Total Revenue / Total Assets
        if (
            self.account_cols["total_revenue"] in self.df.columns
            and self.account_cols["total_assets"] in self.df.columns
        ):
            result_df["asset_turnover"] = self._safe_divide(
                self.df[self.account_cols["total_revenue"]],
                self.df[self.account_cols["total_assets"]],
            )
        else:
            result_df["asset_turnover"] = np.nan

        # 9. Gross Margin: (Revenue - COGS) / Revenue
        if (
            self.account_cols["total_revenue"] in self.df.columns
            and self.account_cols["cost_of_revenue"] in self.df.columns
        ):
            gross_profit = (
                self.df[self.account_cols["total_revenue"]]
                - self.df[self.account_cols["cost_of_revenue"]]
            )
            result_df["gross_margin"] = self._safe_divide(
                gross_profit, self.df[self.account_cols["total_revenue"]]
            )
        else:
            result_df["gross_margin"] = np.nan

        # 10. Working Capital Ratio: (CA - CL) / Current Assets
        if (
            self.account_cols["current_assets"] in self.df.columns
            and self.account_cols["current_liabilities"] in self.df.columns
        ):
            working_capital = (
                self.df[self.account_cols["current_assets"]]
                - self.df[self.account_cols["current_liabilities"]]
            )
            result_df["working_capital_ratio"] = self._safe_divide(
                working_capital, self.df[self.account_cols["current_assets"]]
            )
        else:
            result_df["working_capital_ratio"] = np.nan

        # 11. Net Profit Margin: Net Income / Total Revenue
        if (
            self.account_cols["net_income"] in self.df.columns
            and self.account_cols["total_revenue"] in self.df.columns
        ):
            result_df["net_profit_margin"] = self._safe_divide(
                self.df[self.account_cols["net_income"]],
                self.df[self.account_cols["total_revenue"]],
            )
        else:
            result_df["net_profit_margin"] = np.nan

        # 12. Equity Ratio: Total Equity / Total Assets
        if (
            self.account_cols["total_equity"] in self.df.columns
            and self.account_cols["total_assets"] in self.df.columns
        ):
            result_df["equity_ratio"] = self._safe_divide(
                self.df[self.account_cols["total_equity"]],
                self.df[self.account_cols["total_assets"]],
            )
        else:
            result_df["equity_ratio"] = np.nan

        if self.verbose:
            logger.info(
                f"[FEATURES] Engineered 12 ratios for {len(result_df)} companies"
            )
            missing_pct = result_df.iloc[:, 1:].isnull().sum().mean() * 100
            logger.info(f"[FEATURES] Average missing data: {missing_pct:.1f}%")

        self.features = result_df
        return result_df

    def save_features(self, output_path: str) -> None:
        """Save engineered features to CSV."""
        if self.features is None:
            raise ValueError("No features engineered. Call engineer_features() first.")
        logger.info(f"[FEATURES] Saving {len(self.features)} companies with 12 ratios")
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
