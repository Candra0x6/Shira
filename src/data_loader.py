"""
Data Loader & Transformation Module
Converts long-format financial data to wide format with multi-year aggregation.

Input:  combined_financial_data_idx.csv (89K rows, long format)
Output: companies_processed.csv (~605 rows, wide format, aggregated 2020-2023)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IDXFinancialDataLoader:
    """Load and transform IDX financial data from long to wide format."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.df_long = None
        self.df_wide = None
        self.df_aggregated = None
        self.years = [
            "2020",
            "2021",
            "2022",
            "2023",
        ]  # String keys to match CSV columns

    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """Load long-format CSV file."""
        logger.info(f"[LOAD] Loading CSV from {csv_path}")
        self.df_long = pd.read_csv(csv_path)
        logger.info(
            f"[LOAD] Loaded {len(self.df_long)} rows × {len(self.df_long.columns)} columns"
        )
        logger.info(f"[LOAD] Unique companies: {self.df_long['symbol'].nunique()}")
        return self.df_long

    def pivot_to_wide(self) -> pd.DataFrame:
        """Convert long format to wide format."""
        logger.info("[PIVOT] Converting to wide format...")

        # Pivot: Create separate column for each (account, type, year)
        # Format: account_type_year (e.g., Total_Assets_BS_2020)
        self.df_long["column_name"] = (
            self.df_long["account"].str.replace(" ", "_")
            + "_"
            + self.df_long["type"]
            + "_"
            + self.df_long["2020"].astype(str).str[:4]  # Will be fixed in next step
        )

        # Better pivot approach: melt years, then pivot by account_type_year
        df_melted = pd.melt(
            self.df_long,
            id_vars=["symbol", "account", "type"],
            value_vars=["2020", "2021", "2022", "2023"],  # String column names
            var_name="year",
            value_name="value",
        )

        # Create clean column name
        df_melted["column_name"] = (
            df_melted["account"]
            .str.replace(r"[^a-zA-Z0-9]", "_", regex=True)
            .str.strip("_")
            + "_"
            + df_melted["year"].astype(str)
        )

        # Pivot to wide
        df_wide = df_melted.pivot_table(
            index="symbol",
            columns="column_name",
            values="value",
            aggfunc="first",  # Use first non-null value
        )

        # Keep account type info for reference
        df_wide = df_wide.reset_index()

        logger.info(
            f"[PIVOT] Wide format: {len(df_wide)} companies × {len(df_wide.columns)} columns"
        )
        self.df_wide = df_wide
        return df_wide

    def aggregate_years(self, method: str = "mean") -> pd.DataFrame:
        """Aggregate 2020-2023 data into single row per company."""
        logger.info(f"[AGGREGATE] Aggregating across years using '{method}'...")

        # For each company, aggregate the 4 years into single values
        df_agg = self.df_long.copy()

        # Convert year columns to numeric
        for year in self.years:
            df_agg[year] = pd.to_numeric(df_agg[year], errors="coerce")

        # Calculate aggregated value across years for each account
        value_cols = self.years

        if method == "mean":
            df_agg["value_aggregated"] = df_agg[value_cols].mean(axis=1, skipna=True)
        elif method == "median":
            df_agg["value_aggregated"] = df_agg[value_cols].median(axis=1, skipna=True)
        elif method == "latest":
            df_agg["value_aggregated"] = df_agg[2023]
        else:
            df_agg["value_aggregated"] = df_agg[value_cols].mean(axis=1, skipna=True)

        # Pivot to wide format
        df_agg_wide = df_agg.pivot_table(
            index="symbol",
            columns="account",
            values="value_aggregated",
            aggfunc="first",
        )

        # Clean up column names
        df_agg_wide.columns = [
            col.replace(" ", "_").replace("(", "").replace(")", "")
            for col in df_agg_wide.columns
        ]
        df_agg_wide = df_agg_wide.reset_index()

        logger.info(
            f"[AGGREGATE] Aggregated: {len(df_agg_wide)} companies × {len(df_agg_wide.columns)} features"
        )
        self.df_aggregated = df_agg_wide
        return df_agg_wide

    def clean_data(
        self, df: pd.DataFrame, drop_nulls_threshold: float = 0.5
    ) -> pd.DataFrame:
        """Clean data: handle missing values, convert types."""
        logger.info(
            f"[CLEAN] Cleaning data (null threshold: {drop_nulls_threshold})..."
        )

        original_rows = len(df)

        # Drop rows with too many NaNs
        null_ratio = df.isnull().sum(axis=1) / len(df.columns)
        df = df[null_ratio < drop_nulls_threshold]

        logger.info(
            f"[CLEAN] Dropped {original_rows - len(df)} companies with >{drop_nulls_threshold:.0%} missing data"
        )

        # Fill forward across columns (companies with similar patterns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"[CLEAN] Converted {len(numeric_cols)} columns to numeric")
        logger.info(
            f"[CLEAN] Remaining: {len(df)} companies × {len(df.columns)} columns"
        )

        return df

    def validate_critical_accounts(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Validate that critical financial accounts exist."""
        logger.info("[VALIDATE] Checking for critical financial accounts...")

        critical_accounts = [
            "Total_Assets",
            "Total_Revenue",
            "Total_Debt",
            "Net_Income",
            "Total_Equity_Gross_Minority_Interest",
        ]

        found = {}
        missing = []

        for account in critical_accounts:
            # Look for account with flexible matching
            matches = [col for col in df.columns if account.lower() in col.lower()]
            if matches:
                found[account] = matches[0]
                logger.info(f"  ✓ {account} → {matches[0]}")
            else:
                missing.append(account)
                logger.warning(f"  ✗ {account} NOT FOUND")

        if missing:
            logger.warning(
                f"[VALIDATE] Missing {len(missing)} critical accounts: {missing}"
            )

        return df, found

    def save_processed(self, df: pd.DataFrame, output_path: str) -> None:
        """Save processed data to CSV."""
        logger.info(f"[SAVE] Saving processed data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info(f"[SAVE] Saved {len(df)} companies with {len(df.columns)} features")

    def process_full_pipeline(
        self,
        csv_path: str,
        output_path: str,
        aggregation_method: str = "mean",
        null_threshold: float = 0.5,
    ) -> pd.DataFrame:
        """Execute full data processing pipeline."""
        logger.info("=" * 80)
        logger.info("STARTING FULL DATA PROCESSING PIPELINE")
        logger.info("=" * 80)

        # Step 1: Load
        self.load_csv(csv_path)

        # Step 2: Aggregate years
        df_processed = self.aggregate_years(method=aggregation_method)

        # Step 3: Clean
        df_processed = self.clean_data(
            df_processed, drop_nulls_threshold=null_threshold
        )

        # Step 4: Validate
        df_processed, critical_accounts = self.validate_critical_accounts(df_processed)

        # Step 5: Save
        self.save_processed(df_processed, output_path)

        logger.info("=" * 80)
        logger.info(
            f"PIPELINE COMPLETE: {len(df_processed)} companies ready for feature engineering"
        )
        logger.info("=" * 80)

        return df_processed


def load_and_transform_idx_data(
    csv_path: str,
    output_path: str,
    aggregation_method: str = "mean",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to load and transform IDX financial data.

    Args:
        csv_path: Path to combined_financial_data_idx.csv
        output_path: Where to save processed data
        aggregation_method: 'mean', 'median', or 'latest'
        verbose: Print debug info

    Returns:
        Processed DataFrame with ~605 companies × ~250 financial accounts
    """
    loader = IDXFinancialDataLoader(verbose=verbose)
    return loader.process_full_pipeline(
        csv_path=csv_path,
        output_path=output_path,
        aggregation_method=aggregation_method,
    )


if __name__ == "__main__":
    # Test the loader
    import sys
    import os

    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    csv_path = f"{project_root}/data/raw/combined_financial_data_idx.csv"
    output_path = f"{project_root}/data/processed/companies_processed.csv"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process data
    df = load_and_transform_idx_data(csv_path, output_path)

    print("\n[TEST] Processing complete!")
    print(f"[TEST] Output shape: {df.shape}")
    print(f"[TEST] Saved to: {output_path}")
