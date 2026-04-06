"""
Phase 2: Advanced Feature Engineering
Creates 20+ new features to increase model complexity and capture non-linear relationships

Feature Categories:
1. Interaction Features: Combine ratios to capture relationships (8 features)
2. Polynomial Features: Square terms for non-linear effects (5 features)
3. Domain-Specific Features: Shariah-focused risk indicators (5 features)
4. Sector Risk Features: Sector-level aggregate statistics (3 features)
5. Statistical Features: Aggregate statistics and distributions (5 features)

Expected Impact: 93% → 95% accuracy (+2%)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder


class FeatureEngineer:
    """
    Advanced feature engineering for Shariah compliance prediction

    Transforms 19 base features into 35-40 enhanced features through:
    - Interaction terms
    - Polynomial expansion
    - Domain-specific indicators
    - Risk aggregations
    - Statistical measures
    """

    def __init__(self):
        """Initialize feature engineer"""
        self.base_features = None
        self.engineered_features = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_stats = {}

    def create_interaction_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Create interaction features combining financial metrics

        Rationale:
        - Risk metrics interact (e.g., high debt + high interest = severe riba risk)
        - Profitability interacts with leverage (debt service capacity)
        - Non-halal income interacts with profitability (concentration risk)

        Returns: Dict with 8 interaction features
        """
        interactions = {}

        # 1. Debt servicing capacity: Can company cover riba with profits?
        interactions["debt_service_ratio"] = df["interest_expense"] / (
            df["net_income"] + 1e-5
        )

        # 2. Leverage-profitability interaction
        interactions["leverage_profit_interaction"] = (
            df["debt_to_equity"] * df["profit_margin"]
        )

        # 3. Asset quality: Non-halal revenue as % of assets
        interactions["nonhalal_asset_ratio"] = (
            df["net_revenue"] * df["nonhalal_revenue_percent"] / 100
        ) / (df["total_assets"] + 1e-5)

        # 4. Cash flow health vs debt burden
        interactions["cashflow_debt_coverage"] = df["operating_cash_flow"] / (
            df["total_liabilities"] + 1e-5
        )

        # 5. ROA vs riba intensity (profitability despite interest costs)
        interactions["roa_riba_efficiency"] = df["roa"] / (df["riba_intensity"] + 1e-5)

        # 6. Equity quality: Net income to equity (true returns to shareholders)
        interactions["equity_quality"] = df["net_income"] / (df["total_equity"] + 1e-5)

        # 7. Revenue composition: Halal vs total revenue strength
        interactions["halal_revenue_strength"] = (
            df["net_revenue"]
            * (1 - df["nonhalal_revenue_percent"] / 100)
            / (df["total_assets"] + 1e-5)
        )

        # 8. Financial stability: Combined health score
        interactions["financial_stability"] = (df["cash_flow_to_debt"] * df["roa"]) / (
            df["debt_to_assets"] + 1e-5
        )

        return interactions

    def create_polynomial_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Create polynomial (squared) features for non-linear relationships

        Rationale:
        - Risk metrics show non-linear effects (e.g., debt_to_equity squared)
        - Threshold effects: Problems accelerate at high levels
        - Diminishing returns for positive metrics

        Returns: Dict with 5 polynomial features
        """
        polys = {}

        # 1. Squared debt ratio (exponential risk increase)
        polys["debt_to_assets_squared"] = df["debt_to_assets"] ** 2

        # 2. Squared non-halal percentage (concentration risk)
        polys["nonhalal_percent_squared"] = (df["nonhalal_revenue_percent"] / 100) ** 2

        # 3. Squared profit margin (efficiency amplification)
        polys["profit_margin_squared"] = df["profit_margin"] ** 2

        # 4. Squared interest coverage (survival metric)
        polys["interest_coverage_squared"] = np.maximum(0, df["interest_coverage"]) ** 2

        # 5. Squared ROA (profitability amplification)
        polys["roa_squared"] = df["roa"] ** 2

        return polys

    def create_domain_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Create Shariah-specific domain features

        Rationale:
        - Riba intensity shows Islamic finance risk specifically
        - Non-halal concentration threshold effects
        - Multiple risk factor combinations

        Returns: Dict with 5 domain-specific features
        """
        domain = {}

        # 1. Total Shariah risk score (weighted combination)
        domain["shariah_risk_score"] = (
            0.4 * (df["debt_to_assets"] / df["debt_to_assets"].max())
            + 0.4 * (df["nonhalal_revenue_percent"] / 100)
            + 0.2 * (df["riba_intensity"] / (df["riba_intensity"].max() + 1e-5))
        )

        # 2. Riba exposure level (debt-based financing risk)
        domain["riba_exposure_level"] = df["interest_expense"] / (
            df["total_assets"] + 1e-5
        )

        # 3. Non-halal revenue concentration (business model risk)
        domain["nonhalal_concentration"] = (
            df["nonhalal_revenue_percent"]
            / 100
            * (1 - (df["net_income"] / (df["net_revenue"] + 1e-5)))
        )

        # 4. Compliance capacity (ability to change business model)
        domain["compliance_capacity"] = (1 - df["nonhalal_revenue_percent"] / 100) * df[
            "roa"
        ]

        # 5. Mixed-use asset risk (problematic for Shariah)
        domain["mixed_asset_risk"] = (
            df["debt_to_assets"] * df["nonhalal_revenue_percent"] / 100
        )

        return domain

    def create_sector_risk_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Create sector-level aggregate risk features

        Rationale:
        - Some sectors inherently riskier (alcohol, gaming, etc.)
        - Sector competition affects pricing power and margins
        - Sector debt patterns affect financial health

        Returns: Dict with 3 sector-level features
        """
        sector_features = {}

        # 1. Sector debt concentration
        sector_debt = df.groupby("sector")["debt_to_assets"].transform("mean")
        sector_features["sector_avg_debt"] = sector_debt

        # 2. Sector profitability comparison
        sector_profit = df.groupby("sector")["profit_margin"].transform("mean")
        sector_features["sector_avg_profit"] = sector_profit

        # 3. Relative debt positioning (company vs sector average)
        sector_features["debt_vs_sector"] = df["debt_to_assets"] - sector_debt

        return sector_features

    def create_statistical_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Create statistical aggregate features

        Rationale:
        - Outlier detection (unusual financial profiles)
        - Distribution-based risk scoring
        - Comparative metrics

        Returns: Dict with 5 statistical features
        """
        statistical = {}

        # 1. Z-score for debt ratio (outlier detection)
        debt_mean = df["debt_to_assets"].mean()
        debt_std = df["debt_to_assets"].std()
        statistical["debt_zscore"] = (df["debt_to_assets"] - debt_mean) / (
            debt_std + 1e-5
        )

        # 2. Z-score for non-halal percentage
        nonhalal_mean = df["nonhalal_revenue_percent"].mean()
        nonhalal_std = df["nonhalal_revenue_percent"].std()
        statistical["nonhalal_zscore"] = (
            df["nonhalal_revenue_percent"] - nonhalal_mean
        ) / (nonhalal_std + 1e-5)

        # 3. Z-score for profitability
        profit_mean = df["profit_margin"].mean()
        profit_std = df["profit_margin"].std()
        statistical["profit_zscore"] = (df["profit_margin"] - profit_mean) / (
            profit_std + 1e-5
        )

        # 4. Risk quantile (percentile-based ranking)
        statistical["risk_quantile"] = (
            df["shariah_risk_score"].rank(pct=True)
            if "shariah_risk_score" in df.columns
            else (
                0.4 * (df["debt_to_assets"].rank(pct=True))
                + 0.4 * (df["nonhalal_revenue_percent"].rank(pct=True))
                + 0.2 * (df["riba_intensity"].rank(pct=True))
            )
        )

        # 5. Financial health index (inverse of risk)
        statistical["financial_health_index"] = (
            (1 / (df["debt_to_assets"] + 1))
            * (1 - df["nonhalal_revenue_percent"] / 100)
            * (1 + np.maximum(0, df["roa"]))
        )

        return statistical

    def engineer_features(
        self, df: pd.DataFrame, base_features_list: List[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply all feature engineering steps

        Args:
            df: Input DataFrame with base features
            base_features_list: List of base feature names to preserve

        Returns:
            Tuple of (engineered_df, feature_names)
        """
        print("\n[PHASE 2] Feature Engineering")
        df_engineered = df.copy()

        # Ensure base financial features exist
        base_features = [
            "total_assets",
            "total_liabilities",
            "total_equity",
            "net_revenue",
            "nonhalal_revenue_percent",
            "net_income",
            "operating_cash_flow",
            "interest_expense",
            "sector",
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
        ]

        # Create base ratio features if not present
        print("  Creating base financial ratios...")
        if "debt_to_equity" not in df_engineered.columns:
            df_engineered["debt_to_equity"] = df_engineered["total_liabilities"] / (
                df_engineered["total_equity"] + 1e-5
            )
        if "debt_to_assets" not in df_engineered.columns:
            df_engineered["debt_to_assets"] = (
                df_engineered["total_liabilities"] / df_engineered["total_assets"]
            )
        if "roe" not in df_engineered.columns:
            df_engineered["roe"] = df_engineered["net_income"] / (
                df_engineered["total_equity"] + 1e-5
            )
        if "roa" not in df_engineered.columns:
            df_engineered["roa"] = (
                df_engineered["net_income"] / df_engineered["total_assets"]
            )
        if "profit_margin" not in df_engineered.columns:
            df_engineered["profit_margin"] = df_engineered["net_income"] / (
                df_engineered["net_revenue"] + 1e-5
            )
        if "interest_coverage" not in df_engineered.columns:
            df_engineered["interest_coverage"] = df_engineered["net_income"] / (
                df_engineered["interest_expense"] + 1e-5
            )
        if "cash_flow_to_debt" not in df_engineered.columns:
            df_engineered["cash_flow_to_debt"] = df_engineered[
                "operating_cash_flow"
            ] / (df_engineered["total_liabilities"] + 1e-5)
        if "f_riba" not in df_engineered.columns:
            df_engineered["f_riba"] = df_engineered["debt_to_assets"]
        if "f_nonhalal" not in df_engineered.columns:
            df_engineered["f_nonhalal"] = df_engineered["nonhalal_revenue_percent"]
        if "riba_intensity" not in df_engineered.columns:
            df_engineered["riba_intensity"] = df_engineered["interest_expense"] / (
                df_engineered["net_revenue"] + 1e-5
            )

        # 1. Interaction Features
        print("  Creating interaction features (8 features)...")
        interactions = self.create_interaction_features(df_engineered)
        for name, values in interactions.items():
            df_engineered[name] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        # 2. Polynomial Features
        print("  Creating polynomial features (5 features)...")
        polys = self.create_polynomial_features(df_engineered)
        for name, values in polys.items():
            df_engineered[name] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        # 3. Domain-Specific Features
        print("  Creating domain-specific features (5 features)...")
        domain = self.create_domain_features(df_engineered)
        for name, values in domain.items():
            df_engineered[name] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. Sector Risk Features
        print("  Creating sector risk features (3 features)...")
        sector_features = self.create_sector_risk_features(df_engineered)
        for name, values in sector_features.items():
            df_engineered[name] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        # 5. Statistical Features
        print("  Creating statistical features (5 features)...")
        statistical = self.create_statistical_features(df_engineered)
        for name, values in statistical.items():
            df_engineered[name] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        # Compile feature list
        all_features = (
            base_features
            + list(interactions.keys())
            + list(polys.keys())
            + list(domain.keys())
            + list(sector_features.keys())
            + list(statistical.keys())
        )

        # Keep only existing features
        existing_features = [f for f in all_features if f in df_engineered.columns]

        print(f"  ✓ Base features: {len(base_features)}")
        print(f"  ✓ Interaction features: {len(interactions)}")
        print(f"  ✓ Polynomial features: {len(polys)}")
        print(f"  ✓ Domain features: {len(domain)}")
        print(f"  ✓ Sector features: {len(sector_features)}")
        print(f"  ✓ Statistical features: {len(statistical)}")
        print(f"  Total engineered features: {len(existing_features)}")

        self.engineered_features = existing_features

        return df_engineered, existing_features

    def get_feature_summary(self) -> Dict:
        """Get summary of engineered features"""
        if not self.engineered_features:
            return {}

        return {
            "total_features": len(self.engineered_features),
            "features": self.engineered_features,
            "categories": {
                "base": 19,
                "interaction": 8,
                "polynomial": 5,
                "domain": 5,
                "sector": 3,
                "statistical": 5,
            },
        }
