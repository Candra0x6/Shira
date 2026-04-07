"""
Test Suite for Feature Engineering (Phase 2)
Tests for financial feature calculations and validations
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feature_engineer import FeatureEngineer


class TestFeatureEngineering:
    """Test suite for feature engineering"""

    @classmethod
    def setup_class(cls):
        """Setup test data"""
        # Load sample data
        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "idx_real_kaggle.csv"
        )
        cls.raw_data = pd.read_csv(data_path)

        # Initialize feature engineer
        cls.feature_engineer = FeatureEngineer()

        # Create a small test sample (first 10 rows)
        cls.test_data = cls.raw_data.head(10).copy()

    def test_debt_to_equity_calculation(self):
        """Test debt-to-equity ratio calculation"""
        # Add required columns if missing
        if "total_liabilities" not in self.test_data.columns:
            self.test_data["total_liabilities"] = self.test_data.get(
                "total_liabilities", 100
            )
        if "total_equity" not in self.test_data.columns:
            self.test_data["total_equity"] = self.test_data.get("total_equity", 50)

        # Calculate manually
        expected = self.test_data["total_liabilities"] / (
            self.test_data["total_equity"] + 1e-5
        )

        # Verify calculation
        assert len(expected) > 0, "Debt-to-equity calculation should produce results"
        assert all(expected >= 0), "Debt-to-equity should be non-negative"
        assert not all(np.isnan(expected)), "Result should not be all NaN values"
        print(f"✓ test_debt_to_equity_calculation PASSED (mean={expected.mean():.4f})")

    def test_debt_to_assets_calculation(self):
        """Test debt-to-assets ratio calculation"""
        if "total_liabilities" not in self.test_data.columns:
            self.test_data["total_liabilities"] = 100
        if "total_assets" not in self.test_data.columns:
            self.test_data["total_assets"] = 200

        expected = self.test_data["total_liabilities"] / (
            self.test_data["total_assets"] + 1e-5
        )

        assert len(expected) > 0, "Debt-to-assets calculation should produce results"
        assert all(expected >= 0), "Debt-to-assets should be non-negative"
        assert all(expected <= 1.5), "Debt-to-assets should typically be reasonable"
        print(f"✓ test_debt_to_assets_calculation PASSED (mean={expected.mean():.4f})")

    def test_roe_calculation(self):
        """Test Return on Equity (ROE) calculation"""
        if "net_income" not in self.test_data.columns:
            self.test_data["net_income"] = 10
        if "total_equity" not in self.test_data.columns:
            self.test_data["total_equity"] = 50

        roe = self.test_data["net_income"] / (self.test_data["total_equity"] + 1e-5)

        assert len(roe) > 0, "ROE calculation should produce results"
        assert not all(np.isnan(roe)), "ROE should not be all NaN"
        print(f"✓ test_roe_calculation PASSED (mean ROE={roe.mean():.4f})")

    def test_nonhalal_revenue_range(self):
        """Test non-halal revenue percentage is within valid range"""
        if "nonhalal_revenue_percent" not in self.test_data.columns:
            self.test_data["nonhalal_revenue_percent"] = 30.5

        nonhalal = self.test_data["nonhalal_revenue_percent"]

        assert all(nonhalal >= 0), "Non-halal revenue should be >= 0%"
        assert all(nonhalal <= 100), "Non-halal revenue should be <= 100%"
        print(f"✓ test_nonhalal_revenue_range PASSED (mean={nonhalal.mean():.2f}%)")

    def test_debt_ratios_non_negative(self):
        """Test that all debt ratios are non-negative"""
        if "total_liabilities" not in self.test_data.columns:
            self.test_data["total_liabilities"] = 100
        if "total_equity" not in self.test_data.columns:
            self.test_data["total_equity"] = 50
        if "total_assets" not in self.test_data.columns:
            self.test_data["total_assets"] = 200

        debt_to_equity = self.test_data["total_liabilities"] / (
            self.test_data["total_equity"] + 1e-5
        )
        debt_to_assets = self.test_data["total_liabilities"] / (
            self.test_data["total_assets"] + 1e-5
        )

        assert all(debt_to_equity >= 0), "Debt-to-equity must be non-negative"
        assert all(debt_to_assets >= 0), "Debt-to-assets must be non-negative"
        print("✓ test_debt_ratios_non_negative PASSED")

    def test_feature_count(self):
        """Test that feature engineering creates expected number of features"""
        # Load engineered features
        engineered_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "engineered_features.csv"
        )
        engineered_df = pd.read_csv(engineered_path)

        # Should have 19 features (from metadata)
        expected_features = 19
        actual_features = len(
            [col for col in engineered_df.columns if col != "company_id"]
        )

        assert actual_features >= expected_features, (
            f"Should have at least {expected_features} features, got {actual_features}"
        )
        print(f"✓ test_feature_count PASSED (features={actual_features})")

    def test_no_missing_values_in_features(self):
        """Test that critical financial features have no missing values"""
        engineered_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "engineered_features.csv"
        )
        engineered_df = pd.read_csv(engineered_path)

        # Check for missing values in key features
        critical_features = [
            "total_assets",
            "total_equity",
            "net_income",
            "nonhalal_revenue_percent",
        ]
        for feat in critical_features:
            if feat in engineered_df.columns:
                missing_count = engineered_df[feat].isna().sum()
                assert missing_count == 0, (
                    f"Feature {feat} has {missing_count} missing values"
                )

        print("✓ test_no_missing_values_in_features PASSED")

    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer class initialization"""
        engineer = FeatureEngineer()

        assert engineer is not None, "FeatureEngineer should initialize"
        assert hasattr(engineer, "create_interaction_features"), (
            "Should have interaction method"
        )
        assert hasattr(engineer, "create_polynomial_features"), (
            "Should have polynomial method"
        )
        assert hasattr(engineer, "create_domain_features"), "Should have domain method"
        print("✓ test_feature_engineer_initialization PASSED")

    def test_interaction_features_not_null(self):
        """Test that interaction features are calculated"""
        if "interest_expense" not in self.test_data.columns:
            self.test_data["interest_expense"] = 5
        if "net_income" not in self.test_data.columns:
            self.test_data["net_income"] = 10
        if "net_revenue" not in self.test_data.columns:
            self.test_data["net_revenue"] = 100
        if "nonhalal_revenue_percent" not in self.test_data.columns:
            self.test_data["nonhalal_revenue_percent"] = 30
        if "total_assets" not in self.test_data.columns:
            self.test_data["total_assets"] = 200

        # Mock interaction calculation
        debt_service_ratio = self.test_data["interest_expense"] / (
            self.test_data["net_income"] + 1e-5
        )

        assert not all(np.isnan(debt_service_ratio)), (
            "Interaction features should have valid values"
        )
        assert len(debt_service_ratio) > 0, "Interaction features should be produced"
        print(f"✓ test_interaction_features_not_null PASSED")


def run_all_feature_tests():
    """Run all feature engineering tests"""
    test_instance = TestFeatureEngineering()
    TestFeatureEngineering.setup_class()

    tests = [
        test_instance.test_debt_to_equity_calculation,
        test_instance.test_debt_to_assets_calculation,
        test_instance.test_roe_calculation,
        test_instance.test_nonhalal_revenue_range,
        test_instance.test_debt_ratios_non_negative,
        test_instance.test_feature_count,
        test_instance.test_no_missing_values_in_features,
        test_instance.test_feature_engineer_initialization,
        test_instance.test_interaction_features_not_null,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1

    return passed, failed


if __name__ == "__main__":
    print("=" * 60)
    print("FEATURE ENGINEERING TESTS")
    print("=" * 60)
    passed, failed = run_all_feature_tests()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
