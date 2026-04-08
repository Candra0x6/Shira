"""
Integration Tests for the Complete ML Pipeline
Tests for end-to-end workflow, data processing, and deployment readiness
"""

import sys
import os
import pickle
import json
import pandas as pd
import numpy as np
import tempfile
import csv

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestIntegrationPipeline:
    """Integration tests for complete ML pipeline"""

    @classmethod
    def setup_class(cls):
        """Setup for integration tests"""
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")

        # Load all components
        model_path = os.path.join(model_dir, "final_trained_model.pkl")
        with open(model_path, "rb") as f:
            cls.model = pickle.load(f)

        scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
        with open(scaler_path, "rb") as f:
            cls.scaler = pickle.load(f)

        metadata_path = os.path.join(model_dir, "model_metadata.json")
        with open(metadata_path, "r") as f:
            cls.metadata = json.load(f)

        # Load data
        engineered_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "processed", "companies_with_features.csv"
        )
        cls.features_df = pd.read_csv(engineered_path)
        # Use only feature columns
        cls.X = cls.features_df.drop(["symbol", "sector", "company_id"], axis=1, errors="ignore")

        # Load raw data for context
        raw_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "raw", "combined_financial_data_idx.csv"
        )
        cls.raw_data = pd.read_csv(raw_path)

    def test_full_prediction_pipeline(self):
        """Test complete prediction pipeline end-to-end"""
        # Step 1: Load features
        assert self.X is not None, "Features should load"
        expected_samples = len(self.X)

        # Step 2: Make predictions
        predictions = self.model.predict(self.X)
        assert len(predictions) == expected_samples, f"Should have {expected_samples} predictions"

        # Step 3: Get probabilities
        probs = self.model.predict_proba(self.X)
        assert probs.shape == (expected_samples, 2), f"Should have ({expected_samples}, 2) probability shape"

        # Step 4: Verify outputs
        assert all(p in [0, 1] for p in predictions), "Predictions should be binary"
        assert np.all(probs >= 0) and np.all(probs <= 1), (
            "Probabilities should be in [0, 1]"
        )

        print("✓ test_full_prediction_pipeline PASSED")

    def test_prediction_export_to_csv(self):
        """Test exporting predictions to CSV"""
        # Make predictions
        predictions = self.model.predict(self.X)
        probs = self.model.predict_proba(self.X)

        # Create predictions dataframe (no company_id in features_df, just use index)
        pred_df = pd.DataFrame(
            {
                "company_index": range(len(predictions)),
                "prediction": predictions,
                "probability": probs[:, 1],
            }
        )

        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name
            pred_df.to_csv(f, index=False)

        try:
            # Verify file was created and can be read
            loaded_df = pd.read_csv(temp_path)
            assert len(loaded_df) == len(self.X), f"Exported CSV should have {len(self.X)} rows"
            assert "company_index" in loaded_df.columns, "Should have company_index"
            assert "prediction" in loaded_df.columns, "Should have prediction"
            assert "probability" in loaded_df.columns, "Should have probability"
            print("✓ test_prediction_export_to_csv PASSED")
        finally:
            os.unlink(temp_path)

    def test_compliance_distribution(self):
        """Test that compliance predictions show expected distribution"""
        predictions = self.model.predict(self.X)

        # Based on data: ~385 Non-compliant (class 0), ~110 Compliant (class 1)
        # Prediction counts on this dataset:
        compliant = np.sum(predictions == 1)
        non_compliant = np.sum(predictions == 0)

        assert compliant > 50, f"Should have > 50 compliant (class 1), got {compliant}"
        assert non_compliant + compliant == len(self.X), f"Totals should sum to {len(self.X)}"

        non_compliant_pct = (non_compliant / 496) * 100
        print(
            f"✓ test_compliance_distribution PASSED ({non_compliant_pct:.1f}% non-compliant)"
        )

    def test_metadata_completeness(self):
        """Test that metadata contains all required fields"""
        required_fields = [
            "training_date",
            "data_shape",
            "train_test_split",
            "hyperparameters",
            "performance",
            "confusion_matrix",
            "features",
            "feature_count",
            "phases_applied",
        ]

        for field in required_fields:
            assert field in self.metadata, f"Metadata should have {field}"

        # Check nested fields
        assert "test_accuracy" in self.metadata["performance"], (
            "Should have test_accuracy"
        )
        assert "true_positives" in self.metadata["confusion_matrix"], (
            "Should have TP count"
        )
        assert len(self.metadata["features"]) == 19, "Should have 19 features"

        print("✓ test_metadata_completeness PASSED")

    def test_model_reproducibility(self):
        """Test that model produces consistent results"""
        # Make predictions twice
        pred1 = self.model.predict(self.X.head(50))
        pred2 = self.model.predict(self.X.head(50))

        # Should be identical
        assert np.array_equal(pred1, pred2), (
            "Model should produce consistent predictions"
        )

        # Probabilities should also match
        prob1 = self.model.predict_proba(self.X.head(50))
        prob2 = self.model.predict_proba(self.X.head(50))

        assert np.allclose(prob1, prob2), "Probabilities should be consistent"
        print("✓ test_model_reproducibility PASSED")

    def test_feature_requirements_met(self):
        """Test that all required features are available"""
        required_features = self.metadata["features"]
        available_features = list(self.X.columns)

        for feat in required_features:
            assert feat in available_features, f"Feature {feat} should be available"

        print(
            f"✓ test_feature_requirements_met PASSED ({len(required_features)} features)"
        )

    def test_data_quality_no_nan(self):
        """Test that training data has no NaN values"""
        nan_count = self.X.isna().sum().sum()
        assert nan_count == 0, f"Data should have no NaN values, found {nan_count}"
        print("✓ test_data_quality_no_nan PASSED")

    def test_feature_scaling_compatibility(self):
        """Test that scaler is compatible with features"""
        assert self.scaler is not None, "Scaler should exist"

        # Scaler should have been fitted on the data
        assert hasattr(self.scaler, "mean_"), "Scaler should have mean_"
        assert hasattr(self.scaler, "scale_"), "Scaler should have scale_"

        # Transform a sample to verify compatibility
        sample = self.X.head(1)
        try:
            scaled = self.scaler.transform(sample)
            assert scaled.shape == sample.shape, "Scaled shape should match input"
            print("✓ test_feature_scaling_compatibility PASSED")
        except Exception as e:
            print(f"✗ test_feature_scaling_compatibility FAILED: {e}")
            raise

    def test_model_performance_validated(self):
        """Test that model performance metrics are validated"""
        perf = self.metadata["performance"]

        # Check that metrics are reasonable (all between 0 and 1)
        metrics = [
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_auc",
        ]
        for metric in metrics:
            assert metric in perf, f"Should have {metric}"
            assert 0 <= perf[metric] <= 1, f"{metric} should be between 0 and 1"

        # Check that test metrics meet expected thresholds
        assert perf["test_accuracy"] >= 0.90, "Accuracy should be >= 90%"
        assert perf["test_recall"] >= 0.97, "Recall should be >= 97%"
        assert perf["test_f1"] >= 0.94, "F1 should be >= 94%"

        print("✓ test_model_performance_validated PASSED")

    def test_artifacts_file_sizes(self):
        """Test that model artifacts have reasonable file sizes"""
        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "final_trained_model.pkl"
        )
        scaler_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "feature_scaler.pkl"
        )
        metadata_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "model_metadata.json"
        )

        # Check files exist and have reasonable sizes
        assert os.path.exists(model_path), "Model file should exist"
        assert os.path.exists(scaler_path), "Scaler file should exist"
        assert os.path.exists(metadata_path), "Metadata file should exist"

        model_size = os.path.getsize(model_path)
        scaler_size = os.path.getsize(scaler_path)
        metadata_size = os.path.getsize(metadata_path)

        assert model_size > 100000, (
            f"Model should be > 100KB, got {model_size / 1024:.1f}KB"
        )
        assert scaler_size > 500, f"Scaler should be > 500B, got {scaler_size}B"
        assert metadata_size > 500, f"Metadata should be > 500B, got {metadata_size}B"

        print(f"✓ test_artifacts_file_sizes PASSED (model={model_size / 1024:.1f}KB)")

    def test_batch_prediction_scalability(self):
        """Test that model can handle batch predictions"""
        # Test with different batch sizes
        batch_sizes = [1, 10, 100, 496]

        for batch_size in batch_sizes:
            if batch_size > len(self.X):
                batch_size = len(self.X)
            batch = self.X.head(batch_size)
            preds = self.model.predict(batch)
            assert len(preds) == batch_size, f"Should handle batch size {batch_size}"

        print("✓ test_batch_prediction_scalability PASSED")

    def test_prediction_threshold_compatibility(self):
        """Test that predictions align with different thresholds"""
        # Get probability for non-compliant class (class 1)
        probs = self.model.predict_proba(self.X)
        prob_non_compliant = probs[:, 1]

        # With threshold 0.15 (recommended for compliance)
        threshold_015 = (prob_non_compliant >= 0.15).astype(int)

        # With threshold 0.5 (default)
        threshold_050 = (prob_non_compliant >= 0.5).astype(int)

        # Both should produce reasonable distributions
        assert np.sum(threshold_015) > 50, "Should have compliant predictions with 0.15 threshold"
        assert np.sum(threshold_050) > 30, "Should have compliant predictions with 0.5 threshold"

        print("✓ test_prediction_threshold_compatibility PASSED")


def run_all_integration_tests():
    """Run all integration tests"""
    test_instance = TestIntegrationPipeline()
    TestIntegrationPipeline.setup_class()

    tests = [
        test_instance.test_full_prediction_pipeline,
        test_instance.test_prediction_export_to_csv,
        test_instance.test_compliance_distribution,
        test_instance.test_metadata_completeness,
        test_instance.test_model_reproducibility,
        test_instance.test_feature_requirements_met,
        test_instance.test_data_quality_no_nan,
        test_instance.test_feature_scaling_compatibility,
        test_instance.test_model_performance_validated,
        test_instance.test_artifacts_file_sizes,
        test_instance.test_batch_prediction_scalability,
        test_instance.test_prediction_threshold_compatibility,
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
    print("INTEGRATION TESTS")
    print("=" * 60)
    passed, failed = run_all_integration_tests()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
