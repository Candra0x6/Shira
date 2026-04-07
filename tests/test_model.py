"""
Test Suite for Model Prediction (Phase 3: Hyperparameter Tuning)
Tests for model loading, prediction, and validation
"""

import sys
import os
import pickle
import json
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestModelPrediction:
    """Test suite for model predictions"""

    @classmethod
    def setup_class(cls):
        """Setup model and test data"""
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")

        # Load model
        model_path = os.path.join(model_dir, "final_trained_model.pkl")
        with open(model_path, "rb") as f:
            cls.model = pickle.load(f)

        # Load scaler
        scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
        with open(scaler_path, "rb") as f:
            cls.scaler = pickle.load(f)

        # Load metadata
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        with open(metadata_path, "r") as f:
            cls.metadata = json.load(f)

        # Load engineered features
        engineered_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "engineered_features.csv"
        )
        cls.features_df = pd.read_csv(engineered_path)

        # Use only feature columns (exclude company_id)
        cls.X = cls.features_df.drop("company_id", axis=1, errors="ignore")

        # Load predictions for comparison
        pred_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "reports",
            "predictions_with_explanations.csv",
        )
        if os.path.exists(pred_path):
            cls.predictions_df = pd.read_csv(pred_path)
        else:
            cls.predictions_df = None

    def test_model_loading(self):
        """Test that model loads successfully"""
        assert self.model is not None, "Model should load"
        print("✓ test_model_loading PASSED")

    def test_scaler_loading(self):
        """Test that scaler loads successfully"""
        assert self.scaler is not None, "Scaler should load"
        print("✓ test_scaler_loading PASSED")

    def test_model_type(self):
        """Test that model is XGBoost classifier"""
        model_type = type(self.model).__name__
        assert "XGB" in model_type or "Booster" in model_type, (
            f"Model should be XGBoost, got {model_type}"
        )
        print(f"✓ test_model_type PASSED (type={model_type})")

    def test_predictions_shape(self):
        """Test that predictions have correct shape"""
        predictions = self.model.predict(self.X)

        assert len(predictions) == len(self.X), (
            f"Should have {len(self.X)} predictions, got {len(predictions)}"
        )
        print(f"✓ test_predictions_shape PASSED (samples={len(predictions)})")

    def test_predictions_binary(self):
        """Test that predictions are binary (0 or 1)"""
        predictions = self.model.predict(self.X)

        unique_values = np.unique(predictions)
        assert all(val in [0, 1] for val in unique_values), (
            "Predictions should be binary (0 or 1)"
        )
        print(f"✓ test_predictions_binary PASSED (classes={unique_values.tolist()})")

    def test_prediction_probabilities(self):
        """Test that prediction probabilities are valid"""
        probs = self.model.predict_proba(self.X)

        assert probs is not None, "Should have probability predictions"
        assert probs.shape[0] == len(self.X), (
            f"Should have {len(self.X)} probability rows"
        )
        assert probs.shape[1] == 2, (
            "Should have 2 probability columns (for binary classification)"
        )
        print(f"✓ test_prediction_probabilities PASSED (shape={probs.shape})")

    def test_probabilities_range(self):
        """Test that probabilities are between 0 and 1"""
        probs = self.model.predict_proba(self.X)

        assert np.all(probs >= 0), "Probabilities should be >= 0"
        assert np.all(probs <= 1), "Probabilities should be <= 1"
        print("✓ test_probabilities_range PASSED")

    def test_probabilities_sum_to_one(self):
        """Test that probability columns sum to 1"""
        probs = self.model.predict_proba(self.X)

        sums = probs.sum(axis=1)
        assert np.allclose(sums, 1.0), "Probabilities should sum to 1"
        print("✓ test_probabilities_sum_to_one PASSED")

    def test_metadata_accuracy(self):
        """Test that metadata contains expected accuracy values"""
        assert "performance" in self.metadata, "Metadata should have performance key"
        perf = self.metadata["performance"]

        assert "test_accuracy" in perf, "Should have test accuracy"
        assert perf["test_accuracy"] == 0.92, "Test accuracy should be 92%"
        assert perf["test_recall"] >= 0.974, "Recall should be >= 97.4%"
        assert perf["test_f1"] == 0.95, "F1 score should be 95%"
        print(
            f"✓ test_metadata_accuracy PASSED (accuracy={perf['test_accuracy'] * 100}%)"
        )

    def test_metadata_features(self):
        """Test that metadata lists all features"""
        assert "features" in self.metadata, "Metadata should list features"
        assert len(self.metadata["features"]) == 19, "Should have 19 features"

        # Check critical features are present
        critical = ["debt_to_equity", "nonhalal_revenue_percent", "profit_margin"]
        for feat in critical:
            assert feat in self.metadata["features"], (
                f"Feature {feat} should be in metadata"
            )

        print(
            f"✓ test_metadata_features PASSED (count={len(self.metadata['features'])})"
        )

    def test_confusion_matrix_values(self):
        """Test confusion matrix values in metadata"""
        assert "confusion_matrix" in self.metadata, (
            "Metadata should have confusion matrix"
        )
        cm = self.metadata["confusion_matrix"]

        assert cm["true_positives"] == 76, "Should have 76 true positives"
        assert cm["false_negatives"] == 2, "Should have 2 false negatives"
        assert cm["false_positives"] == 6, "Should have 6 false positives"
        print(
            f"✓ test_confusion_matrix_values PASSED (TP={cm['true_positives']}, FN={cm['false_negatives']})"
        )

    def test_feature_count_matches_metadata(self):
        """Test that feature count matches metadata"""
        actual_features = self.X.shape[1]
        metadata_features = self.metadata["feature_count"]

        assert actual_features == metadata_features, (
            f"Feature count mismatch: {actual_features} vs {metadata_features}"
        )
        print(f"✓ test_feature_count_matches_metadata PASSED (count={actual_features})")

    def test_model_consistency(self):
        """Test that model produces consistent predictions"""
        pred1 = self.model.predict(self.X.head(10))
        pred2 = self.model.predict(self.X.head(10))

        assert np.array_equal(pred1, pred2), (
            "Model should produce consistent predictions"
        )
        print("✓ test_model_consistency PASSED")

    def test_predictions_distribution(self):
        """Test that predictions show reasonable distribution"""
        predictions = self.model.predict(self.X)

        # Count classes
        non_compliant = np.sum(predictions == 1)
        compliant = np.sum(predictions == 0)

        # Based on metadata: ~76 non-compliant (class 1), ~22 compliant (class 0)
        # Allow some variance due to decision threshold
        assert non_compliant > 50, (
            "Should have significant number of non-compliant predictions"
        )
        assert compliant > 10, "Should have some compliant predictions"
        print(
            f"✓ test_predictions_distribution PASSED (non-compliant={non_compliant}, compliant={compliant})"
        )

    def test_hyperparameters_in_metadata(self):
        """Test that hyperparameters are documented"""
        assert "hyperparameters" in self.metadata, "Should have hyperparameters"
        hp = self.metadata["hyperparameters"]

        assert hp["n_estimators"] == 150, "Should have 150 estimators"
        assert hp["max_depth"] == 6, "Should have max_depth=6"
        assert hp["learning_rate"] == 0.01, "Should have learning_rate=0.01"
        print("✓ test_hyperparameters_in_metadata PASSED")


def run_all_model_tests():
    """Run all model tests"""
    test_instance = TestModelPrediction()
    TestModelPrediction.setup_class()

    tests = [
        test_instance.test_model_loading,
        test_instance.test_scaler_loading,
        test_instance.test_model_type,
        test_instance.test_predictions_shape,
        test_instance.test_predictions_binary,
        test_instance.test_prediction_probabilities,
        test_instance.test_probabilities_range,
        test_instance.test_probabilities_sum_to_one,
        test_instance.test_metadata_accuracy,
        test_instance.test_metadata_features,
        test_instance.test_confusion_matrix_values,
        test_instance.test_feature_count_matches_metadata,
        test_instance.test_model_consistency,
        test_instance.test_predictions_distribution,
        test_instance.test_hyperparameters_in_metadata,
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
    print("MODEL PREDICTION TESTS")
    print("=" * 60)
    passed, failed = run_all_model_tests()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
