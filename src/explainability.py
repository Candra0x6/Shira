"""
Explanation Generation for Shariah Compliance Predictions
Generate feature contribution explanations to explain model decisions.

Note: Uses built-in XGBoost feature importance instead of SHAP
for compatibility in restricted environments.
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ExplainabilityGenerator:
    """Generate explanations for Shariah compliance predictions."""

    def __init__(self, model_path: str, scaler_path: str):
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.X_data = None
        self.feature_cols = None
        self.predictions = None

        self._load_model(model_path)
        self._load_scaler(scaler_path)

    def _load_model(self, model_path: str) -> None:
        """Load trained XGBoost model."""
        logger.info(f"[EXPLAIN] Loading model from {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("[EXPLAIN] Model loaded successfully")

    def _load_scaler(self, scaler_path: str) -> None:
        """Load feature scaler."""
        logger.info(f"[EXPLAIN] Loading scaler from {scaler_path}")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        logger.info("[EXPLAIN] Scaler loaded successfully")

    # List of 19 features in the exact order required by the model
    MODEL_FEATURES = [
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

    def prepare_data(
        self, features_csv: str, labels_csv: str, feature_cols: list = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare data for explanations."""
        logger.info("[EXPLAIN] Preparing data for explanations...")

        # Load features
        df_features = pd.read_csv(features_csv)

        # Load labels for merging
        df_labels = pd.read_csv(labels_csv)
        df_labels = df_labels[["symbol", "shariah_compliant"]]

        # Merge
        df = df_features.merge(df_labels, on="symbol", how="left")

        # Use the 19 features required by the production model
        if feature_cols is None:
            feature_cols = self.MODEL_FEATURES

        # Ensure all required features exist
        for col in feature_cols:
            if col not in df.columns:
                logger.warning(f"[EXPLAIN] Missing feature column: {col}. Filling with 0.")
                df[col] = 0.0

        # Select only these features from the dataframe
        X = df[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)

        self.X_data = X_scaled
        self.feature_cols = feature_cols

        logger.info(
            f"[EXPLAIN] Prepared {len(X_scaled)} samples with {len(feature_cols)} features"
        )

        return df, X_scaled

    def compute_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        logger.info("[EXPLAIN] Computing feature importances...")

        importances = self.model.feature_importances_

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_cols,
                "importance": importances,
                "importance_pct": (importances / importances.sum()) * 100,
            }
        ).sort_values("importance", ascending=False)

        logger.info(f"\n[EXPLAIN] Top 12 Features by Importance:")
        for idx, row in importance_df.head(12).iterrows():
            logger.info(
                f"[EXPLAIN]   {row['feature']:30} {row['importance']:.6f} ({row['importance_pct']:5.2f}%)"
            )

        self.feature_importance = importance_df
        return importance_df

    def generate_predictions_with_explanations(
        self, df: pd.DataFrame, output_csv: str
    ) -> pd.DataFrame:
        """Generate predictions with explanations for all samples."""
        logger.info("[EXPLAIN] Generating predictions with explanations...")

        # Get predictions and probabilities
        self.predictions = self.model.predict(self.X_data)
        pred_proba = self.model.predict_proba(self.X_data)

        # Create output dataframe
        result_df = df[["symbol", "shariah_compliant"]].copy()
        result_df["predicted_compliant"] = self.predictions
        result_df["compliance_probability"] = pred_proba[:, 1]
        result_df["confidence"] = np.max(pred_proba, axis=1)
        result_df["prediction_correct"] = (
            result_df["shariah_compliant"] == result_df["predicted_compliant"]
        ).astype(int)

        logger.info(f"[EXPLAIN] Generated predictions for {len(result_df)} samples")
        logger.info(
            f"[EXPLAIN] Accuracy: {result_df['prediction_correct'].sum() / len(result_df) * 100:.2f}%"
        )

        # Save
        logger.info(f"[EXPLAIN] Saving predictions to {output_csv}")
        result_df.to_csv(output_csv, index=False)
        logger.info(f"[EXPLAIN] Saved {len(result_df)} predictions to {output_csv}")

        return result_df

    def get_feature_importance_dataframe(self) -> pd.DataFrame:
        """Get feature importance as dataframe."""
        if self.feature_importance is None:
            self.compute_feature_importance()
        return self.feature_importance


def generate_model_explanations(
    model_path: str,
    scaler_path: str,
    features_csv: str,
    labels_csv: str,
    output_csv: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate explanations for all predictions.

    Args:
        model_path: Path to trained XGBoost model
        scaler_path: Path to feature scaler
        features_csv: Path to engineered features
        labels_csv: Path to labels
        output_csv: Where to save predictions with explanations

    Returns:
        Tuple of (predictions dataframe, feature importance dataframe)
    """
    explainer = ExplainabilityGenerator(model_path, scaler_path)

    # Prepare data
    df, X_scaled = explainer.prepare_data(features_csv, labels_csv)

    # Compute feature importances
    importance_df = explainer.compute_feature_importance()

    # Generate predictions with explanations
    predictions_df = explainer.generate_predictions_with_explanations(df, output_csv)

    return predictions_df, importance_df


if __name__ == "__main__":
    import os

    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    model_path = f"{project_root}/models/final_trained_model.pkl"
    scaler_path = f"{project_root}/models/feature_scaler.pkl"
    features_csv = f"{project_root}/data/processed/companies_with_features.csv"
    labels_csv = f"{project_root}/data/processed/companies_with_labels.csv"
    output_csv = f"{project_root}/reports/model_predictions_explanations.csv"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Generate explanations
    predictions_df, importance_df = generate_model_explanations(
        model_path,
        scaler_path,
        features_csv,
        labels_csv,
        output_csv,
    )

    print("\n[TEST] Model Explanation Generation Complete!")
    print(f"[TEST] Predictions saved to: {output_csv}")
    print(f"[TEST] Output shape: {predictions_df.shape}")
    print(f"\n[TEST] Top 10 Feature Importances:")
    print(importance_df.head(10).to_string(index=False))

    print(f"\n[TEST] Sample predictions:")
    print(
        predictions_df[
            [
                "symbol",
                "shariah_compliant",
                "predicted_compliant",
                "compliance_probability",
                "confidence",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
