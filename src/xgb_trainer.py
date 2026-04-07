"""
XGBoost Model Training on Real Shariah Compliance Data
Train a model to predict Shariah compliance using 12 engineered financial ratios.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import pickle
import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class XGBoostShariaCompliance:
    """Train and evaluate XGBoost model for Shariah compliance prediction."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_results = {}
        self.test_results = {}

    def load_labeled_data(self, csv_path: str) -> pd.DataFrame:
        """Load labeled data with features and Shariah compliance labels."""
        logger.info(f"[XGB] Loading labeled data from {csv_path}")
        df = pd.read_csv(csv_path)

        # Check if we need to merge with features
        feature_cols_available = [
            col
            for col in df.columns
            if col
            not in [
                "symbol",
                "sector",
                "shariah_compliant",
                "sector_compliant",
                "financial_compliant",
                "debt_to_assets_ok",
                "interest_income_ok",
                "interest_bearing_debt_ok",
                "profitability_ok",
                "equity_ok",
                "debt_to_assets_value",
                "interest_income_value",
                "interest_bearing_debt_value",
                "roa_value",
                "equity_ratio_value",
            ]
        ]

        if len(feature_cols_available) == 0:
            # Merge with features CSV
            base_dir = os.path.dirname(csv_path)
            features_csv = os.path.join(base_dir, "companies_with_features.csv")
            logger.info(f"[XGB] Merging with features from {features_csv}")
            df_features = pd.read_csv(features_csv)
            df = df.merge(df_features, on="symbol", how="left")
            logger.info(f"[XGB] Merged {len(df)} companies with features")

        # Get feature columns (all except symbol, sector, and label columns)
        exclude_cols = {
            "symbol",
            "sector",
            "shariah_compliant",
            "sector_compliant",
            "financial_compliant",
            "debt_to_assets_ok",
            "interest_income_ok",
            "interest_bearing_debt_ok",
            "profitability_ok",
            "equity_ok",
            "debt_to_assets_value",
            "interest_income_value",
            "interest_bearing_debt_value",
            "roa_value",
            "equity_ratio_value",
        }

        self.feature_cols = [col for col in df.columns if col not in exclude_cols]

        logger.info(
            f"[XGB] Loaded {len(df)} companies with {len(self.feature_cols)} features"
        )
        return df

    def prepare_data(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare training and test data."""
        logger.info("[XGB] Preparing training and test data...")

        # Fill NaN with median
        X = df[self.feature_cols].fillna(df[self.feature_cols].median())
        y = df["shariah_compliant"]

        # Remove rows with NaN in target
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]

        logger.info(f"[XGB] After NaN removal: {len(X)} samples")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train.values
        self.y_test = y_test.values

        logger.info(
            f"[XGB] Train set: {len(X_train)} samples, Test set: {len(X_test)} samples"
        )
        logger.info(
            f"[XGB] Class distribution - Train: {y_train.value_counts().to_dict()}"
        )

        return X_train_scaled, X_test_scaled, y_train.values, y_test.values

    def train_model(self, **xgb_params):
        """Train XGBoost model."""
        logger.info("[XGB] Training XGBoost model...")

        # Default hyperparameters
        default_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "random_state": self.random_state,
        }

        # Update with provided params
        default_params.update(xgb_params)

        self.model = xgb.XGBClassifier(**default_params, use_label_encoder=False)
        self.model.fit(self.X_train, self.y_train)

        logger.info("[XGB] Model training complete")

    def evaluate(self) -> dict:
        """Evaluate model on test set."""
        logger.info("[XGB] Evaluating model on test set...")

        # Predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        # Metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        cm = confusion_matrix(self.y_test, y_pred)

        self.test_results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
        }

        logger.info(f"[XGB] Test Accuracy: {accuracy:.4f}")
        logger.info(f"[XGB] Test Precision: {precision:.4f}")
        logger.info(f"[XGB] Test Recall: {recall:.4f}")
        logger.info(f"[XGB] Test F1-Score: {f1:.4f}")
        logger.info(f"[XGB] Test ROC-AUC: {roc_auc:.4f}")

        logger.info("\n[XGB] Classification Report:")
        logger.info("\n" + classification_report(self.y_test, y_pred))

        return self.test_results

    def cross_validate(self, n_splits: int = 5) -> dict:
        """Perform stratified k-fold cross-validation."""
        logger.info(f"[XGB] Performing {n_splits}-fold cross-validation...")

        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )

        # Combine train and test for full cross-validation
        X_full = np.vstack([self.X_train, self.X_test])
        y_full = np.hstack([self.y_train, self.y_test])

        cv_scores = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "roc_auc": [],
        }

        fold = 1
        for train_idx, test_idx in skf.split(X_full, y_full):
            X_cv_train = X_full[train_idx]
            X_cv_test = X_full[test_idx]
            y_cv_train = y_full[train_idx]
            y_cv_test = y_full[test_idx]

            # Train model
            model_cv = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                use_label_encoder=False,
            )
            model_cv.fit(X_cv_train, y_cv_train)

            # Evaluate
            y_pred_cv = model_cv.predict(X_cv_test)
            y_pred_proba_cv = model_cv.predict_proba(X_cv_test)[:, 1]

            cv_scores["accuracy"].append(accuracy_score(y_cv_test, y_pred_cv))
            cv_scores["precision"].append(
                precision_score(y_cv_test, y_pred_cv, zero_division=0)
            )
            cv_scores["recall"].append(
                recall_score(y_cv_test, y_pred_cv, zero_division=0)
            )
            cv_scores["f1"].append(f1_score(y_cv_test, y_pred_cv, zero_division=0))
            cv_scores["roc_auc"].append(roc_auc_score(y_cv_test, y_pred_proba_cv))

            logger.info(
                f"[XGB] Fold {fold}: Accuracy={cv_scores['accuracy'][-1]:.4f}, "
                f"F1={cv_scores['f1'][-1]:.4f}, ROC-AUC={cv_scores['roc_auc'][-1]:.4f}"
            )
            fold += 1

        # Summarize
        cv_summary = {
            metric: {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "scores": scores,
            }
            for metric, scores in cv_scores.items()
        }

        self.train_results["cv"] = cv_summary

        logger.info("\n[XGB] Cross-Validation Summary:")
        for metric, stats in cv_summary.items():
            logger.info(
                f"[XGB] {metric.upper()}: {stats['mean']:.4f} (+/- {stats['std']:.4f})"
            )

        return cv_summary

    def save_model(self, model_path: str, scaler_path: str) -> None:
        """Save trained model and scaler."""
        logger.info(f"[XGB] Saving model to {model_path}")
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

        logger.info(f"[XGB] Saving scaler to {scaler_path}")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        logger.info("[XGB] Model and scaler saved successfully")

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importances from trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame(
            {"feature": self.feature_cols, "importance": importances}
        ).sort_values("importance", ascending=False)

        logger.info("\n[XGB] Top 10 Feature Importances:")
        for idx, row in feature_importance_df.head(10).iterrows():
            logger.info(f"[XGB]   {row['feature']:30} {row['importance']:.4f}")

        return feature_importance_df


def train_xgboost_model(
    labeled_data_csv: str,
    model_output_path: str,
    scaler_output_path: str,
) -> Tuple:
    """
    Train XGBoost model on labeled Shariah compliance data.

    Args:
        labeled_data_csv: Path to companies_with_labels.csv
        model_output_path: Where to save the trained model
        scaler_output_path: Where to save the feature scaler

    Returns:
        Tuple of (trained_model, scaler, test_results)
    """
    trainer = XGBoostShariaCompliance()

    # Load data
    df = trainer.load_labeled_data(labeled_data_csv)

    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)

    # Train model
    trainer.train_model()

    # Evaluate
    test_results = trainer.evaluate()

    # Cross-validate
    cv_results = trainer.cross_validate(n_splits=5)

    # Save model
    trainer.save_model(model_output_path, scaler_output_path)

    # Feature importance
    feature_importance = trainer.get_feature_importance()

    return trainer.model, trainer.scaler, test_results, feature_importance


if __name__ == "__main__":
    import os

    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    labeled_data_csv = f"{project_root}/data/processed/companies_with_labels.csv"
    model_output_path = f"{project_root}/models/xgb_shariah_model.pkl"
    scaler_output_path = f"{project_root}/models/xgb_scaler.pkl"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

    # Train model
    model, scaler, test_results, feature_importance = train_xgboost_model(
        labeled_data_csv, model_output_path, scaler_output_path
    )

    print("\n[TEST] XGBoost Training Complete!")
    print(f"[TEST] Model saved to: {model_output_path}")
    print(f"[TEST] Scaler saved to: {scaler_output_path}")
    print(f"\n[TEST] Test Results:")
    for metric, value in test_results.items():
        if metric != "confusion_matrix":
            print(f"  {metric}: {value:.4f}")

    print(f"\n[TEST] Top 5 Important Features:")
    print(feature_importance.head(5).to_string(index=False))
