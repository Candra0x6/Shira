"""
Phase 6: Comprehensive Evaluation & Analysis
Final model evaluation with detailed metrics, visualization data, and feature analysis

Evaluation Components:
1. Classification Metrics: Accuracy, precision, recall, F1, ROC-AUC
2. Confusion Matrix: TP, FP, TN, FN analysis
3. Feature Importance: Which features drive predictions
4. Error Analysis: Characterize misclassifications
5. Business Metrics: Domain-specific performance indicators

Expected Impact: Final tuning and optimization (+0.5% accuracy possible)
Provides actionable insights for model improvement
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef,
    cohen_kappa_score,
)
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")


class EvaluationMetrics:
    """
    Comprehensive model evaluation and analysis

    Provides:
    - Standard ML metrics (accuracy, F1, etc.)
    - Statistical measures (Matthews correlation, Cohen's kappa)
    - Confusion matrix analysis
    - Feature importance ranking
    - Error characterization
    - Business-focused metrics
    """

    def __init__(self):
        """Initialize evaluation metrics tracker"""
        self.metrics = {}
        self.feature_importances = {}
        self.confusion_matrix_data = None

    def compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Compute comprehensive classification metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            Dict with all classification metrics
        """
        print("\n[PHASE 6.1] Classification Metrics")

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "matthews_corr": matthews_corrcoef(y_true, y_pred),
            "cohens_kappa": cohen_kappa_score(y_true, y_pred),
        }

        if y_proba is not None:
            metrics["auc"] = roc_auc_score(y_true, y_proba)

        self.metrics.update(metrics)

        print(f"  Accuracy:            {metrics['accuracy']:.4f}")
        print(f"  Precision:           {metrics['precision']:.4f}")
        print(f"  Recall:              {metrics['recall']:.4f}")
        print(f"  F1 Score:            {metrics['f1']:.4f}")
        print(f"  Matthews Corr Coef:  {metrics['matthews_corr']:.4f}")
        print(f"  Cohen's Kappa:       {metrics['cohens_kappa']:.4f}")
        if y_proba is not None:
            print(f"  AUC-ROC:             {metrics['auc']:.4f}")

        return metrics

    def compute_confusion_matrix_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict:
        """
        Analyze confusion matrix components

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dict with confusion matrix analysis
        """
        print("\n[PHASE 6.2] Confusion Matrix Analysis")

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
            "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0,
        }

        self.confusion_matrix_data = metrics

        print(f"  Confusion Matrix:")
        print(f"    True Positives:      {tp}")
        print(f"    True Negatives:      {tn}")
        print(f"    False Positives:     {fp}")
        print(f"    False Negatives:     {fn}")
        print(f"  Derived Metrics:")
        print(f"    Specificity:         {metrics['specificity']:.4f}")
        print(f"    Sensitivity:         {metrics['sensitivity']:.4f}")
        print(f"    False Positive Rate: {metrics['fpr']:.4f}")
        print(f"    False Negative Rate: {metrics['fnr']:.4f}")

        return metrics

    def get_feature_importance(
        self,
        model: xgb.XGBClassifier,
        feature_names: List[str],
        top_n: int = 15,
    ) -> Dict:
        """
        Extract and rank feature importances

        Args:
            model: Trained XGBoost model
            feature_names: Names of features
            top_n: Number of top features to return

        Returns:
            Dict with feature importances
        """
        print(f"\n[PHASE 6.3] Feature Importance Analysis")

        importances = model.feature_importances_

        # Create feature importance dataframe
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)

        # Get top N
        top_features = importance_df.head(top_n)

        self.feature_importances = {
            "all": dict(zip(importance_df["feature"], importance_df["importance"])),
            "top_n": dict(zip(top_features["feature"], top_features["importance"])),
            "top_n_value": top_n,
        }

        print(f"  Top {top_n} Important Features:")
        for idx, (feat, imp) in enumerate(
            zip(top_features["feature"], top_features["importance"]), 1
        ):
            print(f"    {idx:2d}. {feat:30s} {imp:.4f}")

        return self.feature_importances

    def analyze_errors(
        self,
        X: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Analyze misclassifications to find patterns

        Args:
            X: Feature dataframe (for analysis context)
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)

        Returns:
            Dict with error analysis
        """
        print("\n[PHASE 6.4] Error Analysis")

        # Identify errors
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]

        # Error types
        false_positives = (y_pred == 1) & (y_true == 0)
        false_negatives = (y_pred == 0) & (y_true == 1)

        fp_indices = np.where(false_positives)[0]
        fn_indices = np.where(false_negatives)[0]

        results = {
            "total_errors": int(np.sum(errors)),
            "error_rate": float(np.mean(errors)),
            "false_positives": int(np.sum(false_positives)),
            "false_negative_rate": float(np.mean(false_negatives)),
            "false_negative_indices": fn_indices.tolist(),
            "false_positive_indices": fp_indices.tolist(),
        }

        # Analyze confidence of errors
        if y_proba is not None:
            error_proba = y_proba[error_indices]
            results["mean_error_confidence"] = float(np.mean(error_proba))
            results["min_error_confidence"] = float(np.min(error_proba))
            results["max_error_confidence"] = float(np.max(error_proba))

            print(f"  Error Confidence Statistics:")
            print(
                f"    Mean confidence of errors: {results['mean_error_confidence']:.4f}"
            )
            print(
                f"    Min confidence of errors:  {results['min_error_confidence']:.4f}"
            )
            print(
                f"    Max confidence of errors:  {results['max_error_confidence']:.4f}"
            )

        print(f"  Error Summary:")
        print(f"    Total Errors:              {results['total_errors']}")
        print(f"    Error Rate:                {results['error_rate']:.4f}")
        print(f"    False Positives:           {results['false_positives']}")
        print(f"    False Negatives:           {len(fn_indices)}")

        return results

    def compute_roc_pr_curves(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict:
        """
        Compute ROC and Precision-Recall curves

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Dict with curve data for plotting
        """
        print("\n[PHASE 6.5] ROC & Precision-Recall Curves")

        # ROC curve
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        # Precision-Recall curve
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_proba)

        results = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds_roc": thresholds_roc.tolist(),
            "auc": float(auc),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds_pr": thresholds_pr.tolist(),
        }

        print(f"  ROC-AUC:           {auc:.4f}")
        print(f"  Precision-Recall curve computed")
        print(f"  Curve data available for visualization")

        return results

    def classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """
        Generate detailed classification report

        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names for classes

        Returns:
            Formatted classification report
        """
        print("\n[PHASE 6.6] Detailed Classification Report")

        report = classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            digits=4,
        )

        print(report)

        return report

    def compute_business_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cost_fp: float = 1.0,
        cost_fn: float = 5.0,
    ) -> Dict:
        """
        Compute business-focused metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_fp: Cost of false positive
            cost_fn: Cost of false negative

        Returns:
            Dict with business metrics
        """
        print("\n[PHASE 6.7] Business Metrics")

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Cost analysis
        total_cost = (fp * cost_fp) + (fn * cost_fn)

        # Precision and recall in business context
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        metrics = {
            "total_cost": float(total_cost),
            "cost_per_sample": float(total_cost / len(y_true)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(2 * (precision * recall) / (precision + recall))
            if (precision + recall) > 0
            else 0,
            "cost_fp": cost_fp,
            "cost_fn": cost_fn,
        }

        print(f"  Total Cost:             {total_cost:.2f}")
        print(f"  Cost per Sample:        {metrics['cost_per_sample']:.4f}")
        print(f"  Cost FP:                {cost_fp:.2f}")
        print(f"  Cost FN:                {cost_fn:.2f}")

        return metrics

    def generate_summary_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        model: Optional[xgb.XGBClassifier] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate comprehensive evaluation summary

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            model: Trained model (for feature importance)
            feature_names: Feature names

        Returns:
            Complete evaluation report
        """
        print("\n" + "=" * 70)
        print("[PHASE 6] COMPREHENSIVE MODEL EVALUATION")
        print("=" * 70)

        # Compute all metrics
        class_metrics = self.compute_classification_metrics(y_true, y_pred, y_proba)
        cm_metrics = self.compute_confusion_matrix_metrics(y_true, y_pred)
        error_analysis = self.analyze_errors(pd.DataFrame(), y_true, y_pred, y_proba)

        # Conditional metrics
        curve_data = None
        if y_proba is not None:
            curve_data = self.compute_roc_pr_curves(y_true, y_proba)

        feature_imp = None
        if model is not None and feature_names is not None:
            feature_imp = self.get_feature_importance(model, feature_names)

        business_metrics = self.compute_business_metrics(y_true, y_pred)

        # Compile full report
        report = {
            "classification": class_metrics,
            "confusion_matrix": cm_metrics,
            "error_analysis": error_analysis,
            "curves": curve_data,
            "feature_importance": feature_imp,
            "business_metrics": business_metrics,
        }

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)

        return report
