"""
Phase 4: Ensemble Methods (XGBoost-focused)
Combines multiple XGBoost models with different configurations

Ensemble Strategies:
1. Voting Ensemble: Multiple XGBoost models with different hyperparameters
2. Stacking: Meta-learner trained on base model predictions
3. Blending: Weighted average of model predictions

Expected Impact: 95.5% → 96.5% accuracy (+1%)
Improves robustness and reduces overfitting through model diversity
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore")


class EnsembleModels:
    """
    Ensemble methods combining multiple XGBoost models

    Uses different configurations to create model diversity:
    - Conservative: High regularization, small learning rate
    - Balanced: Standard parameters from Phase 3
    - Aggressive: Low regularization, larger learning rate

    Strategies:
    - Voting: Equal or weighted voting
    - Stacking: Meta-learner with base models
    - Blending: Manual weighted averaging
    """

    def __init__(self, random_state=42, scale_pos_weight: float = 1.0):
        """
        Initialize ensemble builder

        Args:
            random_state: Random seed
            scale_pos_weight: Class weight for imbalance
        """
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self.voting_model = None
        self.stacking_model = None
        self.blend_model = None
        self.base_models = {}

    def _build_xgboost_conservative(self) -> xgb.XGBClassifier:
        """Build conservative XGBoost (high regularization)"""
        return xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.01,
            min_child_weight=5,
            gamma=0.5,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            reg_alpha=1.0,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
            tree_method="hist",
        )

    def _build_xgboost_balanced(self) -> xgb.XGBClassifier:
        """Build balanced XGBoost (from Phase 3 tuning)"""
        return xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.01,
            min_child_weight=1,
            gamma=0.3,
            subsample=0.9,
            colsample_bytree=0.6,
            reg_lambda=0.1,
            reg_alpha=1.0,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
            tree_method="hist",
        )

    def _build_xgboost_aggressive(self) -> xgb.XGBClassifier:
        """Build aggressive XGBoost (low regularization)"""
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            min_child_weight=1,
            gamma=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=0.0,
            reg_alpha=0.0,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
            tree_method="hist",
        )

    def create_voting_ensemble(
        self,
        voting: str = "soft",
        weights: Optional[List[float]] = None,
    ) -> VotingClassifier:
        """
        Create voting ensemble combining multiple XGBoost configurations

        Args:
            voting: 'hard' or 'soft' (soft uses probability averaging)
            weights: Optional weights for each model (must sum to 1)

        Returns:
            VotingClassifier instance
        """
        print("\n[PHASE 4.1] Creating Voting Ensemble")
        print(f"  Base models: Conservative, Balanced, Aggressive XGBoost")
        print(f"  Voting strategy: {voting}")

        estimators = [
            ("conservative", self._build_xgboost_conservative()),
            ("balanced", self._build_xgboost_balanced()),
            ("aggressive", self._build_xgboost_aggressive()),
        ]

        self.voting_model = VotingClassifier(
            estimators=estimators,
            voting=voting,
            weights=weights,
        )

        if weights:
            print(
                f"  Model weights: Conservative={weights[0]}, Balanced={weights[1]}, Aggressive={weights[2]}"
            )
        else:
            print(f"  Model weights: Equal (1/3 each)")

        return self.voting_model

    def create_stacking_ensemble(
        self,
        meta_learner=None,
        cv: int = 5,
    ) -> StackingClassifier:
        """
        Create stacking ensemble with meta-learner

        Approach:
        1. Train base models on training data
        2. Generate predictions from base models (meta-features)
        3. Train meta-learner on meta-features

        Args:
            meta_learner: Estimator for level 1 (default: LogisticRegression)
            cv: Cross-validation folds for stacking

        Returns:
            StackingClassifier instance
        """
        print("\n[PHASE 4.2] Creating Stacking Ensemble")
        print(f"  Base models: Conservative, Balanced, Aggressive XGBoost")
        print(
            f"  Meta-learner: {meta_learner.__class__.__name__ if meta_learner else 'LogisticRegression'}"
        )
        print(f"  Cross-validation folds: {cv}")

        estimators = [
            ("conservative", self._build_xgboost_conservative()),
            ("balanced", self._build_xgboost_balanced()),
            ("aggressive", self._build_xgboost_aggressive()),
        ]

        if meta_learner is None:
            meta_learner = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver="lbfgs",
            )

        self.stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=StratifiedKFold(
                n_splits=cv, shuffle=True, random_state=self.random_state
            ),
            stack_method="predict_proba",
        )

        return self.stacking_model

    def create_blend_ensemble(self) -> Dict:
        """
        Create blending ensemble (manual weighted averaging)

        Approach:
        1. Split data: training for base models, holdout for blending
        2. Train base models on training split
        3. Make predictions on holdout split
        4. Learn optimal weights from holdout predictions

        Returns:
            Dict with models and blending logic
        """
        print("\n[PHASE 4.3] Creating Blending Ensemble")
        print(f"  Base models: Conservative, Balanced, Aggressive XGBoost")
        print(f"  Strategy: Learn optimal weights on holdout set")

        self.blend_model = {
            "conservative": self._build_xgboost_conservative(),
            "balanced": self._build_xgboost_balanced(),
            "aggressive": self._build_xgboost_aggressive(),
            "weights": None,  # Will be learned during fitting
        }

        return self.blend_model

    def fit_voting_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        voting: str = "soft",
        weights: Optional[List[float]] = None,
    ):
        """
        Train voting ensemble

        Args:
            X: Training features
            y: Training targets
            voting: 'hard' or 'soft'
            weights: Optional weights
        """
        print(f"\n[TRAINING] Voting Ensemble (voting={voting})")

        self.create_voting_ensemble(voting, weights)
        self.voting_model.fit(X, y)

        print("  ✓ Voting ensemble trained")

    def fit_stacking_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        meta_learner=None,
        cv: int = 5,
    ):
        """
        Train stacking ensemble

        Args:
            X: Training features
            y: Training targets
            meta_learner: Meta-learner estimator
            cv: CV folds for stacking
        """
        print(f"\n[TRAINING] Stacking Ensemble (cv={cv})")

        self.create_stacking_ensemble(meta_learner, cv)
        self.stacking_model.fit(X, y)

        print("  ✓ Stacking ensemble trained")

    def fit_blend_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        blend_ratio: float = 0.2,
    ):
        """
        Train blending ensemble

        Args:
            X: All training features
            y: All training targets
            blend_ratio: Proportion of data for blending (learning weights)
        """
        print(f"\n[TRAINING] Blending Ensemble (blend_ratio={blend_ratio})")

        # Create blend model if not already created
        if self.blend_model is None:
            self.create_blend_ensemble()

        # Split data
        X_train, X_blend, y_train, y_blend = train_test_split(
            X,
            y,
            test_size=blend_ratio,
            random_state=self.random_state,
            stratify=y,
        )

        # Train base models on training split
        print("  Training base models...")
        self.blend_model["conservative"].fit(X_train, y_train)
        self.blend_model["balanced"].fit(X_train, y_train)
        self.blend_model["aggressive"].fit(X_train, y_train)

        # Get predictions on blend set
        print("  Learning optimal weights on holdout set...")
        pred_conservative = self.blend_model["conservative"].predict_proba(X_blend)[
            :, 1
        ]
        pred_balanced = self.blend_model["balanced"].predict_proba(X_blend)[:, 1]
        pred_aggressive = self.blend_model["aggressive"].predict_proba(X_blend)[:, 1]

        # Learn optimal weights (simple grid search)
        best_f1 = 0
        best_weights = [1 / 3, 1 / 3, 1 / 3]

        for w_conservative in np.arange(0, 1.1, 0.1):
            for w_balanced in np.arange(0, 1 - w_conservative, 0.1):
                w_aggressive = 1 - w_conservative - w_balanced

                # Weighted average
                pred_blend = (
                    w_conservative * pred_conservative
                    + w_balanced * pred_balanced
                    + w_aggressive * pred_aggressive
                )

                # Threshold at 0.5
                y_pred = (pred_blend >= 0.5).astype(int)
                f1 = f1_score(y_blend, y_pred)

                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = [w_conservative, w_balanced, w_aggressive]

        self.blend_model["weights"] = best_weights

        print(
            f"  Optimal weights: Conservative={best_weights[0]:.3f}, Balanced={best_weights[1]:.3f}, Aggressive={best_weights[2]:.3f}"
        )
        print(f"  Blend set F1: {best_f1:.4f}")
        print("  ✓ Blending ensemble trained")

    def predict_voting(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from voting ensemble"""
        if self.voting_model is None:
            raise ValueError(
                "Voting ensemble not trained. Call fit_voting_ensemble() first."
            )

        predictions = self.voting_model.predict(X)
        probabilities = self.voting_model.predict_proba(X)[:, 1]

        return predictions, probabilities

    def predict_stacking(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from stacking ensemble"""
        if self.stacking_model is None:
            raise ValueError(
                "Stacking ensemble not trained. Call fit_stacking_ensemble() first."
            )

        predictions = self.stacking_model.predict(X)
        probabilities = self.stacking_model.predict_proba(X)[:, 1]

        return predictions, probabilities

    def predict_blend(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions from blending ensemble"""
        if self.blend_model is None or self.blend_model["weights"] is None:
            raise ValueError(
                "Blending ensemble not trained. Call fit_blend_ensemble() first."
            )

        # Get predictions from all base models
        pred_conservative = self.blend_model["conservative"].predict_proba(X)[:, 1]
        pred_balanced = self.blend_model["balanced"].predict_proba(X)[:, 1]
        pred_aggressive = self.blend_model["aggressive"].predict_proba(X)[:, 1]

        # Weighted average
        weights = self.blend_model["weights"]
        probabilities = (
            weights[0] * pred_conservative
            + weights[1] * pred_balanced
            + weights[2] * pred_aggressive
        )

        predictions = (probabilities >= 0.5).astype(int)

        return predictions, probabilities

    def compare_ensemble_methods(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Train and compare all three ensemble methods

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            Dict with performance metrics for each method
        """
        print("\n" + "=" * 70)
        print("[PHASE 4] ENSEMBLE COMPARISON")
        print("=" * 70)

        results = {}

        # 1. Voting Ensemble (soft)
        print("\n[1/3] Training Voting Ensemble (soft)...")
        self.fit_voting_ensemble(X_train, y_train, voting="soft")
        y_pred_voting, y_prob_voting = self.predict_voting(X_test)

        voting_accuracy = accuracy_score(y_test, y_pred_voting)
        voting_f1 = f1_score(y_test, y_pred_voting)

        results["voting"] = {
            "accuracy": voting_accuracy,
            "f1": voting_f1,
            "predictions": y_pred_voting,
            "probabilities": y_prob_voting,
        }

        print(f"  Voting Accuracy: {voting_accuracy:.4f}, F1: {voting_f1:.4f}")

        # 2. Stacking Ensemble
        print("\n[2/3] Training Stacking Ensemble...")
        self.fit_stacking_ensemble(X_train, y_train, cv=3)  # Reduced CV for speed
        y_pred_stacking, y_prob_stacking = self.predict_stacking(X_test)

        stacking_accuracy = accuracy_score(y_test, y_pred_stacking)
        stacking_f1 = f1_score(y_test, y_pred_stacking)

        results["stacking"] = {
            "accuracy": stacking_accuracy,
            "f1": stacking_f1,
            "predictions": y_pred_stacking,
            "probabilities": y_prob_stacking,
        }

        print(f"  Stacking Accuracy: {stacking_accuracy:.4f}, F1: {stacking_f1:.4f}")

        # 3. Blending Ensemble
        print("\n[3/3] Training Blending Ensemble...")
        self.fit_blend_ensemble(X_train, y_train, blend_ratio=0.2)
        y_pred_blend, y_prob_blend = self.predict_blend(X_test)

        blend_accuracy = accuracy_score(y_test, y_pred_blend)
        blend_f1 = f1_score(y_test, y_pred_blend)

        results["blending"] = {
            "accuracy": blend_accuracy,
            "f1": blend_f1,
            "predictions": y_pred_blend,
            "probabilities": y_prob_blend,
        }

        print(f"  Blending Accuracy: {blend_accuracy:.4f}, F1: {blend_f1:.4f}")

        # Summary
        print("\n" + "=" * 70)
        print("ENSEMBLE COMPARISON RESULTS")
        print("=" * 70)
        print(f"\nMethod          Accuracy    F1 Score")
        print("-" * 70)
        print(f"Voting          {voting_accuracy:.4f}      {voting_f1:.4f}")
        print(f"Stacking        {stacking_accuracy:.4f}      {stacking_f1:.4f}")
        print(f"Blending        {blend_accuracy:.4f}      {blend_f1:.4f}")

        # Find best
        best_method = max(
            [("voting", voting_f1), ("stacking", stacking_f1), ("blending", blend_f1)],
            key=lambda x: x[1],
        )

        print(f"\nBest performing ensemble: {best_method[0].upper()}")

        return results
