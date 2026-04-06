"""
Phase 3: Hyperparameter Tuning
Systematic optimization of XGBoost hyperparameters using GridSearchCV

Parameters to Optimize:
1. Tree structure: max_depth, min_child_weight, gamma
2. Regularization: lambda (L2), alpha (L1), colsample_bytree, subsample
3. Learning: learning_rate, n_estimators
4. Split selection: max_delta_step

Expected Impact: 93% → 95.5% accuracy (+2.5%)
Uses StratifiedKFold for robust cross-validation on imbalanced data
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
import xgboost as xgb
from dataclasses import dataclass
import json


@dataclass
class HyperparameterResults:
    """Hyperparameter tuning results"""

    best_params: Dict
    best_score: float
    cv_results: Dict
    estimator: xgb.XGBClassifier
    param_grid: Dict


class HyperparameterTuner:
    """
    Systematic XGBoost hyperparameter optimization

    Strategy:
    1. Phase 1: Tune tree structure (max_depth, min_child_weight)
    2. Phase 2: Tune regularization (lambda, alpha, colsample)
    3. Phase 3: Fine-tune learning rate and n_estimators
    4. Phase 4: Final refinement with RandomSearch

    Uses StratifiedKFold for imbalanced data
    """

    def __init__(self, random_state=42, n_jobs=-1):
        """
        Initialize hyperparameter tuner

        Args:
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_model = None
        self.tuning_results = []
        self.scale_pos_weight = 1.0

    def _get_cv_splitter(self, n_splits: int = 5):
        """Get StratifiedKFold for imbalanced classification"""
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )

    def tune_tree_structure(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scale_pos_weight: float = 1.0,
        cv: int = 5,
        verbose: int = 1,
    ) -> HyperparameterResults:
        """
        Phase 1: Tune tree structure parameters

        Focus: max_depth, min_child_weight, gamma
        These control model complexity and overfitting

        Args:
            X: Feature matrix
            y: Target labels
            scale_pos_weight: Class weight for imbalance
            cv: Number of cross-validation folds
            verbose: Verbosity level

        Returns:
            HyperparameterResults with best params
        """
        print("\n[PHASE 3.1] Tuning Tree Structure")
        print("  Parameters: max_depth, min_child_weight, gamma")

        self.scale_pos_weight = scale_pos_weight

        # Grid for tree structure
        param_grid = {
            "max_depth": [4, 5, 6, 7, 8],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0, 0.1, 0.3, 0.5],
        }

        # Base model
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
            tree_method="hist",
        )

        # Grid search
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=self._get_cv_splitter(cv),
            scoring="f1_weighted",
            n_jobs=self.n_jobs,
            verbose=verbose,
        )

        grid_search.fit(X, y)

        results = HyperparameterResults(
            best_params=grid_search.best_params_,
            best_score=grid_search.best_score_,
            cv_results=grid_search.cv_results_,
            estimator=grid_search.best_estimator_,
            param_grid=param_grid,
        )

        print(f"  Best params: {results.best_params}")
        print(f"  Best F1 score: {results.best_score:.4f}")

        return results

    def tune_regularization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        best_tree_params: Dict,
        scale_pos_weight: float = 1.0,
        cv: int = 5,
        verbose: int = 1,
    ) -> HyperparameterResults:
        """
        Phase 2: Tune regularization parameters

        Focus: reg_lambda (L2), reg_alpha (L1), colsample_bytree, subsample
        These prevent overfitting

        Args:
            X: Feature matrix
            y: Target labels
            best_tree_params: Best params from Phase 1
            scale_pos_weight: Class weight
            cv: Number of CV folds
            verbose: Verbosity level

        Returns:
            HyperparameterResults with best params
        """
        print("\n[PHASE 3.2] Tuning Regularization")
        print("  Parameters: reg_lambda, reg_alpha, colsample_bytree, subsample")

        param_grid = {
            "reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0],
            "reg_alpha": [0, 0.1, 0.5, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
            "subsample": [0.6, 0.7, 0.8, 0.9],
        }

        # Base model with best tree params
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
            tree_method="hist",
            **best_tree_params,
        )

        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=self._get_cv_splitter(cv),
            scoring="f1_weighted",
            n_jobs=self.n_jobs,
            verbose=verbose,
        )

        grid_search.fit(X, y)

        results = HyperparameterResults(
            best_params={**best_tree_params, **grid_search.best_params_},
            best_score=grid_search.best_score_,
            cv_results=grid_search.cv_results_,
            estimator=grid_search.best_estimator_,
            param_grid=param_grid,
        )

        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best F1 score: {results.best_score:.4f}")

        return results

    def tune_learning_rate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        best_params: Dict,
        scale_pos_weight: float = 1.0,
        cv: int = 5,
        verbose: int = 1,
    ) -> HyperparameterResults:
        """
        Phase 3: Tune learning dynamics

        Focus: learning_rate, n_estimators
        Trade-off between speed and accuracy

        Args:
            X: Feature matrix
            y: Target labels
            best_params: Best params from Phase 2
            scale_pos_weight: Class weight
            cv: Number of CV folds
            verbose: Verbosity level

        Returns:
            HyperparameterResults with best params
        """
        print("\n[PHASE 3.3] Tuning Learning Rate & Estimators")
        print("  Parameters: learning_rate, n_estimators")

        param_grid = {
            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
            "n_estimators": [100, 150, 200, 300],
        }

        xgb_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
            tree_method="hist",
            **best_params,
        )

        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=self._get_cv_splitter(cv),
            scoring="f1_weighted",
            n_jobs=self.n_jobs,
            verbose=verbose,
        )

        grid_search.fit(X, y)

        results = HyperparameterResults(
            best_params={**best_params, **grid_search.best_params_},
            best_score=grid_search.best_score_,
            cv_results=grid_search.cv_results_,
            estimator=grid_search.best_estimator_,
            param_grid=param_grid,
        )

        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best F1 score: {results.best_score:.4f}")

        return results

    def fine_tune_random_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_params: Dict,
        scale_pos_weight: float = 1.0,
        cv: int = 5,
        n_iter: int = 50,
        verbose: int = 1,
    ) -> HyperparameterResults:
        """
        Phase 4: Fine-tune with RandomizedSearchCV

        Explores broader parameter space with fewer evaluations

        Args:
            X: Feature matrix
            y: Target labels
            base_params: Base parameters from Phase 3
            scale_pos_weight: Class weight
            cv: Number of CV folds
            n_iter: Number of random samples
            verbose: Verbosity level

        Returns:
            HyperparameterResults with best params
        """
        print("\n[PHASE 3.4] Fine-tuning with Random Search")
        print(f"  Evaluating {n_iter} random combinations")

        # Broader search space
        param_dist = {
            "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
            "min_child_weight": [1, 2, 3, 4, 5, 6, 7, 8],
            "gamma": [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_lambda": [0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            "reg_alpha": [0, 0.1, 0.5, 1.0, 2.0],
        }

        # Keep best learning rate and n_estimators
        fixed_params = {
            "learning_rate": base_params.get("learning_rate", 0.1),
            "n_estimators": base_params.get("n_estimators", 200),
        }

        xgb_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
            tree_method="hist",
            **fixed_params,
        )

        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=self._get_cv_splitter(cv),
            scoring="f1_weighted",
            n_jobs=self.n_jobs,
            verbose=verbose,
            random_state=self.random_state,
        )

        random_search.fit(X, y)

        results = HyperparameterResults(
            best_params={**fixed_params, **random_search.best_params_},
            best_score=random_search.best_score_,
            cv_results=random_search.cv_results_,
            estimator=random_search.best_estimator_,
            param_grid=param_dist,
        )

        print(f"  Best F1 score: {results.best_score:.4f}")
        print(f"  Best params: {results.best_params}")

        return results

    def tune_full_pipeline(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scale_pos_weight: float = 1.0,
        cv: int = 5,
        use_random_search: bool = True,
    ) -> HyperparameterResults:
        """
        Run complete 4-phase hyperparameter tuning

        Args:
            X: Feature matrix
            y: Target labels
            scale_pos_weight: Class weight for imbalance
            cv: Number of CV folds
            use_random_search: Use RandomSearch phase 4 (slower but more thorough)

        Returns:
            HyperparameterResults with final best params
        """
        print("\n" + "=" * 70)
        print("[PHASE 3] HYPERPARAMETER TUNING - FULL PIPELINE")
        print("=" * 70)

        # Phase 1: Tree structure
        results1 = self.tune_tree_structure(X, y, scale_pos_weight, cv)

        # Phase 2: Regularization
        results2 = self.tune_regularization(
            X, y, results1.best_params, scale_pos_weight, cv
        )

        # Phase 3: Learning rate
        results3 = self.tune_learning_rate(
            X, y, results2.best_params, scale_pos_weight, cv
        )

        # Phase 4: Random search (optional)
        if use_random_search:
            results4 = self.fine_tune_random_search(
                X, y, results3.best_params, scale_pos_weight, cv, n_iter=50
            )
            final_results = results4
        else:
            final_results = results3

        print("\n" + "=" * 70)
        print("TUNING SUMMARY")
        print("=" * 70)
        print(f"Final Best Params:")
        for key, value in sorted(final_results.best_params.items()):
            print(f"  {key:20} = {value}")
        print(f"\nFinal Best F1 Score: {final_results.best_score:.4f}")

        self.best_model = final_results.estimator
        self.tuning_results = [results1, results2, results3]

        return final_results

    def save_results(self, filepath: str):
        """Save tuning results to JSON"""
        if not self.tuning_results:
            print("No tuning results to save")
            return

        results_dict = {
            "phases": len(self.tuning_results),
            "best_params": self.best_model.get_params() if self.best_model else {},
            "tuning_history": [
                {
                    "phase": i + 1,
                    "best_params": result.best_params,
                    "best_score": float(result.best_score),
                }
                for i, result in enumerate(self.tuning_results)
            ],
        }

        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to {filepath}")
