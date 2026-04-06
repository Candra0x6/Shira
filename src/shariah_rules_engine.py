"""
Shariah Compliance Rules Engine
Hybrid Rules-Based + ML Model for Islamic Finance

Author: OpenCode
Date: April 2026
Purpose: Deterministic rule engine for Shariah compliance screening
         + ML confidence scoring for edge cases
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class ComplianceStatus(Enum):
    """Shariah compliance decision status"""

    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    EDGE_CASE = "EDGE_CASE"  # Requires ML confidence scoring


class RuleViolation(Enum):
    """Types of Shariah rule violations"""

    HARAM_SECTOR = "HARAM_SECTOR"
    EXCESSIVE_DEBT = "EXCESSIVE_DEBT"
    EXCESSIVE_NONHALAL = "EXCESSIVE_NONHALAL"
    AMBIGUOUS = "AMBIGUOUS"
    INCOMPLETE_DATA = "INCOMPLETE_DATA"


@dataclass
class RuleCheckResult:
    """Result of a single rule check"""

    rule_name: str
    passed: bool
    value: float
    threshold: float
    severity: str  # "CRITICAL" | "WARNING" | "INFO"
    explanation: str

    def to_dict(self):
        return {
            "rule": self.rule_name,
            "passed": self.passed,
            "value": round(self.value, 4)
            if isinstance(self.value, float)
            else self.value,
            "threshold": round(self.threshold, 4)
            if isinstance(self.threshold, float)
            else self.threshold,
            "severity": self.severity,
            "explanation": self.explanation,
        }


@dataclass
class ComplianceDecision:
    """Complete Shariah compliance decision for a company"""

    ticker: str
    company_name: str

    # Rule checks
    rule_results: List[RuleCheckResult] = field(default_factory=list)
    violations: List[RuleViolation] = field(default_factory=list)

    # Status (set during evaluation)
    status: Optional[ComplianceStatus] = None

    # ML confidence (for edge cases)
    ml_confidence: Optional[float] = None  # 0.0-1.0
    ml_score: Optional[float] = None  # Probability of compliance

    # Metadata
    risk_factors: List[str] = field(default_factory=list)
    supporting_factors: List[str] = field(default_factory=list)
    final_confidence: float = 1.0

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "status": self.status.value,
            "final_confidence": round(self.final_confidence, 4),
            "rules": [r.to_dict() for r in self.rule_results],
            "violations": [v.value for v in self.violations],
            "ml_confidence": round(self.ml_confidence, 4)
            if self.ml_confidence
            else None,
            "ml_score": round(self.ml_score, 4) if self.ml_score else None,
            "risk_factors": self.risk_factors,
            "supporting_factors": self.supporting_factors,
        }

    def __str__(self):
        """Human-readable format for display"""
        status_emoji = (
            "✓"
            if self.status == ComplianceStatus.COMPLIANT
            else "✗"
            if self.status == ComplianceStatus.NON_COMPLIANT
            else "⚠"
        )

        output = f"\n{'=' * 70}\n"
        output += f"{status_emoji} {self.ticker:8} | {self.company_name}\n"
        output += f"{'=' * 70}\n"
        output += (
            f"Status: {self.status.value} (Confidence: {self.final_confidence:.1%})\n\n"
        )

        output += "RULE CHECKS:\n"
        for rule in self.rule_results:
            check_mark = "✓" if rule.passed else "✗"
            output += f"  {check_mark} {rule.rule_name:20} {rule.value:8.4f} vs {rule.threshold:8.4f} | {rule.explanation}\n"

        if self.ml_confidence:
            output += f"\nML CONFIDENCE SCORE: {self.ml_confidence:.1%}\n"

        if self.risk_factors:
            output += f"\nRISK FACTORS:\n"
            for factor in self.risk_factors:
                output += f"  ⚠ {factor}\n"

        if self.supporting_factors:
            output += f"\nSUPPORTING FACTORS:\n"
            for factor in self.supporting_factors:
                output += f"  ✓ {factor}\n"

        output += f"\n{'=' * 70}\n"
        return output


class ShariaRulesEngine:
    """
    Deterministic Shariah compliance rules engine

    Rules (in order of evaluation):
    1. HARAM SECTORS → Automatic REJECT
    2. DEBT-TO-ASSETS ≥ 33% → Automatic REJECT
    3. NON-HALAL INCOME ≥ 25% → Automatic REJECT
    4. EDGE CASES → Flag for ML confidence scoring
    5. OTHERWISE → PASS (defer to ML if edge case)
    """

    # --- CONFIGURATION ---

    # Haram sectors (automatic rejection)
    HARAM_SECTORS = {
        "Alcohol",
        "Beverages",
        "Tobacco",
        "Gambling",
        "Casino",
        "Gaming",
        "Pork",
        "Pork Products",
        "Banking",
        "Conventional Banking",
        "Insurance",
        "Conventional Insurance",
        "Financial Services",
        "Credit Card",
        "Mortgage",
        "Weapons",
        "Defense",
        "Military",
        "Entertainment",
        "Cinema",
        "Music",
        "Adult Entertainment",
    }

    # Hard thresholds (automatic fail if exceeded)
    HARD_THRESHOLDS = {
        "debt_to_assets": 0.33,  # F_RIBA: Debt cannot exceed 33% of assets
        "nonhalal_income": 0.25,  # F_NONHALAL: Max 25% non-halal income
    }

    # Edge case thresholds (ML takes over)
    EDGE_CASE_THRESHOLDS = {
        "debt_to_assets": (0.30, 0.35),  # 30-35% = gray zone
        "nonhalal_income": (0.20, 0.25),  # 20-25% = gray zone
    }

    def __init__(self):
        """Initialize rules engine"""
        self.ml_model = None  # Will be set later if using ML confidence
        self.ml_feature_importance = {}

    def evaluate(self, row: pd.Series) -> ComplianceDecision:
        """
        Evaluate a company against Shariah rules

        Args:
            row: pandas Series with company financial data

        Returns:
            ComplianceDecision object with full evaluation
        """
        decision = ComplianceDecision(
            ticker=row["ticker"], company_name=row["company_name"]
        )

        # RULE 1: Check haram sector (automatic fail)
        sector_check = self._check_haram_sector(row)
        decision.rule_results.append(sector_check)

        if not sector_check.passed:
            decision.status = ComplianceStatus.NON_COMPLIANT
            decision.violations.append(RuleViolation.HARAM_SECTOR)
            decision.final_confidence = 1.0
            decision.risk_factors.append(f"Operates in haram sector: {row['sector']}")
            return decision  # Short-circuit: haram sector = automatic fail

        # RULE 2: Check debt-to-assets ratio (F_RIBA proxy)
        debt_check = self._check_debt_ratio(row)
        decision.rule_results.append(debt_check)

        if not debt_check.passed:
            decision.status = ComplianceStatus.NON_COMPLIANT
            decision.violations.append(RuleViolation.EXCESSIVE_DEBT)
            decision.final_confidence = 1.0
            decision.risk_factors.append(
                f"Excessive debt exposure: {debt_check.value:.1%} > {debt_check.threshold:.1%}"
            )
            return decision  # Short-circuit: hard fail

        # RULE 3: Check non-halal income ratio (F_NONHALAL)
        nonhalal_check = self._check_nonhalal_income(row)
        decision.rule_results.append(nonhalal_check)

        if not nonhalal_check.passed:
            decision.status = ComplianceStatus.NON_COMPLIANT
            decision.violations.append(RuleViolation.EXCESSIVE_NONHALAL)
            decision.final_confidence = 1.0
            decision.risk_factors.append(
                f"Excessive non-halal income: {nonhalal_check.value:.1%} > {nonhalal_check.threshold:.1%}"
            )
            return decision  # Short-circuit: hard fail

        # RULE 4: Check for edge cases (borderline thresholds)
        edge_case_result = self._detect_edge_case(row)

        if edge_case_result["is_edge_case"]:
            decision.status = ComplianceStatus.EDGE_CASE
            decision.violations.append(RuleViolation.AMBIGUOUS)
            decision.risk_factors.extend(edge_case_result["factors"])
            # ML will provide confidence score
            return decision

        # RULE 5: All checks passed → COMPLIANT
        decision.status = ComplianceStatus.COMPLIANT
        decision.final_confidence = 1.0
        decision.supporting_factors.append("Passes all hard Shariah rules")
        decision.supporting_factors.append(
            f"Debt ratio: {debt_check.value:.1%} < {debt_check.threshold:.1%}"
        )
        decision.supporting_factors.append(
            f"Non-halal income: {nonhalal_check.value:.1%} < {nonhalal_check.threshold:.1%}"
        )

        return decision

    # --- INDIVIDUAL RULE CHECKS ---

    def _check_haram_sector(self, row: pd.Series) -> RuleCheckResult:
        """Check if company operates in haram sector"""
        sector = str(row["sector"]).strip()
        is_haram = any(haram in sector for haram in self.HARAM_SECTORS)

        return RuleCheckResult(
            rule_name="Haram Sector",
            passed=not is_haram,
            value=1.0 if is_haram else 0.0,
            threshold=0.0,
            severity="CRITICAL",
            explanation=f"Sector: {sector}"
            + (f" (HARAM)" if is_haram else f" (Permissible)"),
        )

    def _check_debt_ratio(self, row: pd.Series) -> RuleCheckResult:
        """Check debt-to-assets ratio (F_RIBA proxy)"""
        debt_ratio = row["total_liabilities"] / row["total_assets"]
        threshold = self.HARD_THRESHOLDS["debt_to_assets"]
        passed = debt_ratio < threshold

        return RuleCheckResult(
            rule_name="F_RIBA (Debt-to-Assets)",
            passed=passed,
            value=debt_ratio,
            threshold=threshold,
            severity="CRITICAL",
            explanation=f"Debt exposure {debt_ratio:.1%} {'<' if passed else '≥'} {threshold:.0%}",
        )

    def _check_nonhalal_income(self, row: pd.Series) -> RuleCheckResult:
        """Check non-halal income ratio (F_NONHALAL)"""
        nonhalal_ratio = row["nonhalal_revenue_percent"]
        threshold = self.HARD_THRESHOLDS["nonhalal_income"]
        passed = nonhalal_ratio < threshold

        return RuleCheckResult(
            rule_name="F_NONHALAL (Income Purity)",
            passed=passed,
            value=nonhalal_ratio,
            threshold=threshold,
            severity="CRITICAL",
            explanation=f"Non-halal income {nonhalal_ratio:.1%} {'<' if passed else '≥'} {threshold:.0%}",
        )

    def _detect_edge_case(self, row: pd.Series) -> Dict:
        """Detect edge cases (borderline thresholds)"""
        edge_factors = []
        is_edge_case = False

        # Check debt ratio edge case
        debt_ratio = row["total_liabilities"] / row["total_assets"]
        debt_lower, debt_upper = self.EDGE_CASE_THRESHOLDS["debt_to_assets"]
        if debt_lower <= debt_ratio < self.HARD_THRESHOLDS["debt_to_assets"]:
            edge_factors.append(
                f"Borderline debt ratio: {debt_ratio:.1%} (warning zone: {debt_lower:.0%}-{debt_upper:.0%})"
            )
            is_edge_case = True

        # Check nonhalal income edge case
        nonhalal_ratio = row["nonhalal_revenue_percent"]
        nonhalal_lower, nonhalal_upper = self.EDGE_CASE_THRESHOLDS["nonhalal_income"]
        if nonhalal_lower <= nonhalal_ratio < self.HARD_THRESHOLDS["nonhalal_income"]:
            edge_factors.append(
                f"Borderline non-halal income: {nonhalal_ratio:.1%} (warning zone: {nonhalal_lower:.0%}-{nonhalal_upper:.0%})"
            )
            is_edge_case = True

        # Check for missing/noisy data
        if pd.isna(row).any() or (row["total_assets"] == 0):
            edge_factors.append("Incomplete or noisy financial data")
            is_edge_case = True

        return {"is_edge_case": is_edge_case, "factors": edge_factors}

    # --- ML INTEGRATION ---

    def set_ml_model(self, model, feature_importance: Dict = None):
        """
        Attach ML model for confidence scoring on edge cases

        Args:
            model: Trained XGBoost model
            feature_importance: Dict of feature importance scores
        """
        self.ml_model = model
        self.ml_feature_importance = feature_importance or {}

    def score_with_ml_confidence(
        self,
        decision: ComplianceDecision,
        features: np.ndarray,
        feature_names: List[str],
    ) -> ComplianceDecision:
        """
        Add ML confidence score to edge case decisions

        Args:
            decision: ComplianceDecision from rule engine
            features: Feature vector for ML model
            feature_names: List of feature names for importance ranking

        Returns:
            Updated ComplianceDecision with ML scores
        """
        if self.ml_model is None or decision.status != ComplianceStatus.EDGE_CASE:
            return decision

        # Get ML prediction
        ml_proba = self.ml_model.predict_proba([features])[0]
        ml_score_compliant = ml_proba[1]  # Probability of class 1 (compliant)

        decision.ml_score = ml_score_compliant
        decision.ml_confidence = ml_score_compliant

        # Update final confidence based on rules + ML
        # Edge case starts at 50%, adjusted by ML confidence
        decision.final_confidence = 0.5 + (ml_score_compliant * 0.5)

        # Add supporting/risk factors based on feature importance
        if hasattr(self.ml_model, "feature_importances_"):
            importances = self.ml_model.feature_importances_
            top_indices = np.argsort(importances)[-3:][::-1]

            for idx in top_indices:
                if idx < len(feature_names):
                    factor = feature_names[idx]
                    importance = importances[idx]
                    if importance > 0.05:  # Only include meaningful features
                        if ml_score_compliant > 0.5:
                            decision.supporting_factors.append(
                                f"ML: {factor} (importance: {importance:.1%})"
                            )
                        else:
                            decision.risk_factors.append(
                                f"ML: {factor} (importance: {importance:.1%})"
                            )

        return decision

    def evaluate_batch(self, df: pd.DataFrame) -> List[ComplianceDecision]:
        """
        Evaluate multiple companies

        Args:
            df: DataFrame with company records

        Returns:
            List of ComplianceDecision objects
        """
        decisions = []
        for idx, row in df.iterrows():
            decision = self.evaluate(row)
            decisions.append(decision)
        return decisions


# --- HELPER FUNCTIONS ---


def print_compliance_report(decisions: List[ComplianceDecision]):
    """Print formatted compliance report for batch evaluation"""
    compliant_count = sum(
        1 for d in decisions if d.status == ComplianceStatus.COMPLIANT
    )
    non_compliant_count = sum(
        1 for d in decisions if d.status == ComplianceStatus.NON_COMPLIANT
    )
    edge_case_count = sum(
        1 for d in decisions if d.status == ComplianceStatus.EDGE_CASE
    )

    print("\n" + "=" * 80)
    print("SHARIAH COMPLIANCE BATCH EVALUATION REPORT")
    print("=" * 80)
    print(f"\nSummary:")
    print(
        f"  ✓ Compliant:        {compliant_count:4d} ({compliant_count / len(decisions) * 100:5.1f}%)"
    )
    print(
        f"  ✗ Non-Compliant:    {non_compliant_count:4d} ({non_compliant_count / len(decisions) * 100:5.1f}%)"
    )
    print(
        f"  ⚠ Edge Cases:       {edge_case_count:4d} ({edge_case_count / len(decisions) * 100:5.1f}%)"
    )
    print(f"  ─────────────────────────────────")
    print(f"  Total Evaluated:    {len(decisions):4d}")

    print("\n" + "=" * 80)
    for decision in decisions:
        if decision.status != ComplianceStatus.COMPLIANT:  # Show failures + edge cases
            print(decision)


def to_dataframe(decisions: List[ComplianceDecision]) -> pd.DataFrame:
    """Convert decisions to DataFrame for analysis"""
    records = []
    for d in decisions:
        record = {
            "ticker": d.ticker,
            "company_name": d.company_name,
            "status": d.status.value,
            "final_confidence": d.final_confidence,
            "ml_score": d.ml_score,
            "violations": "; ".join([v.value for v in d.violations])
            if d.violations
            else "None",
        }
        records.append(record)
    return pd.DataFrame(records)
