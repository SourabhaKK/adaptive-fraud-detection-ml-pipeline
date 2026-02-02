"""
Test suite for fraud decision threshold logic.

These tests enforce that:
1. Decision thresholds are configurable
2. Low-risk transactions are auto-approved
3. Medium-risk transactions are flagged for review
4. High-risk transactions are blocked
5. Invalid configurations raise clear errors
"""

import pytest
import numpy as np
import pandas as pd


class TestFraudDecisionThresholds:
    """Test suite for fraud decision threshold logic."""

    @pytest.fixture
    def sample_predictions(self):
        """
        Create sample fraud probability predictions.
        
        Returns:
            np.ndarray: Array of fraud probabilities [0.0, 1.0]
        """
        return np.array([
            0.01,  # Very low risk
            0.05,  # Low risk
            0.15,  # Low-medium risk
            0.35,  # Medium risk
            0.55,  # Medium-high risk
            0.75,  # High risk
            0.95,  # Very high risk
        ])

    def test_low_risk_auto_approval(self, sample_predictions):
        """
        Test that low-risk transactions are auto-approved.
        
        Business logic:
        - Fraud probability < low_threshold → AUTO_APPROVE
        - These transactions proceed without manual review
        
        Expected behavior:
        - Predictions [0.01, 0.05] → AUTO_APPROVE (threshold=0.1)
        """
        from src.decision_logic.decision_maker import FraudDecisionMaker
        
        decision_maker = FraudDecisionMaker(
            low_threshold=0.1,
            high_threshold=0.5
        )
        
        # Low-risk predictions
        low_risk_probs = np.array([0.01, 0.05, 0.08])
        decisions = decision_maker.make_decisions(low_risk_probs)
        
        # All should be auto-approved
        assert all(d == 'AUTO_APPROVE' for d in decisions), \
            "All low-risk transactions should be auto-approved"

    def test_medium_risk_flagged_for_review(self, sample_predictions):
        """
        Test that medium-risk transactions are flagged for review.
        
        Business logic:
        - low_threshold <= fraud_prob < high_threshold → FLAG_FOR_REVIEW
        - These transactions require manual investigation
        
        Expected behavior:
        - Predictions [0.15, 0.35] → FLAG_FOR_REVIEW (thresholds: 0.1, 0.5)
        """
        from src.decision_logic.decision_maker import FraudDecisionMaker
        
        decision_maker = FraudDecisionMaker(
            low_threshold=0.1,
            high_threshold=0.5
        )
        
        # Medium-risk predictions
        medium_risk_probs = np.array([0.15, 0.25, 0.35, 0.45])
        decisions = decision_maker.make_decisions(medium_risk_probs)
        
        # All should be flagged for review
        assert all(d == 'FLAG_FOR_REVIEW' for d in decisions), \
            "All medium-risk transactions should be flagged for review"

    def test_high_risk_blocked(self, sample_predictions):
        """
        Test that high-risk transactions are blocked.
        
        Business logic:
        - fraud_prob >= high_threshold → BLOCK
        - These transactions are immediately declined
        
        Expected behavior:
        - Predictions [0.75, 0.95] → BLOCK (threshold=0.5)
        """
        from src.decision_logic.decision_maker import FraudDecisionMaker
        
        decision_maker = FraudDecisionMaker(
            low_threshold=0.1,
            high_threshold=0.5
        )
        
        # High-risk predictions
        high_risk_probs = np.array([0.55, 0.75, 0.85, 0.95])
        decisions = decision_maker.make_decisions(high_risk_probs)
        
        # All should be blocked
        assert all(d == 'BLOCK' for d in decisions), \
            "All high-risk transactions should be blocked"

    def test_threshold_boundary_conditions(self):
        """
        Test exact threshold boundary conditions.
        
        Business logic:
        - prob < low_threshold → AUTO_APPROVE
        - prob == low_threshold → FLAG_FOR_REVIEW (inclusive)
        - prob == high_threshold → BLOCK (inclusive)
        
        Expected behavior:
        - 0.099 → AUTO_APPROVE
        - 0.100 → FLAG_FOR_REVIEW (exactly at low threshold)
        - 0.499 → FLAG_FOR_REVIEW
        - 0.500 → BLOCK (exactly at high threshold)
        """
        from src.decision_logic.decision_maker import FraudDecisionMaker
        
        decision_maker = FraudDecisionMaker(
            low_threshold=0.1,
            high_threshold=0.5
        )
        
        boundary_probs = np.array([0.099, 0.100, 0.499, 0.500])
        decisions = decision_maker.make_decisions(boundary_probs)
        
        expected = ['AUTO_APPROVE', 'FLAG_FOR_REVIEW', 'FLAG_FOR_REVIEW', 'BLOCK']
        
        assert decisions.tolist() == expected, \
            f"Boundary conditions failed. Got {decisions.tolist()}, expected {expected}"

    def test_configurable_thresholds(self):
        """
        Test that thresholds are configurable.
        
        Business logic:
        - Different threshold configurations should produce different decisions
        - Same prediction, different thresholds → different decisions
        
        Expected behavior:
        - prob=0.3 with thresholds (0.2, 0.4) → FLAG_FOR_REVIEW
        - prob=0.3 with thresholds (0.4, 0.6) → AUTO_APPROVE
        - prob=0.3 with thresholds (0.1, 0.2) → BLOCK
        """
        from src.decision_logic.decision_maker import FraudDecisionMaker
        
        test_prob = np.array([0.3])
        
        # Configuration 1: Medium risk
        dm1 = FraudDecisionMaker(low_threshold=0.2, high_threshold=0.4)
        decision1 = dm1.make_decisions(test_prob)[0]
        assert decision1 == 'FLAG_FOR_REVIEW', \
            "0.3 should be flagged with thresholds (0.2, 0.4)"
        
        # Configuration 2: Low risk
        dm2 = FraudDecisionMaker(low_threshold=0.4, high_threshold=0.6)
        decision2 = dm2.make_decisions(test_prob)[0]
        assert decision2 == 'AUTO_APPROVE', \
            "0.3 should be auto-approved with thresholds (0.4, 0.6)"
        
        # Configuration 3: High risk
        dm3 = FraudDecisionMaker(low_threshold=0.1, high_threshold=0.2)
        decision3 = dm3.make_decisions(test_prob)[0]
        assert decision3 == 'BLOCK', \
            "0.3 should be blocked with thresholds (0.1, 0.2)"

    def test_invalid_threshold_order_raises_error(self):
        """
        Test that invalid threshold ordering raises error.
        
        Business logic:
        - low_threshold must be < high_threshold
        - Invalid: low=0.5, high=0.3 (low > high)
        
        Expected behavior:
        - Raises ValueError with clear message
        """
        from src.decision_logic.decision_maker import FraudDecisionMaker
        
        with pytest.raises(ValueError, match=".*low_threshold.*high_threshold.*"):
            FraudDecisionMaker(low_threshold=0.5, high_threshold=0.3)

    def test_threshold_out_of_range_raises_error(self):
        """
        Test that thresholds outside [0, 1] raise error.
        
        Business logic:
        - Thresholds must be valid probabilities: 0 <= threshold <= 1
        - Invalid: threshold = 1.5 or threshold = -0.1
        
        Expected behavior:
        - Raises ValueError for out-of-range thresholds
        """
        from src.decision_logic.decision_maker import FraudDecisionMaker
        
        # Threshold > 1
        with pytest.raises(ValueError, match=".*threshold.*range.*|.*0.*1.*"):
            FraudDecisionMaker(low_threshold=0.1, high_threshold=1.5)
        
        # Threshold < 0
        with pytest.raises(ValueError, match=".*threshold.*range.*|.*0.*1.*"):
            FraudDecisionMaker(low_threshold=-0.1, high_threshold=0.5)

    def test_batch_predictions(self):
        """
        Test decision making on batch of predictions.
        
        Business logic:
        - Should handle arrays of predictions efficiently
        - Output array length should match input length
        
        Expected behavior:
        - Input: 1000 predictions
        - Output: 1000 decisions
        - Decisions match individual threshold logic
        """
        from src.decision_logic.decision_maker import FraudDecisionMaker
        
        decision_maker = FraudDecisionMaker(
            low_threshold=0.2,
            high_threshold=0.7
        )
        
        # Generate 1000 random predictions
        np.random.seed(42)
        predictions = np.random.uniform(0, 1, size=1000)
        
        decisions = decision_maker.make_decisions(predictions)
        
        # Check output length
        assert len(decisions) == len(predictions), \
            "Output length should match input length"
        
        # Verify decision logic
        auto_approve_count = np.sum(decisions == 'AUTO_APPROVE')
        flag_count = np.sum(decisions == 'FLAG_FOR_REVIEW')
        block_count = np.sum(decisions == 'BLOCK')
        
        # All predictions should be classified
        assert auto_approve_count + flag_count + block_count == len(predictions), \
            "All predictions should be classified into one of three categories"

    def test_decision_with_dataframe(self):
        """
        Test decision making with pandas DataFrame input.
        
        Business logic:
        - Should accept DataFrame with fraud_probability column
        - Should return DataFrame with decision column added
        
        Expected behavior:
        - Input: DataFrame with fraud_probability
        - Output: DataFrame with decision column
        """
        from src.decision_logic.decision_maker import FraudDecisionMaker
        
        decision_maker = FraudDecisionMaker(
            low_threshold=0.2,
            high_threshold=0.6
        )
        
        # Create sample DataFrame
        df = pd.DataFrame({
            'transaction_id': [1, 2, 3, 4],
            'fraud_probability': [0.1, 0.3, 0.7, 0.95]
        })
        
        result = decision_maker.make_decisions_df(df, prob_column='fraud_probability')
        
        # Check decision column exists
        assert 'decision' in result.columns, \
            "Result should have 'decision' column"
        
        # Verify decisions
        expected_decisions = ['AUTO_APPROVE', 'FLAG_FOR_REVIEW', 'BLOCK', 'BLOCK']
        assert result['decision'].tolist() == expected_decisions, \
            f"Decisions don't match. Got {result['decision'].tolist()}"

    def test_decision_statistics(self):
        """
        Test that decision maker can provide statistics.
        
        Business logic:
        - Should track decision counts and percentages
        - Useful for monitoring and reporting
        
        Expected behavior:
        - Returns dict with counts and percentages
        """
        from src.decision_logic.decision_maker import FraudDecisionMaker
        
        decision_maker = FraudDecisionMaker(
            low_threshold=0.3,
            high_threshold=0.7
        )
        
        # 10 predictions: 3 low, 4 medium, 3 high
        predictions = np.array([0.1, 0.15, 0.2,  # Low
                               0.4, 0.5, 0.55, 0.6,  # Medium
                               0.8, 0.9, 0.95])  # High
        
        decisions = decision_maker.make_decisions(predictions)
        stats = decision_maker.get_decision_stats(decisions)
        
        # Check statistics
        assert stats['AUTO_APPROVE']['count'] == 3, "Should have 3 auto-approvals"
        assert stats['FLAG_FOR_REVIEW']['count'] == 4, "Should have 4 flagged"
        assert stats['BLOCK']['count'] == 3, "Should have 3 blocked"
        
        assert stats['AUTO_APPROVE']['percentage'] == 30.0, "Should be 30%"
        assert stats['FLAG_FOR_REVIEW']['percentage'] == 40.0, "Should be 40%"
        assert stats['BLOCK']['percentage'] == 30.0, "Should be 30%"
