"""
Fraud decision threshold logic.

This module provides decision-making logic that converts fraud probability
predictions into business actions:
- AUTO_APPROVE: Low-risk transactions
- FLAG_FOR_REVIEW: Medium-risk transactions requiring manual review
- BLOCK: High-risk transactions

Key principles:
1. Clear separation between ML prediction and business decisioning
2. Configurable thresholds for business flexibility
3. Deterministic behavior
"""

import numpy as np
import pandas as pd
from typing import Union, Dict


class FraudDecisionMaker:
    """
    Fraud decision maker with configurable thresholds.
    
    Converts fraud probability predictions into business decisions:
    - prob < low_threshold → AUTO_APPROVE
    - low_threshold <= prob < high_threshold → FLAG_FOR_REVIEW
    - prob >= high_threshold → BLOCK
    
    Attributes:
        low_threshold (float): Threshold for auto-approval (0.0 to 1.0)
        high_threshold (float): Threshold for blocking (0.0 to 1.0)
    """
    
    def __init__(self, low_threshold: float, high_threshold: float):
        """
        Initialize fraud decision maker.
        
        Args:
            low_threshold (float): Threshold below which transactions are auto-approved
            high_threshold (float): Threshold above which transactions are blocked
            
        Raises:
            ValueError: If thresholds are invalid
        """
        # Validate threshold range
        if not (0 <= low_threshold <= 1):
            raise ValueError(
                f"low_threshold must be in range [0, 1], got {low_threshold}"
            )
        if not (0 <= high_threshold <= 1):
            raise ValueError(
                f"high_threshold must be in range [0, 1], got {high_threshold}"
            )
        
        # Validate threshold ordering
        if low_threshold >= high_threshold:
            raise ValueError(
                f"low_threshold ({low_threshold}) must be less than "
                f"high_threshold ({high_threshold})"
            )
        
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def make_decisions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Make decisions based on fraud probability predictions.
        
        Args:
            predictions (np.ndarray): Array of fraud probabilities [0.0, 1.0]
            
        Returns:
            np.ndarray: Array of decisions ('AUTO_APPROVE', 'FLAG_FOR_REVIEW', 'BLOCK')
        """
        predictions = np.asarray(predictions)
        decisions = np.empty(predictions.shape, dtype=object)
        
        # Apply decision logic
        # Low risk: prob < low_threshold
        decisions[predictions < self.low_threshold] = 'AUTO_APPROVE'
        
        # Medium risk: low_threshold <= prob < high_threshold
        medium_risk_mask = (predictions >= self.low_threshold) & (predictions < self.high_threshold)
        decisions[medium_risk_mask] = 'FLAG_FOR_REVIEW'
        
        # High risk: prob >= high_threshold
        decisions[predictions >= self.high_threshold] = 'BLOCK'
        
        return decisions
    
    def make_decisions_df(self, df: pd.DataFrame, prob_column: str = 'fraud_probability') -> pd.DataFrame:
        """
        Make decisions on DataFrame with fraud probabilities.
        
        Args:
            df (pd.DataFrame): DataFrame containing fraud probabilities
            prob_column (str): Name of column containing fraud probabilities
            
        Returns:
            pd.DataFrame: Original DataFrame with 'decision' column added
            
        Raises:
            ValueError: If prob_column not found in DataFrame
        """
        if prob_column not in df.columns:
            raise ValueError(
                f"Column '{prob_column}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Make decisions
        decisions = self.make_decisions(df[prob_column].values)
        
        # Add decision column to DataFrame
        result = df.copy()
        result['decision'] = decisions
        
        return result
    
    def get_decision_stats(self, decisions: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for decisions.
        
        Args:
            decisions (np.ndarray): Array of decisions
            
        Returns:
            Dict[str, Dict[str, float]]: Statistics with counts and percentages
        """
        decisions = np.asarray(decisions)
        total = len(decisions)
        
        stats = {}
        for decision_type in ['AUTO_APPROVE', 'FLAG_FOR_REVIEW', 'BLOCK']:
            count = np.sum(decisions == decision_type)
            percentage = (count / total * 100) if total > 0 else 0.0
            
            stats[decision_type] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        return stats
