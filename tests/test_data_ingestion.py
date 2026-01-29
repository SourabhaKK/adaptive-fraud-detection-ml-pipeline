"""
Test suite for data ingestion and deterministic data splitting.

These tests enforce that:
1. Data splitting is deterministic and reproducible
2. Time-aware splitting prevents temporal leakage
3. Random seeds are explicitly controlled
4. Same input always produces same train/validation split
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestDeterministicDataSplitting:
    """Test suite for deterministic and time-aware data splitting."""

    @pytest.fixture
    def sample_fraud_data(self):
        """
        Create deterministic sample fraud detection data with timestamps.
        
        Returns:
            pd.DataFrame: Sample data with transaction_date, features, and fraud labels
        """
        np.random.seed(42)
        
        # Create 1000 transactions over 100 days
        n_samples = 1000
        base_date = datetime(2024, 1, 1)
        
        data = {
            'transaction_id': range(n_samples),
            'transaction_date': [
                base_date + timedelta(days=int(i / 10)) 
                for i in range(n_samples)
            ],
            'amount': np.random.uniform(10, 1000, n_samples),
            'merchant_category': np.random.choice(['retail', 'online', 'travel'], n_samples),
            'is_fraud': np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        }
        
        return pd.DataFrame(data)

    def test_same_input_produces_same_split(self, sample_fraud_data):
        """
        Test that same input dataset produces identical train/val splits.
        
        Determinism scenario enforced:
        - Running split twice on same data with same seed must produce
          identical train/val indices.
        
        Expected behavior:
        - train_indices_1 == train_indices_2
        - val_indices_1 == val_indices_2
        """
        from src.data_ingestion.splitter import DeterministicSplitter
        
        splitter = DeterministicSplitter(random_seed=42, val_size=0.2)
        
        # First split
        train_1, val_1 = splitter.split(sample_fraud_data)
        train_indices_1 = train_1.index.tolist()
        val_indices_1 = val_1.index.tolist()
        
        # Second split with same seed
        splitter_2 = DeterministicSplitter(random_seed=42, val_size=0.2)
        train_2, val_2 = splitter_2.split(sample_fraud_data)
        train_indices_2 = train_2.index.tolist()
        val_indices_2 = val_2.index.tolist()
        
        # Verify identical splits
        assert train_indices_1 == train_indices_2, \
            "Same seed should produce identical training indices"
        assert val_indices_1 == val_indices_2, \
            "Same seed should produce identical validation indices"
        
        # Verify no overlap
        assert len(set(train_indices_1) & set(val_indices_1)) == 0, \
            "Train and validation sets should not overlap"

    def test_different_seeds_produce_different_splits(self, sample_fraud_data):
        """
        Test that different random seeds produce different splits.
        
        Determinism scenario enforced:
        - Different seeds should produce different (but valid) splits.
        
        Expected behavior:
        - train_indices_1 != train_indices_2
        - Both splits should be valid (no overlap, correct sizes)
        """
        from src.data_ingestion.splitter import DeterministicSplitter
        
        splitter_1 = DeterministicSplitter(random_seed=42, val_size=0.2)
        splitter_2 = DeterministicSplitter(random_seed=123, val_size=0.2)
        
        train_1, val_1 = splitter_1.split(sample_fraud_data)
        train_2, val_2 = splitter_2.split(sample_fraud_data)
        
        # Different seeds should produce different splits
        assert train_1.index.tolist() != train_2.index.tolist(), \
            "Different seeds should produce different training sets"
        
        # But both should have correct sizes
        assert len(train_1) == len(train_2), \
            "Both splits should have same training set size"
        assert len(val_1) == len(val_2), \
            "Both splits should have same validation set size"

    def test_time_aware_split_prevents_future_leakage(self, sample_fraud_data):
        """
        Test that time-aware splitting prevents future information leakage.
        
        Temporal leakage scenario prevented:
        - Validation data must not contain transactions that occurred
          BEFORE training data transactions.
        - This simulates real-world deployment where we train on past
          data and validate on future data.
        
        Expected behavior:
        - max(train_dates) <= min(val_dates)
        - Chronological ordering is maintained
        """
        from src.data_ingestion.splitter import TimeAwareSplitter
        
        splitter = TimeAwareSplitter(
            time_column='transaction_date',
            val_size=0.2
        )
        
        train, val = splitter.split(sample_fraud_data)
        
        # Get date ranges
        max_train_date = train['transaction_date'].max()
        min_val_date = val['transaction_date'].min()
        
        # Validation should be chronologically after training
        assert max_train_date <= min_val_date, \
            f"Temporal leakage detected: validation data ({min_val_date}) " \
            f"contains dates before training data ends ({max_train_date})"

    def test_time_aware_split_is_deterministic(self, sample_fraud_data):
        """
        Test that time-aware splitting is deterministic.
        
        Determinism scenario enforced:
        - Time-based splits should be deterministic (no randomness).
        - Same input should always produce same split.
        
        Expected behavior:
        - Multiple runs produce identical splits
        - No random seed needed (deterministic by time)
        """
        from src.data_ingestion.splitter import TimeAwareSplitter
        
        splitter_1 = TimeAwareSplitter(
            time_column='transaction_date',
            val_size=0.2
        )
        splitter_2 = TimeAwareSplitter(
            time_column='transaction_date',
            val_size=0.2
        )
        
        train_1, val_1 = splitter_1.split(sample_fraud_data)
        train_2, val_2 = splitter_2.split(sample_fraud_data)
        
        # Time-based splits should be identical
        assert train_1.index.tolist() == train_2.index.tolist(), \
            "Time-aware splits should be deterministic"
        assert val_1.index.tolist() == val_2.index.tolist(), \
            "Time-aware splits should be deterministic"

    def test_stratified_split_maintains_class_distribution(self, sample_fraud_data):
        """
        Test that stratified splitting maintains fraud class distribution.
        
        Determinism scenario enforced:
        - Stratified split should maintain fraud ratio in train/val sets.
        - Must be deterministic with fixed random seed.
        
        Expected behavior:
        - train_fraud_ratio ≈ val_fraud_ratio ≈ overall_fraud_ratio
        - Same seed produces same stratified split
        """
        from src.data_ingestion.splitter import StratifiedSplitter
        
        splitter = StratifiedSplitter(
            target_column='is_fraud',
            random_seed=42,
            val_size=0.2
        )
        
        train, val = splitter.split(sample_fraud_data)
        
        # Calculate fraud ratios
        overall_fraud_ratio = sample_fraud_data['is_fraud'].mean()
        train_fraud_ratio = train['is_fraud'].mean()
        val_fraud_ratio = val['is_fraud'].mean()
        
        # Ratios should be similar (within 5% tolerance)
        tolerance = 0.05
        assert abs(train_fraud_ratio - overall_fraud_ratio) < tolerance, \
            f"Training set fraud ratio ({train_fraud_ratio:.3f}) differs from " \
            f"overall ratio ({overall_fraud_ratio:.3f})"
        assert abs(val_fraud_ratio - overall_fraud_ratio) < tolerance, \
            f"Validation set fraud ratio ({val_fraud_ratio:.3f}) differs from " \
            f"overall ratio ({overall_fraud_ratio:.3f})"

    def test_split_sizes_are_correct(self, sample_fraud_data):
        """
        Test that split sizes match specified validation size.
        
        Determinism scenario enforced:
        - Validation size should be exactly as specified.
        - Total samples should equal original dataset size.
        
        Expected behavior:
        - len(val) ≈ val_size * len(data)
        - len(train) + len(val) == len(data)
        """
        from src.data_ingestion.splitter import DeterministicSplitter
        
        val_size = 0.2
        splitter = DeterministicSplitter(random_seed=42, val_size=val_size)
        
        train, val = splitter.split(sample_fraud_data)
        
        # Check sizes
        total_size = len(sample_fraud_data)
        expected_val_size = int(total_size * val_size)
        
        assert len(train) + len(val) == total_size, \
            "Train + validation should equal total dataset size"
        
        # Allow ±1 sample tolerance due to rounding
        assert abs(len(val) - expected_val_size) <= 1, \
            f"Validation size ({len(val)}) should be approximately " \
            f"{val_size * 100}% of total ({expected_val_size})"

    def test_split_without_random_seed_raises_warning(self, sample_fraud_data):
        """
        Test that splitting without explicit random seed raises warning.
        
        Determinism scenario enforced:
        - Random splits MUST have explicit seed for reproducibility.
        - Missing seed should raise warning or error.
        
        Expected behavior:
        - Creating splitter without seed raises ValueError or Warning
        """
        from src.data_ingestion.splitter import DeterministicSplitter
        
        # Attempting to create splitter without seed should fail
        with pytest.raises(ValueError, match=".*random_seed.*required.*"):
            splitter = DeterministicSplitter(val_size=0.2)  # No seed!

    def test_time_column_validation(self, sample_fraud_data):
        """
        Test that time-aware splitter validates time column exists.
        
        Determinism scenario enforced:
        - Time column must exist in dataset.
        - Invalid column should raise clear error.
        
        Expected behavior:
        - Missing time column raises ValueError
        """
        from src.data_ingestion.splitter import TimeAwareSplitter
        
        splitter = TimeAwareSplitter(
            time_column='nonexistent_column',
            val_size=0.2
        )
        
        with pytest.raises(ValueError, match=".*time_column.*not found.*"):
            splitter.split(sample_fraud_data)
