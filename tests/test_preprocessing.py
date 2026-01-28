"""
Test suite for leakage-safe preprocessing.

These tests enforce that preprocessing transformers:
1. Are fitted ONLY on training data
2. Do not allow validation/test data to influence fitted parameters
3. Explicitly prevent accidental leakage scenarios

All tests are deterministic and focus on fit/transform separation.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class TestLeakageSafePreprocessing:
    """Test suite to prevent data leakage in preprocessing."""

    @pytest.fixture
    def sample_data(self):
        """Create deterministic sample data for testing.
        
        Returns:
            tuple: (X_train, X_val, y_train, y_val) with known statistics
        """
        np.random.seed(42)
        
        # Create data with distinct train/val statistics
        # Train: mean=10, std=2
        X_train = np.random.normal(loc=10, scale=2, size=(100, 3))
        y_train = np.random.randint(0, 2, size=100)
        
        # Validation: mean=20, std=5 (deliberately different)
        X_val = np.random.normal(loc=20, scale=5, size=(30, 3))
        y_val = np.random.randint(0, 2, size=30)
        
        return X_train, X_val, y_train, y_val

    def test_scaler_fitted_only_on_training_data(self, sample_data):
        """
        Test that StandardScaler is fitted ONLY on training data.
        
        Leakage scenario prevented:
        - Fitting scaler on combined train+val data would leak validation
          statistics (mean, std) into the training process.
        
        Expected behavior:
        - Scaler's mean_ and scale_ should match training data statistics
        - Should NOT match combined dataset statistics
        """
        from src.preprocessing.transformers import LeakageSafeScaler
        
        X_train, X_val, y_train, y_val = sample_data
        
        # Initialize and fit scaler
        scaler = LeakageSafeScaler()
        scaler.fit(X_train, y_train)
        
        # Verify scaler was fitted on training data only
        # Training data has mean ≈ 10, std ≈ 2
        expected_mean = X_train.mean(axis=0)
        expected_std = X_train.std(axis=0)
        
        np.testing.assert_array_almost_equal(
            scaler.mean_, expected_mean, decimal=2,
            err_msg="Scaler mean should match training data only"
        )
        
        np.testing.assert_array_almost_equal(
            scaler.scale_, expected_std, decimal=2,
            err_msg="Scaler std should match training data only"
        )
        
        # Verify it does NOT match combined statistics
        X_combined = np.vstack([X_train, X_val])
        combined_mean = X_combined.mean(axis=0)
        
        # This should fail if leakage occurred
        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(
                scaler.mean_, combined_mean, decimal=2
            )

    def test_validation_data_does_not_influence_fitted_parameters(self, sample_data):
        """
        Test that validation data does not influence fitted parameters.
        
        Leakage scenario prevented:
        - Passing validation data during fit() should not change the
          fitted parameters.
        
        Expected behavior:
        - Scaler fitted with fit(X_train) should have identical parameters
          to scaler fitted with fit(X_train, validation_data=X_val)
        """
        from src.preprocessing.transformers import LeakageSafeScaler
        
        X_train, X_val, y_train, y_val = sample_data
        
        # Fit scaler without validation data
        scaler_no_val = LeakageSafeScaler()
        scaler_no_val.fit(X_train, y_train)
        
        # Fit scaler with validation data (should be ignored)
        scaler_with_val = LeakageSafeScaler()
        scaler_with_val.fit(X_train, y_train, validation_data=(X_val, y_val))
        
        # Parameters should be identical
        np.testing.assert_array_equal(
            scaler_no_val.mean_, scaler_with_val.mean_,
            err_msg="Validation data should not influence fitted mean"
        )
        
        np.testing.assert_array_equal(
            scaler_no_val.scale_, scaler_with_val.scale_,
            err_msg="Validation data should not influence fitted scale"
        )

    def test_fitting_on_full_dataset_raises_error(self, sample_data):
        """
        Test that attempting to fit on full dataset raises explicit error.
        
        Leakage scenario prevented:
        - Accidentally fitting on combined train+val+test data is a common
          mistake that causes severe leakage.
        
        Expected behavior:
        - If user attempts to fit on data that looks like it contains
          validation/test splits, raise a clear error.
        - This is enforced by requiring explicit train/val split metadata.
        """
        from src.preprocessing.transformers import LeakageSafeScaler
        
        X_train, X_val, y_train, y_val = sample_data
        
        # Combine datasets (common mistake)
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])
        
        scaler = LeakageSafeScaler()
        
        # Attempting to fit on combined data without explicit split should fail
        with pytest.raises(ValueError, match=".*leakage.*|.*split.*|.*train.*"):
            scaler.fit(X_combined, y_combined, enforce_split_validation=True)

    def test_transform_without_fit_raises_error(self, sample_data):
        """
        Test that transform() without prior fit() raises error.
        
        Leakage scenario prevented:
        - Ensures proper fit/transform workflow is followed.
        
        Expected behavior:
        - Calling transform() before fit() should raise NotFittedError.
        """
        from src.preprocessing.transformers import LeakageSafeScaler
        
        X_train, X_val, y_train, y_val = sample_data
        
        scaler = LeakageSafeScaler()
        
        # Transform without fit should fail
        with pytest.raises(Exception, match=".*not.*fit.*|.*NotFitted.*"):
            scaler.transform(X_val)

    def test_fit_transform_only_on_training_data(self, sample_data):
        """
        Test that fit_transform() is only used on training data.
        
        Leakage scenario prevented:
        - fit_transform() on validation/test data would cause leakage.
        - Should only be used on training data.
        
        Expected behavior:
        - fit_transform() should work on training data
        - fit_transform() on validation data should raise warning or error
        """
        from src.preprocessing.transformers import LeakageSafeScaler
        
        X_train, X_val, y_train, y_val = sample_data
        
        scaler = LeakageSafeScaler()
        
        # fit_transform on training data should work
        X_train_scaled = scaler.fit_transform(X_train, y_train)
        assert X_train_scaled.shape == X_train.shape
        
        # fit_transform on validation data should raise error
        scaler_val = LeakageSafeScaler()
        with pytest.raises(ValueError, match=".*validation.*|.*test.*|.*leakage.*"):
            scaler_val.fit_transform(X_val, y_val, is_training=False)

    def test_pipeline_prevents_leakage_across_steps(self, sample_data):
        """
        Test that sklearn Pipeline maintains leakage-safe behavior.
        
        Leakage scenario prevented:
        - Multi-step pipelines (e.g., imputer -> scaler) must maintain
          fit/transform separation across all steps.
        
        Expected behavior:
        - Each pipeline step should be fitted only on training data
        - Validation data should only flow through transform()
        """
        from src.preprocessing.transformers import LeakageSafePipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        X_train, X_val, y_train, y_val = sample_data
        
        # Add some missing values
        X_train_missing = X_train.copy()
        X_train_missing[0, 0] = np.nan
        
        # Create pipeline
        pipeline = LeakageSafePipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        # Fit on training data
        pipeline.fit(X_train_missing, y_train)
        
        # Verify imputer was fitted on training data only
        expected_imputer_mean = np.nanmean(X_train_missing, axis=0)
        np.testing.assert_array_almost_equal(
            pipeline.named_steps['imputer'].statistics_,
            expected_imputer_mean,
            decimal=2,
            err_msg="Imputer should be fitted on training data only"
        )
        
        # Verify scaler was fitted on training data only (after imputation)
        X_train_imputed = pipeline.named_steps['imputer'].transform(X_train_missing)
        expected_scaler_mean = X_train_imputed.mean(axis=0)
        
        np.testing.assert_array_almost_equal(
            pipeline.named_steps['scaler'].mean_,
            expected_scaler_mean,
            decimal=2,
            err_msg="Scaler should be fitted on training data only"
        )
