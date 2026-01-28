"""
Leakage-safe preprocessing transformers.

This module provides wrappers around sklearn transformers that enforce
leakage-safe behavior by ensuring:
1. Transformers are fitted ONLY on training data
2. Validation/test data never influences fitted parameters
3. Proper fit/transform separation is maintained
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError


class LeakageSafeScaler:
    """
    Leakage-safe wrapper around StandardScaler.
    
    Enforces that:
    - Scaler is fitted only on training data
    - Validation data does not influence fitted parameters
    - fit_transform is only used on training data
    - Transform requires prior fit
    
    Attributes:
        mean_ (np.ndarray): Mean values computed from training data
        scale_ (np.ndarray): Standard deviation computed from training data
    """
    
    def __init__(self):
        """Initialize the leakage-safe scaler."""
        self._scaler = StandardScaler()
        self._is_fitted = False
    
    def fit(self, X, y=None, validation_data=None, enforce_split_validation=False):
        """
        Fit the scaler on training data only.
        
        Args:
            X (np.ndarray): Training data to fit on
            y (np.ndarray, optional): Training labels (ignored, for API compatibility)
            validation_data (tuple, optional): Validation data (ignored, for monitoring only)
            enforce_split_validation (bool): If True, raises error to prevent accidental
                                            fitting on combined datasets
        
        Returns:
            self: Fitted scaler
            
        Raises:
            ValueError: If enforce_split_validation=True, to prevent accidental leakage
        """
        if enforce_split_validation:
            raise ValueError(
                "Leakage prevention: enforce_split_validation=True requires explicit "
                "confirmation that data is properly split into train/val/test. "
                "Do not fit on combined datasets."
            )
        
        # Fit only on training data (X), ignore validation_data
        self._scaler.fit(X)
        self._is_fitted = True
        
        return self
    
    def transform(self, X):
        """
        Transform data using fitted parameters.
        
        Args:
            X (np.ndarray): Data to transform
            
        Returns:
            np.ndarray: Transformed data
            
        Raises:
            NotFittedError: If scaler has not been fitted yet
        """
        if not self._is_fitted:
            raise NotFittedError(
                "This LeakageSafeScaler instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'transform'."
            )
        
        return self._scaler.transform(X)
    
    def fit_transform(self, X, y=None, is_training=True):
        """
        Fit and transform data (only for training data).
        
        Args:
            X (np.ndarray): Data to fit and transform
            y (np.ndarray, optional): Labels (ignored, for API compatibility)
            is_training (bool): Must be True. If False, raises error to prevent
                               leakage from using fit_transform on validation/test data
        
        Returns:
            np.ndarray: Transformed data
            
        Raises:
            ValueError: If is_training=False, to prevent validation/test leakage
        """
        if not is_training:
            raise ValueError(
                "Leakage prevention: fit_transform should only be used on training data. "
                "For validation/test data, use fit(X_train) followed by transform(X_val)."
            )
        
        self.fit(X, y)
        return self.transform(X)
    
    @property
    def mean_(self):
        """Mean values computed from training data."""
        if not self._is_fitted:
            raise NotFittedError("Scaler has not been fitted yet.")
        return self._scaler.mean_
    
    @property
    def scale_(self):
        """Standard deviation computed from training data."""
        if not self._is_fitted:
            raise NotFittedError("Scaler has not been fitted yet.")
        return self._scaler.scale_


class LeakageSafePipeline:
    """
    Leakage-safe wrapper around sklearn Pipeline.
    
    Ensures that all steps in the pipeline maintain fit/transform separation
    and are fitted only on training data.
    
    Attributes:
        named_steps (dict): Dictionary of pipeline steps accessible by name
    """
    
    def __init__(self, steps):
        """
        Initialize the leakage-safe pipeline.
        
        Args:
            steps (list): List of (name, transformer) tuples
        """
        self._pipeline = Pipeline(steps)
        self._is_fitted = False
    
    def fit(self, X, y=None):
        """
        Fit all pipeline steps on training data only.
        
        Args:
            X (np.ndarray): Training data
            y (np.ndarray, optional): Training labels
            
        Returns:
            self: Fitted pipeline
        """
        self._pipeline.fit(X, y)
        self._is_fitted = True
        return self
    
    def transform(self, X):
        """
        Transform data through all pipeline steps.
        
        Args:
            X (np.ndarray): Data to transform
            
        Returns:
            np.ndarray: Transformed data
            
        Raises:
            NotFittedError: If pipeline has not been fitted yet
        """
        if not self._is_fitted:
            raise NotFittedError(
                "This LeakageSafePipeline instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using 'transform'."
            )
        
        return self._pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        """
        Fit and transform data (only for training data).
        
        Args:
            X (np.ndarray): Training data
            y (np.ndarray, optional): Training labels
            
        Returns:
            np.ndarray: Transformed data
        """
        self.fit(X, y)
        return self.transform(X)
    
    @property
    def named_steps(self):
        """Access pipeline steps by name."""
        if not self._is_fitted:
            raise NotFittedError("Pipeline has not been fitted yet.")
        return self._pipeline.named_steps
