"""
Deterministic and time-aware data splitting for fraud detection.

This module provides splitters that enforce:
1. Deterministic behavior with explicit random seed control
2. Time-aware splitting to prevent temporal leakage
3. Stratified splitting to maintain class distribution
"""

import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split


class DeterministicSplitter:
    """
    Deterministic random splitter with explicit seed control.
    
    Enforces reproducibility by requiring explicit random_seed parameter.
    Same seed always produces same train/validation split.
    
    Attributes:
        random_seed (int): Random seed for reproducibility
        val_size (float): Fraction of data for validation (0.0 to 1.0)
    """
    
    def __init__(self, random_seed: int = None, val_size: float = 0.2):
        """
        Initialize deterministic splitter.
        
        Args:
            random_seed (int): Random seed for reproducibility (REQUIRED)
            val_size (float): Validation set size (default: 0.2)
            
        Raises:
            ValueError: If random_seed is not provided
        """
        if random_seed is None:
            raise ValueError(
                "random_seed is required for deterministic splitting. "
                "Provide an explicit integer seed for reproducibility."
            )
        
        self.random_seed = random_seed
        self.val_size = val_size
    
    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and validation sets.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_data, val_data)
        """
        train, val = train_test_split(
            data,
            test_size=self.val_size,
            random_state=self.random_seed
        )
        
        return train, val


class TimeAwareSplitter:
    """
    Time-aware splitter that prevents temporal leakage.
    
    Splits data chronologically: training data contains earlier transactions,
    validation data contains later transactions. This simulates real-world
    deployment where we train on past data and predict future events.
    
    Attributes:
        time_column (str): Name of timestamp column
        val_size (float): Fraction of data for validation (0.0 to 1.0)
    """
    
    def __init__(self, time_column: str, val_size: float = 0.2):
        """
        Initialize time-aware splitter.
        
        Args:
            time_column (str): Name of timestamp column in data
            val_size (float): Validation set size (default: 0.2)
        """
        self.time_column = time_column
        self.val_size = val_size
    
    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data chronologically.
        
        Args:
            data (pd.DataFrame): Input dataset with time column
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_data, val_data)
            
        Raises:
            ValueError: If time_column not found in data
        """
        if self.time_column not in data.columns:
            raise ValueError(
                f"time_column '{self.time_column}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )
        
        # Sort by time (deterministic, no randomness)
        sorted_data = data.sort_values(by=self.time_column).reset_index(drop=True)
        
        # Calculate split point
        split_idx = int(len(sorted_data) * (1 - self.val_size))
        
        # Split chronologically
        train = sorted_data.iloc[:split_idx].copy()
        val = sorted_data.iloc[split_idx:].copy()
        
        return train, val


class StratifiedSplitter:
    """
    Stratified splitter that maintains class distribution.
    
    Ensures train and validation sets have similar fraud ratios,
    critical for imbalanced fraud detection datasets.
    
    Attributes:
        target_column (str): Name of target column for stratification
        random_seed (int): Random seed for reproducibility
        val_size (float): Fraction of data for validation (0.0 to 1.0)
    """
    
    def __init__(self, target_column: str, random_seed: int = None, val_size: float = 0.2):
        """
        Initialize stratified splitter.
        
        Args:
            target_column (str): Name of target column (e.g., 'is_fraud')
            random_seed (int): Random seed for reproducibility (REQUIRED)
            val_size (float): Validation set size (default: 0.2)
            
        Raises:
            ValueError: If random_seed is not provided
        """
        if random_seed is None:
            raise ValueError(
                "random_seed is required for stratified splitting. "
                "Provide an explicit integer seed for reproducibility."
            )
        
        self.target_column = target_column
        self.random_seed = random_seed
        self.val_size = val_size
    
    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data with stratification on target column.
        
        Args:
            data (pd.DataFrame): Input dataset with target column
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_data, val_data)
            
        Raises:
            ValueError: If target_column not found in data
        """
        if self.target_column not in data.columns:
            raise ValueError(
                f"target_column '{self.target_column}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )
        
        train, val = train_test_split(
            data,
            test_size=self.val_size,
            random_state=self.random_seed,
            stratify=data[self.target_column]
        )
        
        return train, val
