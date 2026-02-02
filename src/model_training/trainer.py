"""
Baseline fraud detection model training using Logistic Regression.

This module provides model training with:
1. Leakage-safe preprocessing integration
2. Class imbalance handling
3. Reproducible training with fixed random seeds
4. Clear model persistence

Key principles:
- No data leakage (fit on train, transform on val)
- Handle imbalanced fraud data
- Deterministic behavior
- Production-ready model artifacts
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from pathlib import Path
from typing import Tuple, Dict, Any
import json
from datetime import datetime


class FraudModelTrainer:
    """
    Baseline fraud detection model trainer.
    
    Uses Logistic Regression with class imbalance handling.
    Integrates with leakage-safe preprocessing.
    
    Attributes:
        random_seed: Random seed for reproducibility
        class_weight: Strategy for handling class imbalance
        model: Trained Logistic Regression model
        scaler: Fitted preprocessing scaler
    """
    
    def __init__(self, random_seed: int = 42, class_weight: str = 'balanced'):
        """
        Initialize fraud model trainer.
        
        Args:
            random_seed: Random seed for reproducibility
            class_weight: Class weight strategy ('balanced' or None)
                         'balanced' automatically adjusts weights for imbalanced data
        """
        self.random_seed = random_seed
        self.class_weight = class_weight
        self.model = None
        self.scaler = None
        self.training_metadata = {}
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ) -> Dict[str, Any]:
        """
        Train fraud detection model with leakage-safe preprocessing.
        
        Args:
            X_train: Training features
            y_train: Training labels (0=legitimate, 1=fraud)
            X_val: Optional validation features
            y_val: Optional validation labels
            
        Returns:
            Dict with training metrics and metadata
        """
        from src.preprocessing.transformers import LeakageSafeScaler
        
        print(f"Training fraud detection model...")
        print(f"Training set: {len(X_train)} samples")
        print(f"Fraud ratio: {y_train.mean():.2%}")
        
        # Initialize leakage-safe scaler
        self.scaler = LeakageSafeScaler()
        
        # Fit scaler ONLY on training data (no leakage!)
        X_train_scaled = self.scaler.fit_transform(X_train, is_training=True)
        
        # Initialize Logistic Regression with class imbalance handling
        self.model = LogisticRegression(
            random_state=self.random_seed,
            class_weight=self.class_weight,
            max_iter=1000,
            solver='lbfgs'
        )
        
        # Train model
        print("Fitting Logistic Regression...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on training set
        train_metrics = self._evaluate(X_train_scaled, y_train, "Training")
        
        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            print(f"\nValidation set: {len(X_val)} samples")
            # Transform validation data (no fitting!)
            X_val_scaled = self.scaler.transform(X_val)
            val_metrics = self._evaluate(X_val_scaled, y_val, "Validation")
        
        # Store training metadata
        self.training_metadata = {
            'timestamp': datetime.now().isoformat(),
            'random_seed': self.random_seed,
            'class_weight': self.class_weight,
            'n_train_samples': len(X_train),
            'train_fraud_ratio': float(y_train.mean()),
            'n_features': X_train.shape[1],
            'feature_names': list(X_train.columns),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics if val_metrics else None
        }
        
        print("\n✓ Model training complete!")
        return self.training_metadata
    
    def _evaluate(self, X_scaled: np.ndarray, y_true: pd.Series, dataset_name: str) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X_scaled: Scaled features
            y_true: True labels
            dataset_name: Name of dataset (for logging)
            
        Returns:
            Dict with evaluation metrics
        """
        # Predictions
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        print(f"\n{dataset_name} Metrics:")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud']))
        
        return {
            'roc_auc': float(roc_auc),
            'n_samples': len(y_true),
            'fraud_ratio': float(y_true.mean())
        }
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of fraud probabilities [0.0, 1.0]
            
        Raises:
            ValueError: If model not trained yet
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Transform features (no fitting!)
        X_scaled = self.scaler.transform(X)
        
        # Return fraud probabilities (class 1)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save_model(self, model_dir: str = 'models') -> Dict[str, str]:
        """
        Save trained model and metadata.
        
        Args:
            model_dir: Directory to save model artifacts
            
        Returns:
            Dict with paths to saved artifacts
            
        Raises:
            ValueError: If model not trained yet
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Create model directory
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_path / 'fraud_model.joblib'
        joblib.dump(self.model, model_file)
        
        # Save scaler
        scaler_file = model_path / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_file)
        
        # Save metadata
        metadata_file = model_path / 'training_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        
        print(f"\n✓ Model artifacts saved to: {model_dir}/")
        print(f"  - Model: {model_file.name}")
        print(f"  - Scaler: {scaler_file.name}")
        print(f"  - Metadata: {metadata_file.name}")
        
        return {
            'model': str(model_file),
            'scaler': str(scaler_file),
            'metadata': str(metadata_file)
        }
    
    @classmethod
    def load_model(cls, model_dir: str = 'models') -> 'FraudModelTrainer':
        """
        Load trained model from disk.
        
        Args:
            model_dir: Directory containing model artifacts
            
        Returns:
            FraudModelTrainer instance with loaded model
            
        Raises:
            FileNotFoundError: If model files not found
        """
        model_path = Path(model_dir)
        
        # Load model
        model_file = model_path / 'fraud_model.joblib'
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load scaler
        scaler_file = model_path / 'scaler.joblib'
        if not scaler_file.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
        
        # Load metadata
        metadata_file = model_path / 'training_metadata.json'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Create trainer instance
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        trainer = cls(
            random_seed=metadata.get('random_seed', 42),
            class_weight=metadata.get('class_weight', 'balanced')
        )
        
        # Load artifacts
        trainer.model = joblib.load(model_file)
        trainer.scaler = joblib.load(scaler_file)
        trainer.training_metadata = metadata
        
        print(f"✓ Model loaded from: {model_dir}/")
        print(f"  Trained: {metadata.get('timestamp', 'unknown')}")
        print(f"  Features: {metadata.get('n_features', 'unknown')}")
        
        return trainer
