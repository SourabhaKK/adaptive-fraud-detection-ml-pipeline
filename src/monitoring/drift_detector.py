"""
Feature distribution drift monitoring for fraud detection.

This module provides simple drift detection by comparing:
- Training data feature statistics (mean, std)
- New incoming batch feature statistics

Alert triggered when drift exceeds threshold.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from datetime import datetime


class FeatureDriftMonitor:
    """
    Monitor feature distribution drift using statistical comparison.
    
    Compares mean and standard deviation of features between
    training data and new incoming batches.
    
    Attributes:
        training_stats: Statistics computed from training data
        drift_threshold: Threshold for triggering drift alert (default: 0.3)
    """
    
    def __init__(self, drift_threshold: float = 0.3):
        """
        Initialize drift monitor.
        
        Args:
            drift_threshold: Relative change threshold for drift alert
                            (e.g., 0.3 = 30% change triggers alert)
        """
        self.drift_threshold = drift_threshold
        self.training_stats = None
    
    def fit(self, X_train: pd.DataFrame):
        """
        Compute baseline statistics from training data.
        
        Args:
            X_train: Training features
        """
        self.training_stats = {
            'mean': X_train.mean().to_dict(),
            'std': X_train.std().to_dict(),
            'n_samples': len(X_train),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"✓ Baseline statistics computed from {len(X_train):,} training samples")
    
    def detect_drift(self, X_new: pd.DataFrame) -> Dict:
        """
        Detect feature drift in new data batch.
        
        Args:
            X_new: New data batch to check for drift
            
        Returns:
            Dict with drift detection results
        """
        if self.training_stats is None:
            raise ValueError("Monitor not fitted. Call fit() first.")
        
        # Compute statistics for new batch
        new_mean = X_new.mean()
        new_std = X_new.std()
        
        # Calculate relative drift for each feature
        drift_scores = {}
        alerts = []
        
        for feature in X_new.columns:
            if feature not in self.training_stats['mean']:
                continue
            
            train_mean = self.training_stats['mean'][feature]
            train_std = self.training_stats['std'][feature]
            
            # Calculate relative change in mean
            if train_mean != 0:
                mean_drift = abs(new_mean[feature] - train_mean) / abs(train_mean)
            else:
                mean_drift = 0.0
            
            # Calculate relative change in std
            if train_std != 0:
                std_drift = abs(new_std[feature] - train_std) / abs(train_std)
            else:
                std_drift = 0.0
            
            # Overall drift score (max of mean and std drift)
            drift_score = max(mean_drift, std_drift)
            drift_scores[feature] = drift_score
            
            # Check if drift exceeds threshold
            if drift_score > self.drift_threshold:
                alerts.append({
                    'feature': feature,
                    'drift_score': drift_score,
                    'train_mean': train_mean,
                    'new_mean': new_mean[feature],
                    'train_std': train_std,
                    'new_std': new_std[feature]
                })
        
        # Prepare results
        results = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(X_new),
            'drift_detected': len(alerts) > 0,
            'n_features_drifted': len(alerts),
            'drift_scores': drift_scores,
            'alerts': alerts,
            'threshold': self.drift_threshold
        }
        
        return results
    
    def log_drift_report(self, results: Dict):
        """
        Log drift detection results in human-readable format.
        
        Args:
            results: Results from detect_drift()
        """
        print("\n" + "="*60)
        print("FEATURE DRIFT MONITORING REPORT")
        print("="*60)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Batch size: {results['n_samples']:,} samples")
        print(f"Drift threshold: {results['threshold']:.1%}")
        print(f"\nDrift detected: {'⚠️  YES' if results['drift_detected'] else '✓ NO'}")
        
        if results['drift_detected']:
            print(f"Features with drift: {results['n_features_drifted']}")
            print("\nDrift Alerts:")
            print("-" * 60)
            
            for alert in results['alerts']:
                print(f"\n⚠️  Feature: {alert['feature']}")
                print(f"   Drift score: {alert['drift_score']:.2%} (threshold: {results['threshold']:.1%})")
                print(f"   Mean: {alert['train_mean']:.2f} → {alert['new_mean']:.2f} "
                      f"({(alert['new_mean'] - alert['train_mean']) / alert['train_mean']:.1%} change)")
                print(f"   Std:  {alert['train_std']:.2f} → {alert['new_std']:.2f} "
                      f"({(alert['new_std'] - alert['train_std']) / alert['train_std']:.1%} change)")
            
            print("\n" + "="*60)
            print("⚠️  ACTION REQUIRED: Investigate drift and consider retraining")
            print("="*60)
        else:
            print("\nAll features within acceptable drift range ✓")
            print("="*60)


def demonstrate_drift_monitoring():
    """
    Demonstrate drift monitoring with example data.
    
    Shows two scenarios:
    1. No drift: New data similar to training data
    2. Drift detected: New data significantly different
    """
    print("\n" + "="*60)
    print("DRIFT MONITORING DEMONSTRATION")
    print("="*60)
    
    # Generate training data
    np.random.seed(42)
    n_train = 1000
    X_train = pd.DataFrame({
        'amount': np.random.lognormal(mean=4, sigma=1, size=n_train),
        'hour_of_day': np.random.randint(0, 24, size=n_train),
        'day_of_week': np.random.randint(0, 7, size=n_train)
    })
    
    print(f"\nTraining data: {len(X_train):,} samples")
    print(f"  amount: mean={X_train['amount'].mean():.2f}, std={X_train['amount'].std():.2f}")
    print(f"  hour_of_day: mean={X_train['hour_of_day'].mean():.2f}, std={X_train['hour_of_day'].std():.2f}")
    
    # Initialize monitor
    monitor = FeatureDriftMonitor(drift_threshold=0.3)
    monitor.fit(X_train)
    
    # Scenario 1: No drift (similar distribution)
    print("\n" + "="*60)
    print("SCENARIO 1: Normal batch (no drift expected)")
    print("="*60)
    
    X_normal = pd.DataFrame({
        'amount': np.random.lognormal(mean=4, sigma=1, size=100),
        'hour_of_day': np.random.randint(0, 24, size=100),
        'day_of_week': np.random.randint(0, 7, size=100)
    })
    
    results_normal = monitor.detect_drift(X_normal)
    monitor.log_drift_report(results_normal)
    
    # Scenario 2: Drift detected (different distribution)
    print("\n" + "="*60)
    print("SCENARIO 2: Anomalous batch (drift expected)")
    print("="*60)
    print("Simulating: Higher transaction amounts, different time patterns")
    
    X_drifted = pd.DataFrame({
        'amount': np.random.lognormal(mean=5.5, sigma=1.5, size=100),  # Higher amounts!
        'hour_of_day': np.random.randint(20, 24, size=100),  # Only late hours!
        'day_of_week': np.random.randint(0, 7, size=100)
    })
    
    results_drifted = monitor.detect_drift(X_drifted)
    monitor.log_drift_report(results_drifted)


if __name__ == '__main__':
    demonstrate_drift_monitoring()
