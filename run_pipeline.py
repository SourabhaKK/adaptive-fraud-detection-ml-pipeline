"""
End-to-end fraud detection ML pipeline.

This script orchestrates the complete ML pipeline:
1. Load raw transaction data
2. Apply leakage-safe preprocessing
3. Perform deterministic train/validation split
4. Train baseline fraud detection model
5. Evaluate model performance
6. Persist trained model and metadata

Usage:
    python run_pipeline.py

Requirements:
    - Sample data in data/ directory (or generates synthetic data)
    - All dependencies installed (see requirements.txt)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_ingestion.splitter import TimeAwareSplitter, StratifiedSplitter
from src.model_training.trainer import FraudModelTrainer


def generate_synthetic_data(n_samples: int = 10000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic fraud detection data for demonstration.
    
    Args:
        n_samples: Number of transactions to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with transaction data
    """
    print(f"\n{'='*60}")
    print("STEP 1: GENERATING SYNTHETIC DATA")
    print(f"{'='*60}")
    
    np.random.seed(random_seed)
    
    # Generate timestamps over 100 days
    base_date = datetime(2024, 1, 1)
    timestamps = [
        base_date + timedelta(days=int(i / 100))
        for i in range(n_samples)
    ]
    
    # Generate features
    data = {
        'transaction_id': [f'TXN{i:06d}' for i in range(n_samples)],
        'timestamp': timestamps,
        'amount': np.random.lognormal(mean=4, sigma=1.5, size=n_samples),
        'merchant_category': np.random.choice(
            ['retail', 'online', 'travel', 'food', 'entertainment'],
            size=n_samples
        ),
        'hour_of_day': np.random.randint(0, 24, size=n_samples),
        'day_of_week': np.random.randint(0, 7, size=n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate fraud labels (5% fraud rate)
    # Fraud more likely for high amounts and certain categories
    fraud_score = (
        (df['amount'] > df['amount'].quantile(0.9)).astype(float) * 0.3 +
        (df['merchant_category'] == 'online').astype(float) * 0.2 +
        np.random.uniform(0, 0.5, size=n_samples)
    )
    df['is_fraud'] = (fraud_score > 0.7).astype(int)
    
    print(f"✓ Generated {len(df):,} transactions")
    print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Features: {list(df.columns)}")
    
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for modeling.
    
    Args:
        df: Raw transaction data
        
    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    print(f"\n{'='*60}")
    print("STEP 2: PREPARING FEATURES")
    print(f"{'='*60}")
    
    # Select numeric features for baseline model
    feature_cols = ['amount', 'hour_of_day', 'day_of_week']
    
    # One-hot encode merchant category
    merchant_dummies = pd.get_dummies(df['merchant_category'], prefix='merchant')
    
    # Combine features
    X = pd.concat([df[feature_cols], merchant_dummies], axis=1)
    y = df['is_fraud']
    
    print(f"✓ Prepared {X.shape[1]} features")
    print(f"  Numeric features: {feature_cols}")
    print(f"  Categorical features: merchant_category ({len(merchant_dummies.columns)} categories)")
    
    return X, y


def split_data(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, random_seed: int = 42):
    """
    Perform time-aware train/validation split.
    
    Args:
        df: Full dataframe with timestamp
        X: Features
        y: Target
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    print(f"\n{'='*60}")
    print("STEP 3: DETERMINISTIC TRAIN/VALIDATION SPLIT")
    print(f"{'='*60}")
    
    # Time-aware split (80/20)
    splitter = TimeAwareSplitter(time_column='timestamp', val_size=0.2)
    train_df, val_df = splitter.split(df)
    
    # Get indices
    train_idx = train_df.index
    val_idx = val_df.index
    
    # Split features and target
    X_train = X.loc[train_idx]
    X_val = X.loc[val_idx]
    y_train = y.loc[train_idx]
    y_val = y.loc[val_idx]
    
    print(f"✓ Time-aware split complete")
    print(f"  Training set: {len(X_train):,} samples ({len(X_train)/len(X):.1%})")
    print(f"    Fraud rate: {y_train.mean():.2%}")
    print(f"    Date range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"  Validation set: {len(X_val):,} samples ({len(X_val)/len(X):.1%})")
    print(f"    Fraud rate: {y_val.mean():.2%}")
    print(f"    Date range: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    
    # Verify no temporal leakage
    assert train_df['timestamp'].max() <= val_df['timestamp'].min(), \
        "Temporal leakage detected!"
    print(f"  ✓ No temporal leakage (train ends before validation starts)")
    
    return X_train, X_val, y_train, y_val


def train_model(X_train, X_val, y_train, y_val, random_seed: int = 42):
    """
    Train fraud detection model with leakage-safe preprocessing.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        y_val: Validation target
        random_seed: Random seed for reproducibility
        
    Returns:
        Trained FraudModelTrainer instance
    """
    print(f"\n{'='*60}")
    print("STEP 4: TRAINING FRAUD DETECTION MODEL")
    print(f"{'='*60}")
    
    # Initialize trainer
    trainer = FraudModelTrainer(random_seed=random_seed, class_weight='balanced')
    
    # Train model (includes leakage-safe preprocessing)
    metadata = trainer.train(X_train, y_train, X_val, y_val)
    
    return trainer


def save_artifacts(trainer: FraudModelTrainer, model_dir: str = 'models'):
    """
    Save trained model and metadata.
    
    Args:
        trainer: Trained model trainer
        model_dir: Directory to save artifacts
    """
    print(f"\n{'='*60}")
    print("STEP 5: SAVING MODEL ARTIFACTS")
    print(f"{'='*60}")
    
    paths = trainer.save_model(model_dir=model_dir)
    
    return paths


def main():
    """Run the complete fraud detection ML pipeline."""
    print("\n" + "="*60)
    print("FRAUD DETECTION ML PIPELINE")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    RANDOM_SEED = 42
    N_SAMPLES = 10000
    MODEL_DIR = 'models'
    
    try:
        # Step 1: Generate/load data
        df = generate_synthetic_data(n_samples=N_SAMPLES, random_seed=RANDOM_SEED)
        
        # Step 2: Prepare features
        X, y = prepare_features(df)
        
        # Step 3: Split data (time-aware, no leakage)
        X_train, X_val, y_train, y_val = split_data(df, X, y, random_seed=RANDOM_SEED)
        
        # Step 4: Train model (leakage-safe preprocessing)
        trainer = train_model(X_train, X_val, y_train, y_val, random_seed=RANDOM_SEED)
        
        # Step 5: Save artifacts
        paths = save_artifacts(trainer, model_dir=MODEL_DIR)
        
        # Summary
        print(f"\n{'='*60}")
        print("PIPELINE COMPLETE ✓")
        print(f"{'='*60}")
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nModel artifacts saved to: {MODEL_DIR}/")
        print("  - fraud_model.joblib (trained model)")
        print("  - scaler.joblib (fitted scaler)")
        print("  - training_metadata.json (training info)")
        print(f"\nTo use the model:")
        print(f"  from src.model_training.trainer import FraudModelTrainer")
        print(f"  trainer = FraudModelTrainer.load_model('{MODEL_DIR}')")
        print(f"  fraud_probs = trainer.predict_proba(X_new)")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("PIPELINE FAILED ✗")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
