# Adaptive Fraud Detection ML Pipeline

Production-grade fraud detection system built with Test-Driven Development (TDD) for critical components, leakage-safe preprocessing, and deterministic behavior.

## Overview

This project implements an end-to-end machine learning pipeline for fraud detection with a focus on:
- **Data leakage prevention** through strict train/test separation
- **Deterministic behavior** for reproducible experiments
- **Production-ready components** with clear interfaces and error handling
- **Selective TDD** for business-critical logic

## Features

- ✅ **Leakage-Safe Preprocessing**: Transformers that enforce fit/transform separation
- ✅ **Deterministic Data Splitting**: Time-aware and stratified splitting with explicit seed control
- ✅ **Configurable Decision Logic**: Three-tier fraud decision system (AUTO_APPROVE, FLAG_FOR_REVIEW, BLOCK)
- ✅ **API Input Validation**: Pydantic schemas with business rule enforcement
- ✅ **Baseline Model Training**: Logistic Regression with class imbalance handling
- ✅ **Feature Drift Monitoring**: Statistical drift detection with threshold-based alerts
- ✅ **End-to-End Pipeline**: Integrated script orchestrating all components

## Testing Strategy

This project applies **selective Test-Driven Development (TDD)** to maximize reliability where it matters most while avoiding over-testing of components where manual validation is more appropriate.

### Components with TDD Coverage (36 tests)

**1. Leakage-Safe Preprocessing (6 tests)**
- **Why**: Data leakage is a silent killer in ML. Tests enforce that transformers are fitted only on training data and validation data never influences fitted parameters.
- **Coverage**: `LeakageSafeScaler`, `LeakageSafePipeline`

**2. Deterministic Data Splitting (8 tests)**
- **Why**: Reproducibility is critical for debugging and model comparison. Tests ensure same seed produces same split and time-aware splitting prevents temporal leakage.
- **Coverage**: `DeterministicSplitter`, `TimeAwareSplitter`, `StratifiedSplitter`

**3. Fraud Decision Thresholds (10 tests)**
- **Why**: Business logic must be correct and configurable. Tests enforce three-tier decision system and validate threshold configurations.
- **Coverage**: `FraudDecisionMaker` with AUTO_APPROVE, FLAG_FOR_REVIEW, BLOCK decisions

**4. API Input Validation (12 tests)**
- **Why**: Invalid inputs cause production failures. Tests ensure schema validation, business rules (positive amounts, ISO timestamps), and consistent error responses.
- **Coverage**: `TransactionRequest`, `PredictionResponse`, `BatchRequest`, `ErrorResponse` schemas

### Components Intentionally Not Over-Tested

**1. Model Training (`FraudModelTrainer`)**
- **Why**: Model accuracy is validated through evaluation metrics (ROC-AUC, precision, recall), not unit tests. Testing ML model outputs is brittle and provides false confidence.
- **Validation**: Manual evaluation on train/validation sets with logged metrics

**2. Feature Drift Monitoring (`FeatureDriftMonitor`)**
- **Why**: Drift detection thresholds are business decisions, not correctness criteria. Monitoring is validated through demonstration and manual inspection.
- **Validation**: Demonstration script showing normal vs. drifted scenarios

**3. End-to-End Pipeline (`run_pipeline.py`)**
- **Why**: Integration testing is done through execution. The pipeline orchestrates tested components, so testing the orchestration itself adds minimal value.
- **Validation**: Successful execution with logged outputs and saved artifacts

### Trade-offs

**TDD Applied**: Components with clear correctness criteria, high risk of silent failures, and deterministic behavior.

**TDD Skipped**: Components where correctness is subjective (model performance), validated through metrics (evaluation), or better tested through execution (orchestration).

This approach balances **reliability** (critical logic is tested) with **pragmatism** (avoid testing for the sake of testing).

## Installation

```bash
# Clone repository
git clone https://github.com/SourabhaKK/adaptive-fraud-detection-ml-pipeline.git
cd adaptive-fraud-detection-ml-pipeline

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run End-to-End Pipeline

```bash
python run_pipeline.py
```

This will:
1. Generate synthetic fraud detection data
2. Apply leakage-safe preprocessing
3. Perform time-aware train/validation split
4. Train baseline Logistic Regression model
5. Evaluate on train and validation sets
6. Save model artifacts to `models/` directory

### Use Trained Model

```python
from src.model_training.trainer import FraudModelTrainer

# Load trained model
trainer = FraudModelTrainer.load_model('models')

# Predict fraud probabilities
fraud_probs = trainer.predict_proba(X_new)

# Make business decisions
from src.decision_logic.decision_maker import FraudDecisionMaker

decision_maker = FraudDecisionMaker(low_threshold=0.1, high_threshold=0.5)
decisions = decision_maker.make_decisions(fraud_probs)
```

### Monitor Feature Drift

```python
from src.monitoring.drift_detector import FeatureDriftMonitor

# Initialize monitor
monitor = FeatureDriftMonitor(drift_threshold=0.3)
monitor.fit(X_train)

# Check new batch for drift
results = monitor.detect_drift(X_new)
monitor.log_drift_report(results)
```

Or run demonstration:
```bash
python -m src.monitoring.drift_detector
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_preprocessing.py -v
```

## Project Structure

```
adaptive-fraud-detection-ml-pipeline/
├── src/
│   ├── data_ingestion/
│   │   ├── loader.py              # Data loading utilities
│   │   └── splitter.py            # Deterministic data splitting
│   ├── preprocessing/
│   │   └── transformers.py        # Leakage-safe preprocessing
│   ├── model_training/
│   │   └── trainer.py             # Baseline model training
│   ├── decision_logic/
│   │   └── decision_maker.py      # Fraud decision thresholds
│   ├── inference_api/
│   │   └── schemas.py             # API validation schemas
│   └── monitoring/
│       └── drift_detector.py      # Feature drift monitoring
├── tests/
│   ├── test_preprocessing.py      # Leakage prevention tests
│   ├── test_data_ingestion.py     # Data splitting tests
│   ├── test_decision_logic.py     # Decision threshold tests
│   └── test_api_validation.py     # API validation tests
├── run_pipeline.py                # End-to-end pipeline script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Key Design Principles

1. **No Data Leakage**: Strict separation of training and validation data
2. **Deterministic Behavior**: Fixed random seeds for reproducibility
3. **Production-Ready**: Error handling, validation, and clear interfaces
4. **Selective Testing**: TDD for critical logic, manual validation for ML components
5. **Clear Documentation**: Explicit about trade-offs and design decisions

## License

MIT License
