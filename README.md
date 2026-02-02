# Adaptive Fraud Detection ML Pipeline

A production-oriented fraud detection system demonstrating Test-Driven Development (TDD) for critical components, leakage-safe preprocessing, and deterministic ML workflows.

---

## Problem Statement

Credit card fraud detection systems face three critical engineering challenges:

1. **Data Leakage**: Accidentally using validation/test data during training leads to overly optimistic metrics that collapse in production
2. **Non-Deterministic Behavior**: Irreproducible experiments make debugging impossible and prevent reliable model comparison
3. **Silent Failures**: Business logic errors (wrong thresholds, invalid inputs) can pass undetected until production

This project addresses these challenges through **selective TDD**, **leakage-safe preprocessing**, and **deterministic pipeline design**.

---

## Solution Overview

This ML pipeline enforces correctness where it matters most:

- **Leakage Prevention**: Transformers that enforce strict fit/transform separation
- **Deterministic Splitting**: Time-aware data splitting with explicit seed control
- **Tested Business Logic**: Decision thresholds and API validation covered by 36 tests
- **Baseline Model**: Logistic Regression with class imbalance handling
- **Drift Monitoring**: Statistical feature drift detection with threshold-based alerts

**Key Insight**: Not all ML code needs tests. We apply TDD to **deterministic, high-risk logic** (preprocessing, splitting, decisions) and use **metric-based validation** for ML components (model training, drift thresholds).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                      │
│  • TimeAwareSplitter (prevents temporal leakage)            │
│  • StratifiedSplitter (maintains class distribution)        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Preprocessing Layer                         │
│  • LeakageSafeScaler (fit on train only)                    │
│  • LeakageSafePipeline (enforces separation)                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Model Training Layer                       │
│  • FraudModelTrainer (Logistic Regression)                  │
│  • Class imbalance handling (balanced weights)              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Decision Logic Layer                        │
│  • FraudDecisionMaker (configurable thresholds)             │
│  • Three-tier: AUTO_APPROVE / FLAG_FOR_REVIEW / BLOCK       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    API & Validation Layer                    │
│  • Pydantic schemas (TransactionRequest, PredictionResponse)│
│  • Business rule enforcement (positive amounts, ISO dates)  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Monitoring Layer                          │
│  • FeatureDriftMonitor (statistical drift detection)        │
│  • Threshold-based alerts (30% default)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Machine Learning Approach

### Model Selection: Logistic Regression

**Why Logistic Regression?**
- **Interpretability**: Coefficients show feature importance for compliance/auditing
- **Speed**: Fast training and inference for real-time fraud detection
- **Baseline**: Establishes performance floor before trying complex models
- **Class Imbalance**: `class_weight='balanced'` handles ~5% fraud rate

**Trade-off**: Sacrifices non-linear pattern detection for interpretability and speed. In production, this would be A/B tested against gradient boosting models.

### Class Imbalance Strategy

Fraud datasets are heavily imbalanced (1-5% fraud). We handle this through:
- **Balanced Class Weights**: Automatically adjusts loss function to penalize false negatives
- **Stratified Splitting**: Maintains fraud ratio across train/validation sets
- **Evaluation Metrics**: ROC-AUC (threshold-independent) over accuracy

---

## Pipeline Design

### 1. Data Splitting (Time-Aware)

```python
# Training data: Earlier transactions
# Validation data: Later transactions
# Ensures no temporal leakage
splitter = TimeAwareSplitter(time_column='timestamp', val_size=0.2)
train_df, val_df = splitter.split(df)
```

**Why Time-Aware?** Fraud patterns evolve. Training on future data to predict the past is unrealistic.

### 2. Leakage-Safe Preprocessing

```python
scaler = LeakageSafeScaler()
X_train_scaled = scaler.fit_transform(X_train, is_training=True)  # Fit on train
X_val_scaled = scaler.transform(X_val)  # Transform only (no fit!)
```

**Enforced Separation**: Scaler fitted **only** on training data. Validation data never influences fitted parameters.

### 3. Model Training

```python
trainer = FraudModelTrainer(random_seed=42, class_weight='balanced')
metadata = trainer.train(X_train, y_train, X_val, y_val)
```

**Deterministic**: Fixed `random_seed=42` ensures reproducible results.

### 4. Decision Logic

```python
decision_maker = FraudDecisionMaker(low_threshold=0.1, high_threshold=0.5)
decisions = decision_maker.make_decisions(fraud_probabilities)
```

**Three-Tier System**:
- `prob < 0.1` → AUTO_APPROVE (low risk)
- `0.1 ≤ prob < 0.5` → FLAG_FOR_REVIEW (medium risk)
- `prob ≥ 0.5` → BLOCK (high risk)

**Configurable**: Thresholds adjustable based on business risk tolerance.

---

## Testing Strategy

### Selective TDD Approach

**TDD Applied (36 tests across 4 components):**

1. **Leakage-Safe Preprocessing (6 tests)**
   - **Why**: Data leakage is silent and catastrophic. Tests enforce fit/transform separation.
   - **Example**: `test_scaler_fitted_only_on_training_data` verifies scaler mean matches training data only.

2. **Deterministic Data Splitting (8 tests)**
   - **Why**: Reproducibility is critical for debugging. Tests ensure same seed → same split.
   - **Example**: `test_time_aware_split_prevents_future_leakage` verifies training dates < validation dates.

3. **Fraud Decision Thresholds (10 tests)**
   - **Why**: Business logic errors cause production failures. Tests enforce three-tier decision system.
   - **Example**: `test_low_risk_auto_approval` verifies `prob < 0.1` → AUTO_APPROVE.

4. **API Input Validation (12 tests)**
   - **Why**: Invalid inputs crash production systems. Tests enforce schema validation and business rules.
   - **Example**: `test_negative_amount_rejected` verifies amounts must be positive.

**TDD Skipped (Intentionally):**

1. **Model Training** → Validated through ROC-AUC, precision, recall (not unit tests)
2. **Drift Monitoring** → Thresholds are business decisions (demonstrated, not tested)
3. **End-to-End Pipeline** → Integration tested through execution (orchestration adds minimal risk)

**Rationale**: Test **deterministic correctness** (preprocessing, splitting, decisions). Validate **ML performance** through metrics (model accuracy, drift thresholds).

---

## Monitoring & Drift Detection

### Feature Distribution Drift

```python
monitor = FeatureDriftMonitor(drift_threshold=0.3)
monitor.fit(X_train)  # Baseline statistics

results = monitor.detect_drift(X_new)  # Check new batch
monitor.log_drift_report(results)  # Human-readable alert
```

**Metric**: Relative change in mean and standard deviation  
**Alert Condition**: Drift score > 30% triggers alert  
**Output**: Feature-level drift scores with old/new statistics

**Example Alert**:
```
⚠️  Feature: amount
   Drift score: 684.80% (threshold: 30.0%)
   Mean: 91.85 → 627.58 (583.3% change)
   
⚠️  ACTION REQUIRED: Investigate drift and consider retraining
```

**Trade-off**: Simple statistical drift (mean/std) over complex KL-divergence. Easier to explain to stakeholders.

---

## API & Inference Overview

### Request/Response Schemas (Pydantic)

**TransactionRequest**:
```python
{
  "transaction_id": "TXN123456",
  "amount": 150.50,  # Must be positive
  "merchant_category": "retail",
  "timestamp": "2024-01-15T10:30:00Z"  # ISO 8601
}
```

**PredictionResponse**:
```python
{
  "transaction_id": "TXN123456",
  "fraud_probability": 0.35,  # [0.0, 1.0]
  "decision": "FLAG_FOR_REVIEW",  # AUTO_APPROVE / FLAG_FOR_REVIEW / BLOCK
  "timestamp": "2024-01-15T10:30:05Z"
}
```

**Validation Rules**:
- Amounts must be positive (no negative transactions)
- Timestamps must be ISO 8601 format
- Transaction IDs must be non-empty
- Fraud probabilities must be in [0.0, 1.0]

---

## Project Structure

```
adaptive-fraud-detection-ml-pipeline/
├── src/
│   ├── data_ingestion/
│   │   └── splitter.py            # Time-aware, stratified splitting
│   ├── preprocessing/
│   │   └── transformers.py        # Leakage-safe preprocessing
│   ├── model_training/
│   │   └── trainer.py             # Logistic Regression trainer
│   ├── decision_logic/
│   │   └── decision_maker.py      # Three-tier decision system
│   ├── inference_api/
│   │   └── schemas.py             # Pydantic validation schemas
│   └── monitoring/
│       └── drift_detector.py      # Feature drift monitoring
├── tests/
│   ├── test_preprocessing.py      # 6 leakage prevention tests
│   ├── test_data_ingestion.py     # 8 data splitting tests
│   ├── test_decision_logic.py     # 10 decision threshold tests
│   └── test_api_validation.py     # 12 API validation tests
├── run_pipeline.py                # End-to-end orchestration
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

---

## How to Run

### 1. Installation

```bash
git clone https://github.com/SourabhaKK/adaptive-fraud-detection-ml-pipeline.git
cd adaptive-fraud-detection-ml-pipeline
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline

```bash
python run_pipeline.py
```

**Output**:
- Generates 10,000 synthetic transactions (5% fraud rate)
- Performs time-aware train/validation split (80/20)
- Trains Logistic Regression with leakage-safe preprocessing
- Evaluates on train and validation sets (ROC-AUC, precision, recall)
- Saves model artifacts to `models/` directory

### 3. Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=term-missing

# Specific component
pytest tests/test_preprocessing.py -v
```

### 4. Monitor Drift

```bash
python -m src.monitoring.drift_detector
```

**Output**: Demonstrates normal vs. drifted scenarios with alerts.

### 5. Use Trained Model

```python
from src.model_training.trainer import FraudModelTrainer
from src.decision_logic.decision_maker import FraudDecisionMaker

# Load model
trainer = FraudModelTrainer.load_model('models')

# Predict
fraud_probs = trainer.predict_proba(X_new)

# Make decisions
decision_maker = FraudDecisionMaker(low_threshold=0.1, high_threshold=0.5)
decisions = decision_maker.make_decisions(fraud_probs)
```

---

## Design Trade-offs & Limitations

### Trade-offs

1. **Logistic Regression over Gradient Boosting**
   - **Gain**: Interpretability, speed, simplicity
   - **Loss**: Non-linear pattern detection
   - **Justification**: Baseline model; complex models added after establishing performance floor

2. **Statistical Drift over KL-Divergence**
   - **Gain**: Explainability to stakeholders (mean/std changes)
   - **Loss**: Sensitivity to distribution shape changes
   - **Justification**: Simpler to debug and explain

3. **Selective TDD over 100% Coverage**
   - **Gain**: Focus on high-risk logic, avoid brittle ML tests
   - **Loss**: No unit tests for model accuracy
   - **Justification**: ML performance validated through metrics, not tests

### Limitations

1. **Synthetic Data**: Real fraud data has complex patterns not captured by synthetic generation
2. **No Feature Engineering**: Baseline uses raw features; production would add derived features (velocity, geographic patterns)
3. **No Model Retraining**: Pipeline trains once; production needs scheduled retraining
4. **No A/B Testing**: Decision thresholds set manually; production needs experimentation framework
5. **No Real-Time Inference**: Batch processing only; production needs streaming inference

---

## Future Improvements

### Short-Term (Production Readiness)

1. **Real Data Integration**: Replace synthetic data with anonymized transaction logs
2. **Feature Engineering**: Add velocity features (transactions/hour), geographic anomalies
3. **Model Comparison**: A/B test Logistic Regression vs. XGBoost/LightGBM
4. **Automated Retraining**: Scheduled pipeline runs (daily/weekly) with drift-triggered retraining

### Medium-Term (Scalability)

5. **Streaming Inference**: Kafka + real-time prediction API
6. **Distributed Training**: Spark MLlib for large-scale datasets
7. **Model Registry**: MLflow for versioning and experiment tracking
8. **Alerting System**: PagerDuty integration for drift alerts

### Long-Term (Advanced ML)

9. **Deep Learning**: LSTM for sequential transaction patterns
10. **Explainability**: SHAP values for fraud decision explanations
11. **Adversarial Robustness**: Defense against adversarial fraud patterns
12. **Multi-Model Ensemble**: Combine multiple models for improved accuracy

---

## Key Takeaways

1. **TDD for Critical Logic**: 36 tests enforce correctness in preprocessing, splitting, decisions, and API validation
2. **Leakage Prevention**: Strict fit/transform separation prevents overly optimistic metrics
3. **Deterministic Behavior**: Fixed seeds enable reproducible experiments and debugging
4. **Production-Oriented**: Clear interfaces, error handling, and explicit trade-offs
5. **Pragmatic Testing**: Test deterministic logic, validate ML through metrics

**This project demonstrates production ML engineering practices, not just model training.**

---

## License

MIT License

---

## Contact

For questions or collaboration: [GitHub Repository](https://github.com/SourabhaKK/adaptive-fraud-detection-ml-pipeline)
