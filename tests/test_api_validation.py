"""
Test suite for API input validation.

These tests enforce that:
1. Missing required fields raise validation errors
2. Invalid data types are rejected
3. API returns consistent error responses
4. Schema validation is enforced
"""

import pytest
from pydantic import ValidationError


class TestAPIInputValidation:
    """Test suite for API input validation and schema enforcement."""

    def test_valid_transaction_request(self):
        """
        Test that valid transaction request passes validation.
        
        Schema requirements:
        - transaction_id: string (required)
        - amount: float (required, positive)
        - merchant_category: string (required)
        - timestamp: string (required, ISO format)
        
        Expected behavior:
        - Valid request creates TransactionRequest object
        - All fields accessible
        """
        from src.inference_api.schemas import TransactionRequest
        
        valid_data = {
            'transaction_id': 'TXN123456',
            'amount': 150.50,
            'merchant_category': 'retail',
            'timestamp': '2024-01-15T10:30:00Z'
        }
        
        request = TransactionRequest(**valid_data)
        
        assert request.transaction_id == 'TXN123456'
        assert request.amount == 150.50
        assert request.merchant_category == 'retail'
        assert request.timestamp == '2024-01-15T10:30:00Z'

    def test_missing_required_field_raises_error(self):
        """
        Test that missing required fields raise validation errors.
        
        Schema enforcement:
        - All required fields must be present
        - Missing field raises ValidationError
        
        Expected behavior:
        - Missing 'amount' raises ValidationError
        - Error message mentions missing field
        """
        from src.inference_api.schemas import TransactionRequest
        
        invalid_data = {
            'transaction_id': 'TXN123456',
            # 'amount': missing!
            'merchant_category': 'retail',
            'timestamp': '2024-01-15T10:30:00Z'
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TransactionRequest(**invalid_data)
        
        error = exc_info.value
        assert 'amount' in str(error).lower(), \
            "Error should mention missing 'amount' field"

    def test_invalid_amount_type_rejected(self):
        """
        Test that invalid amount data type is rejected.
        
        Schema enforcement:
        - amount must be numeric (float)
        - String amount should be rejected
        
        Expected behavior:
        - String amount raises ValidationError
        """
        from src.inference_api.schemas import TransactionRequest
        
        invalid_data = {
            'transaction_id': 'TXN123456',
            'amount': 'not_a_number',  # Invalid type!
            'merchant_category': 'retail',
            'timestamp': '2024-01-15T10:30:00Z'
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TransactionRequest(**invalid_data)
        
        error = exc_info.value
        assert 'amount' in str(error).lower()

    def test_negative_amount_rejected(self):
        """
        Test that negative amounts are rejected.
        
        Business rule:
        - Transaction amounts must be positive (> 0)
        - Negative or zero amounts are invalid
        
        Expected behavior:
        - Negative amount raises ValidationError
        """
        from src.inference_api.schemas import TransactionRequest
        
        invalid_data = {
            'transaction_id': 'TXN123456',
            'amount': -50.00,  # Negative!
            'merchant_category': 'retail',
            'timestamp': '2024-01-15T10:30:00Z'
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TransactionRequest(**invalid_data)
        
        error = exc_info.value
        assert 'amount' in str(error).lower()

    def test_invalid_timestamp_format_rejected(self):
        """
        Test that invalid timestamp format is rejected.
        
        Schema enforcement:
        - timestamp must be ISO 8601 format
        - Invalid format raises ValidationError
        
        Expected behavior:
        - Non-ISO timestamp raises ValidationError
        """
        from src.inference_api.schemas import TransactionRequest
        
        invalid_data = {
            'transaction_id': 'TXN123456',
            'amount': 150.50,
            'merchant_category': 'retail',
            'timestamp': '15/01/2024 10:30'  # Invalid format!
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TransactionRequest(**invalid_data)
        
        error = exc_info.value
        assert 'timestamp' in str(error).lower()

    def test_empty_transaction_id_rejected(self):
        """
        Test that empty transaction_id is rejected.
        
        Business rule:
        - transaction_id must be non-empty string
        - Empty string is invalid
        
        Expected behavior:
        - Empty transaction_id raises ValidationError
        """
        from src.inference_api.schemas import TransactionRequest
        
        invalid_data = {
            'transaction_id': '',  # Empty!
            'amount': 150.50,
            'merchant_category': 'retail',
            'timestamp': '2024-01-15T10:30:00Z'
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TransactionRequest(**invalid_data)
        
        error = exc_info.value
        assert 'transaction_id' in str(error).lower()

    def test_prediction_response_schema(self):
        """
        Test that prediction response has correct schema.
        
        Response schema:
        - transaction_id: string
        - fraud_probability: float [0.0, 1.0]
        - decision: string (AUTO_APPROVE, FLAG_FOR_REVIEW, BLOCK)
        - timestamp: string (ISO format)
        
        Expected behavior:
        - Valid response creates PredictionResponse object
        """
        from src.inference_api.schemas import PredictionResponse
        
        valid_response = {
            'transaction_id': 'TXN123456',
            'fraud_probability': 0.35,
            'decision': 'FLAG_FOR_REVIEW',
            'timestamp': '2024-01-15T10:30:05Z'
        }
        
        response = PredictionResponse(**valid_response)
        
        assert response.transaction_id == 'TXN123456'
        assert response.fraud_probability == 0.35
        assert response.decision == 'FLAG_FOR_REVIEW'

    def test_fraud_probability_out_of_range_rejected(self):
        """
        Test that fraud_probability outside [0, 1] is rejected.
        
        Schema enforcement:
        - fraud_probability must be in range [0.0, 1.0]
        - Values > 1.0 or < 0.0 are invalid
        
        Expected behavior:
        - Out-of-range probability raises ValidationError
        """
        from src.inference_api.schemas import PredictionResponse
        
        invalid_response = {
            'transaction_id': 'TXN123456',
            'fraud_probability': 1.5,  # Out of range!
            'decision': 'BLOCK',
            'timestamp': '2024-01-15T10:30:05Z'
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**invalid_response)
        
        error = exc_info.value
        assert 'fraud_probability' in str(error).lower()

    def test_invalid_decision_value_rejected(self):
        """
        Test that invalid decision values are rejected.
        
        Schema enforcement:
        - decision must be one of: AUTO_APPROVE, FLAG_FOR_REVIEW, BLOCK
        - Other values are invalid
        
        Expected behavior:
        - Invalid decision raises ValidationError
        """
        from src.inference_api.schemas import PredictionResponse
        
        invalid_response = {
            'transaction_id': 'TXN123456',
            'fraud_probability': 0.35,
            'decision': 'MAYBE',  # Invalid!
            'timestamp': '2024-01-15T10:30:05Z'
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**invalid_response)
        
        error = exc_info.value
        assert 'decision' in str(error).lower()

    def test_batch_request_validation(self):
        """
        Test batch request validation.
        
        Schema requirements:
        - transactions: list of TransactionRequest
        - List must not be empty
        
        Expected behavior:
        - Valid batch creates BatchRequest object
        - Empty batch raises ValidationError
        """
        from src.inference_api.schemas import BatchRequest, TransactionRequest
        
        # Valid batch
        valid_batch = {
            'transactions': [
                {
                    'transaction_id': 'TXN001',
                    'amount': 100.0,
                    'merchant_category': 'retail',
                    'timestamp': '2024-01-15T10:30:00Z'
                },
                {
                    'transaction_id': 'TXN002',
                    'amount': 200.0,
                    'merchant_category': 'online',
                    'timestamp': '2024-01-15T10:31:00Z'
                }
            ]
        }
        
        batch = BatchRequest(**valid_batch)
        assert len(batch.transactions) == 2

    def test_empty_batch_rejected(self):
        """
        Test that empty batch is rejected.
        
        Business rule:
        - Batch must contain at least one transaction
        - Empty list is invalid
        
        Expected behavior:
        - Empty batch raises ValidationError
        """
        from src.inference_api.schemas import BatchRequest
        
        invalid_batch = {
            'transactions': []  # Empty!
        }
        
        with pytest.raises(ValidationError) as exc_info:
            BatchRequest(**invalid_batch)
        
        error = exc_info.value
        assert 'transactions' in str(error).lower()

    def test_error_response_schema(self):
        """
        Test that error responses have consistent schema.
        
        Error response schema:
        - error: string (error type)
        - message: string (human-readable message)
        - details: dict (optional, validation details)
        
        Expected behavior:
        - Consistent error format for all validation errors
        """
        from src.inference_api.schemas import ErrorResponse
        
        error_response = {
            'error': 'ValidationError',
            'message': 'Invalid transaction data',
            'details': {
                'field': 'amount',
                'issue': 'must be positive'
            }
        }
        
        response = ErrorResponse(**error_response)
        
        assert response.error == 'ValidationError'
        assert response.message == 'Invalid transaction data'
        assert 'field' in response.details
