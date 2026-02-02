"""
API request and response schemas using Pydantic.

This module provides schema validation for:
- Transaction requests
- Prediction responses
- Batch requests
- Error responses

Key principles:
1. Schema-based validation (Pydantic)
2. Clear error messaging
3. Business rule enforcement
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Literal
from datetime import datetime


class TransactionRequest(BaseModel):
    """
    Schema for fraud detection transaction request.
    
    Attributes:
        transaction_id: Unique transaction identifier (non-empty)
        amount: Transaction amount (positive)
        merchant_category: Merchant category code
        timestamp: Transaction timestamp (ISO 8601 format)
    """
    transaction_id: str = Field(..., min_length=1, description="Unique transaction ID")
    amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    merchant_category: str = Field(..., description="Merchant category")
    timestamp: str = Field(..., description="Transaction timestamp (ISO 8601)")
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """
        Validate timestamp is in ISO 8601 format.
        
        Args:
            v: Timestamp string
            
        Returns:
            str: Validated timestamp
            
        Raises:
            ValueError: If timestamp format is invalid
        """
        try:
            # Try to parse as ISO 8601
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except (ValueError, AttributeError):
            raise ValueError(
                f"timestamp must be in ISO 8601 format, got: {v}"
            )


class PredictionResponse(BaseModel):
    """
    Schema for fraud prediction response.
    
    Attributes:
        transaction_id: Transaction identifier
        fraud_probability: Fraud probability [0.0, 1.0]
        decision: Business decision (AUTO_APPROVE, FLAG_FOR_REVIEW, BLOCK)
        timestamp: Response timestamp (ISO 8601)
    """
    transaction_id: str
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Fraud probability [0, 1]")
    decision: Literal['AUTO_APPROVE', 'FLAG_FOR_REVIEW', 'BLOCK'] = Field(
        ..., 
        description="Decision: AUTO_APPROVE, FLAG_FOR_REVIEW, or BLOCK"
    )
    timestamp: str
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate timestamp is in ISO 8601 format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except (ValueError, AttributeError):
            raise ValueError(
                f"timestamp must be in ISO 8601 format, got: {v}"
            )


class BatchRequest(BaseModel):
    """
    Schema for batch fraud detection request.
    
    Attributes:
        transactions: List of transaction requests (non-empty)
    """
    transactions: List[TransactionRequest] = Field(
        ..., 
        min_length=1,
        description="List of transactions (must contain at least one)"
    )


class ErrorResponse(BaseModel):
    """
    Schema for consistent error responses.
    
    Attributes:
        error: Error type/code
        message: Human-readable error message
        details: Optional additional error details
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")
