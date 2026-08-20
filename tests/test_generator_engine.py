import pytest
from uuid import uuid4
from pydantic import ValidationError
from src.data.generator_engine import SyntheticStateRegistry, CustomerRecord, TransactionRecord

def test_registry_compound_key_collision():
    registry = SyntheticStateRegistry()
    registry.register_customer("user1", "tax123", "email1@example.com")
    
    with pytest.raises(ValueError, match="Duplicate compound key detected"):
        registry.register_customer("user2", "tax123", "email1@example.com")

def test_registry_duplicate_user_id():
    registry = SyntheticStateRegistry()
    registry.register_customer("user1", "tax123", "email1@example.com")
    
    with pytest.raises(ValueError, match="Duplicate user_id detected"):
        registry.register_customer("user1", "tax999", "email999@example.com")

def test_immutability_validation():
    registry = SyntheticStateRegistry()
    customer = registry.register_customer("user1", "tax123", "email1@example.com")
    
    with pytest.raises(ValidationError):
        customer.user_id = "user2"
        
    tx = TransactionRecord(
        transaction_id=uuid4(),
        sender_id="user1",
        recipient_id="user2",
        transaction_amount=50.0,
        timestamp="2025-01-01T12:00:00",
        channel_type="App",
        is_fuliza=False,
        is_mshwari=False
    )
    with pytest.raises(ValidationError):
        tx.transaction_amount = 100.0

def test_transactional_row_mapping():
    registry = SyntheticStateRegistry()
    registry.register_customer("user1", "tax1", "email1@example.com")
    registry.register_customer("user2", "tax2", "email2@example.com")
    
    tx = TransactionRecord(
        transaction_id=uuid4(),
        sender_id="user1",
        recipient_id="user2",
        transaction_amount=150.0,
        timestamp="2025-01-01T12:00:00",
        channel_type="USSD",
        is_fuliza=False,
        is_mshwari=True
    )
    
    # Simulate parsing back
    sender = registry.get_customer(tx.sender_id)
    recipient = registry.get_customer(tx.recipient_id)
    
    assert sender.user_id == "user1"
    assert recipient.user_id == "user2"
    assert sender.tax_id == "tax1"
