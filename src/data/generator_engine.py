from typing import Literal, Dict, Set, Tuple
from uuid import UUID
from pydantic import BaseModel, Field, EmailStr

class CustomerRecord(BaseModel):
    """
    Immutable Customer Directory Record enforcing structural integrity.
    """
    model_config = {"frozen": True}

    user_id: str
    tax_id: str
    email: EmailStr


class TransactionRecord(BaseModel):
    """
    Immutable Mobile Money Transaction Ledger Record.
    """
    model_config = {"frozen": True}

    transaction_id: UUID
    sender_id: str
    recipient_id: str
    transaction_amount: float = Field(ge=0.0)
    timestamp: str
    channel_type: Literal["USSD", "App", "STK_Push"]
    is_fuliza: bool
    is_mshwari: bool


class SyntheticStateRegistry:
    """
    Stateful Customer Lookup Registry enforcing physical uniqueness 
    of the compound natural primary key tuple (tax_id, email).
    """
    def __init__(self) -> None:
        self._registered_customers: Dict[str, CustomerRecord] = {}
        self._unique_tuples: Set[Tuple[str, str]] = set()

    def register_customer(self, user_id: str, tax_id: str, email: str) -> CustomerRecord:
        """
        Registers a new customer, enforcing constraints on user_id and compound key.
        """
        if user_id in self._registered_customers:
            raise ValueError(f"Duplicate user_id detected: {user_id} cannot be reused.")
            
        compound_key = (tax_id, email)
        if compound_key in self._unique_tuples:
            raise ValueError(f"Duplicate compound key detected: (tax_id={tax_id}, email={email}) already registered.")
            
        record = CustomerRecord(user_id=user_id, tax_id=tax_id, email=email)
        self._registered_customers[user_id] = record
        self._unique_tuples.add(compound_key)
        return record

    def get_customer(self, user_id: str) -> CustomerRecord:
        if user_id not in self._registered_customers:
            raise KeyError(f"Customer with user_id {user_id} not found.")
        return self._registered_customers[user_id]

    @property
    def total_customers(self) -> int:
        return len(self._registered_customers)

    def is_registered(self, user_id: str) -> bool:
        return user_id in self._registered_customers
