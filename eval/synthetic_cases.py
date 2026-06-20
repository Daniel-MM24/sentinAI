"""
Synthetic Audit Cases for Kenyan Financial Crime Compliance Evaluation

This module generates ground-truth test cases for evaluating the agentic RAG system's
ability to identify, cite, and justify compliance-related audit decisions.

Each case includes:
- Scenario: Description of the transaction pattern
- Evidence: Mock transaction logs
- Target Citation: Specific section/clause from Kenyan regulatory corpus
- Expected Reasoning: Logical path the Auditor Agent should follow
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timedelta


class TransactionLog(BaseModel):
    """Mock transaction log entry."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: datetime = Field(..., description="Transaction timestamp")
    amount: float = Field(..., description="Transaction amount in KES")
    transaction_type: str = Field(..., description="Type of transaction")
    source_account: str = Field(..., description="Source account identifier")
    destination_account: str = Field(..., description="Destination account identifier")
    agent_id: str = Field(default="", description="Agent identifier if applicable")
    location: str = Field(default="", description="Transaction location")


class SyntheticAuditCase(BaseModel):
    """Ground-truth audit case for evaluation."""
    case_id: str = Field(..., description="Unique case identifier")
    scenario_category: str = Field(..., description="Category of financial crime")
    description: str = Field(..., description="Detailed scenario description")
    evidence: List[TransactionLog] = Field(..., description="Mock transaction logs")
    target_citation: str = Field(..., description="Target regulatory citation")
    statutory_authority: str = Field(..., description="Governing law/regulation")
    expected_reasoning: List[str] = Field(..., description="Expected logical reasoning steps")
    expected_decision: str = Field(..., description="Expected audit decision")


def generate_synthetic_cases() -> List[SyntheticAuditCase]:
    """
    Generate 10+ synthetic audit cases covering realistic Kenyan financial crime typologies.
    """
    base_time = datetime(2024, 1, 1, 9, 0, 0)
    
    cases = []
    
    # Case 001: Structuring (POCAMLA Regulations 2013)
    case_001 = SyntheticAuditCase(
        case_id="001",
        scenario_category="Structuring",
        description="Customer performs 12 transactions of KES 90,000 each within 24 hours to avoid the KES 1M CTR threshold.",
        evidence=[
            TransactionLog(
                transaction_id=f"TXN_001_{i}",
                timestamp=base_time + timedelta(hours=i*2),
                amount=90000.0,
                transaction_type="CASH_DEPOSIT",
                source_account="ACC_001234",
                destination_account="ACC_001234",
                agent_id=f"AGENT_{100 + i}",
                location="Nairobi"
            )
            for i in range(12)
        ],
        target_citation="POCAMLA Regulations 2013, Section 16(1) - Cash Transaction Reporting",
        statutory_authority="Proceeds of Crime and Anti-Money Laundering Act (POCAMLA) Regulations 2013",
        expected_reasoning=[
            "Identify pattern of multiple transactions below KES 1M threshold",
            "Aggregate transactions within 24-hour period: KES 1,080,000",
            "Determine total exceeds CTR threshold of KES 1M",
            "Conclude structuring behavior to evade reporting requirements",
            "Flag as suspicious transaction requiring STR filing"
        ],
        expected_decision="FILE_STR - Suspicious Transaction Report required for structuring violation"
    )
    cases.append(case_001)
    
    # Case 002: Agent Fraud (CBK Guideline on Agent Banking)
    case_002 = SyntheticAuditCase(
        case_id="002",
        scenario_category="Agent Fraud",
        description="An agent account shows a 400% increase in volume during off-peak hours, with immediate transfers to diverse personal wallets.",
        evidence=[
            TransactionLog(
                transaction_id=f"TXN_002_{i}",
                timestamp=base_time + timedelta(days=i, hours=2),  # Off-peak: 2 AM
                amount=500000.0 * (1 + i*0.1),
                transaction_type="CASH_IN",
                source_account="CASH",
                destination_account="AGENT_999",
                agent_id="AGENT_999",
                location="Mombasa"
            )
            for i in range(5)
        ] + [
            TransactionLog(
                transaction_id=f"TXN_002_TRANSFER_{i}",
                timestamp=base_time + timedelta(days=i, hours=2, minutes=5),
                amount=450000.0 * (1 + i*0.1),
                transaction_type="TRANSFER",
                source_account="AGENT_999",
                destination_account=f"WALLET_{2000 + i}",
                agent_id="AGENT_999",
                location="Mombasa"
            )
            for i in range(5)
        ],
        target_citation="CBK Prudential Guidelines on Agent Banking, Section 7.3 - Agent Monitoring",
        statutory_authority="Central Bank of Kenya (CBK) Guidelines on Agent Banking",
        expected_reasoning=[
            "Detect abnormal volume increase (400%) during off-peak hours (2 AM)",
            "Identify pattern of immediate transfers to diverse personal wallets",
            "Note lack of legitimate business justification for off-peak activity",
            "Recognize potential agent complicity in money laundering",
            "Flag for immediate investigation and potential agent suspension"
        ],
        expected_decision="INVESTIGATE - Immediate agent account freeze and investigation required"
    )
    cases.append(case_002)
    
    # Case 003: Digital Lending (FRC National Risk Assessment)
    case_003 = SyntheticAuditCase(
        case_id="003",
        scenario_category="Digital Lending",
        description="A user borrows M-Shwari funds and immediately transfers them to a known high-risk jurisdiction wallet.",
        evidence=[
            TransactionLog(
                transaction_id="TXN_003_LOAN",
                timestamp=base_time,
                amount=50000.0,
                transaction_type="DIGITAL_LOAN_DISBURSEMENT",
                source_account="M_SHWARI_POOL",
                destination_account="ACC_USER_789",
                agent_id="",
                location="Mobile"
            ),
            TransactionLog(
                transaction_id="TXN_003_TRANSFER",
                timestamp=base_time + timedelta(minutes=2),
                amount=50000.0,
                transaction_type="CROSS_BORDER_TRANSFER",
                source_account="ACC_USER_789",
                destination_account="OFFSHORE_HIGH_RISK_001",
                agent_id="",
                location="Mobile"
            )
        ],
        target_citation="FRC National Risk Assessment 2022, Section 4.2 - Digital Lending Risks",
        statutory_authority="Financial Reporting Centre (FRC) National Risk Assessment",
        expected_reasoning=[
            "Identify digital loan disbursement from M-Shwari",
            "Detect immediate transfer (2 minutes) to high-risk jurisdiction",
            "Cross-reference with FRC high-risk jurisdiction list",
            "Recognize potential loan fraud or terrorist financing",
            "Flag for enhanced due diligence and STR filing"
        ],
        expected_decision="FILE_STR - High-risk cross-border transfer requires immediate STR"
    )
    cases.append(case_003)
    
    # Case 004: Agent Collusion (Rapid cash-in/cash-out)
    case_004 = SyntheticAuditCase(
        case_id="004",
        scenario_category="Agent Collusion",
        description="Rapid cash-in/cash-out at single agent point within 30 minutes, indicating potential layering.",
        evidence=[
            TransactionLog(
                transaction_id=f"TXN_004_IN_{i}",
                timestamp=base_time + timedelta(minutes=i*5),
                amount=200000.0,
                transaction_type="CASH_IN",
                source_account="CASH",
                destination_account=f"ACC_TEMP_{i}",
                agent_id="AGENT_555",
                location="Kisumu"
            )
            for i in range(4)
        ] + [
            TransactionLog(
                transaction_id=f"TXN_004_OUT_{i}",
                timestamp=base_time + timedelta(minutes=i*5 + 2),
                amount=195000.0,
                transaction_type="CASH_OUT",
                source_account=f"ACC_TEMP_{i}",
                destination_account="CASH",
                agent_id="AGENT_555",
                location="Kisumu"
            )
            for i in range(4)
        ],
        target_citation="POCAMLA Section 44 - Obligations of Reporting Institutions",
        statutory_authority="Proceeds of Crime and Anti-Money Laundering Act (POCAMLA)",
        expected_reasoning=[
            "Detect pattern of rapid cash-in/cash-out at single agent",
            "Calculate total volume: KES 800,000 in 30 minutes",
            "Identify 2.5% fee consistent with money laundering commission",
            "Note use of temporary accounts for layering",
            "Flag as potential agent collusion in layering scheme"
        ],
        expected_decision="FILE_STR - Agent collusion suspected, immediate investigation required"
    )
    cases.append(case_004)
    
    # Case 005: Digital Credit Misuse (Fuliza/M-Shwari layering)
    case_005 = SyntheticAuditCase(
        case_id="005",
        scenario_category="Digital Credit Misuse",
        description="Rapid turnover of Fuliza loans with immediate transfers to multiple recipients for layering.",
        evidence=[
            TransactionLog(
                transaction_id=f"TXN_005_LOAN_{i}",
                timestamp=base_time + timedelta(hours=i*6),
                amount=10000.0,
                transaction_type="FULIZA_DISBURSEMENT",
                source_account="FULIZA_POOL",
                destination_account="ACC_USER_456",
                agent_id="",
                location="Mobile"
            )
            for i in range(6)
        ] + [
            TransactionLog(
                transaction_id=f"TXN_005_TRANSFER_{i}",
                timestamp=base_time + timedelta(hours=i*6, minutes=10),
                amount=9500.0,
                transaction_type="TRANSFER",
                source_account="ACC_USER_456",
                destination_account=f"ACC_RECIPIENT_{3000 + i}",
                agent_id="",
                location="Mobile"
            )
            for i in range(6)
        ],
        target_citation="CBK Regulation on Digital Credit, Section 12 - Responsible Lending",
        statutory_authority="Central Bank of Kenya (CBK) Regulation on Digital Credit",
        expected_reasoning=[
            "Identify pattern of rapid Fuliza loan usage (6 loans in 36 hours)",
            "Detect immediate transfers to multiple recipients",
            "Calculate total exposure: KES 60,000 exceeding typical limits",
            "Recognize layering pattern using digital credit facilities",
            "Flag for credit risk assessment and potential STR"
        ],
        expected_decision="MONITOR - Enhanced monitoring and credit limit review required"
    )
    cases.append(case_005)
    
    # Case 006: Trade-Based Money Laundering
    case_006 = SyntheticAuditCase(
        case_id="006",
        scenario_category="Trade-Based Money Laundering",
        description="Over-invoicing of agricultural exports with inconsistent pricing patterns.",
        evidence=[
            TransactionLog(
                transaction_id=f"TXN_006_{i}",
                timestamp=base_time + timedelta(days=i),
                amount=15000000.0 * (1.5 if i % 2 == 0 else 0.8),
                transaction_type="TRADE_PAYMENT",
                source_account="BUYER_OVERSEAS",
                destination_account="EXPORTER_KENYA",
                agent_id="",
                location="International"
            )
            for i in range(5)
        ],
        target_citation="POCAMLA Section 23 - Trade-Based Money Laundering",
        statutory_authority="Proceeds of Crime and Anti-Money Laundering Act (POCAMLA)",
        expected_reasoning=[
            "Detect inconsistent pricing in agricultural export transactions",
            "Identify 50% price variance between similar transactions",
            "Cross-reference with market prices for agricultural commodities",
            "Recognize over-invoicing pattern for value transfer",
            "Flag for trade-based money laundering investigation"
        ],
        expected_decision="FILE_STR - Trade-based money laundering suspected, documentation review required"
    )
    cases.append(case_006)
    
    # Case 007: Politically Exposed Person (PEP) Transaction
    case_007 = SyntheticAuditCase(
        case_id="007",
        scenario_category="PEP Transaction",
        description="Large transactions from a PEP family member without apparent legitimate source.",
        evidence=[
            TransactionLog(
                transaction_id=f"TXN_007_{i}",
                timestamp=base_time + timedelta(days=i*7),
                amount=5000000.0,
                transaction_type="WIRE_TRANSFER",
                source_account="PEP_FAMILY_MEMBER",
                destination_account="OFFSHORE_SHELL_001",
                agent_id="",
                location="International"
            )
            for i in range(3)
        ],
        target_citation="CBK AML/CFT Guidelines, Section 9.2 - PEP Enhanced Due Diligence",
        statutory_authority="Central Bank of Kenya (CBK) AML/CFT Guidelines",
        expected_reasoning=[
            "Identify transactions from PEP family member",
            "Note large amounts (KES 5M each) without clear business purpose",
            "Detect transfers to offshore shell company jurisdictions",
            "Recognize potential corruption or bribery proceeds",
            "Flag for enhanced PEP due diligence and STR filing"
        ],
        expected_decision="FILE_STR - PEP-related suspicious activity requires immediate STR"
    )
    cases.append(case_007)
    
    # Case 008: Terrorist Financing Pattern
    case_008 = SyntheticAuditCase(
        case_id="008",
        scenario_category="Terrorist Financing",
        description="Small, regular payments to multiple individuals in high-risk regions.",
        evidence=[
            TransactionLog(
                transaction_id=f"TXN_008_{i}",
                timestamp=base_time + timedelta(days=i*3),
                amount=25000.0,
                transaction_type="TRANSFER",
                source_account="ACC_SOURCE_X",
                destination_account=f"ACC_HIGH_RISK_{i}",
                agent_id="",
                location="Cross-border"
            )
            for i in range(8)
        ],
        target_citation="POCAMLA Section 28 - Terrorist Financing Offenses",
        statutory_authority="Proceeds of Crime and Anti-Money Laundering Act (POCAMLA)",
        expected_reasoning=[
            "Identify pattern of small, regular payments (KES 25K)",
            "Detect multiple recipients in high-risk regions",
            "Note consistent 3-day interval suggesting structured payments",
            "Cross-reference with terrorist financing watchlists",
            "Flag as potential terrorist financing network"
        ],
        expected_decision="FILE_STR - Terrorist financing suspected, immediate law enforcement notification"
    )
    cases.append(case_008)
    
    # Case 009: Shell Company Layering
    case_009 = SyntheticAuditCase(
        case_id="009",
        scenario_category="Shell Company Layering",
        description="Funds moving through multiple shell companies with no apparent business purpose.",
        evidence=[
            TransactionLog(
                transaction_id=f"TXN_009_STEP_{i}",
                timestamp=base_time + timedelta(hours=i*12),
                amount=10000000.0 - (i * 500000.0),
                transaction_type="WIRE_TRANSFER",
                source_account=f"SHELL_{i}",
                destination_account=f"SHELL_{i+1}",
                agent_id="",
                location="International"
            )
            for i in range(5)
        ],
        target_citation="POCAMLA Section 21 - Shell Company Due Diligence",
        statutory_authority="Proceeds of Crime and Anti-Money Laundering Act (POCAMLA)",
        expected_reasoning=[
            "Identify chain of transactions through multiple shell companies",
            "Detect decreasing amounts suggesting fee extraction",
            "Note lack of legitimate business justification",
            "Recognize classic layering technique",
            "Flag for shell company beneficial ownership investigation"
        ],
        expected_decision="FILE_STR - Complex shell company layering requires comprehensive investigation"
    )
    cases.append(case_009)
    
    # Case 010: Cash Intensive Business Smurfing
    case_010 = SyntheticAuditCase(
        case_id="010",
        scenario_category="Cash Intensive Business",
        description="Multiple cash deposits just below reporting threshold at different branches by same business.",
        evidence=[
            TransactionLog(
                transaction_id=f"TXN_010_{i}",
                timestamp=base_time + timedelta(days=i),
                amount=950000.0,
                transaction_type="CASH_DEPOSIT",
                source_account="CASH_BUSINESS_X",
                destination_account="ACC_BUSINESS_X",
                agent_id=f"BRANCH_{i}",
                location=f"Branch_{i}"
            )
            for i in range(10)
        ],
        target_citation="CBK AML Guidelines, Section 6.4 - Cash Intensive Business Monitoring",
        statutory_authority="Central Bank of Kenya (CBK) AML/CFT Guidelines",
        expected_reasoning=[
            "Identify pattern of cash deposits just below KES 1M threshold",
            "Detect same business using multiple branches (smurfing)",
            "Calculate total deposits: KES 9.5M in 10 days",
            "Note inconsistent with typical business cash flow patterns",
            "Flag as cash-intensive business structuring"
        ],
        expected_decision="FILE_STR - Cash structuring by business requires STR and branch audit"
    )
    cases.append(case_010)
    
    # Case 011: Real Estate Money Laundering
    case_011 = SyntheticAuditCase(
        case_id="011",
        scenario_category="Real Estate",
        description="Large cash payments for property purchases with no mortgage financing.",
        evidence=[
            TransactionLog(
                transaction_id="TXN_011_PROPERTY",
                timestamp=base_time,
                amount=25000000.0,
                transaction_type="PROPERTY_PAYMENT",
                source_account="CASH_BUYER",
                destination_account="REAL_ESTATE_DEVELOPER",
                agent_id="",
                location="Nairobi"
            ),
            TransactionLog(
                transaction_id="TXN_011_SOURCE",
                timestamp=base_time - timedelta(days=1),
                amount=25000000.0,
                transaction_type="CASH_WITHDRAWAL",
                source_account="ACC_OFFSHORE",
                destination_account="CASH_BUYER",
                agent_id="",
                location="Nairobi"
            )
        ],
        target_citation="POCAMLA Section 25 - Real Estate Due Diligence",
        statutory_authority="Proceeds of Crime and Anti-Money Laundering Act (POCAMLA)",
        expected_reasoning=[
            "Identify large cash property payment (KES 25M)",
            "Detect recent cash withdrawal from offshore account",
            "Note absence of mortgage financing for large purchase",
            "Recognize potential real estate money laundering",
            "Flag for source of funds investigation"
        ],
        expected_decision="FILE_STR - Large cash property purchase requires source of funds verification"
    )
    cases.append(case_011)
    
    # Case 012: Cryptocurrency Conversion
    case_012 = SyntheticAuditCase(
        case_id="012",
        scenario_category="Cryptocurrency",
        description="Rapid conversion of funds to cryptocurrency with immediate international transfers.",
        evidence=[
            TransactionLog(
                transaction_id="TXN_012_CRYPTO",
                timestamp=base_time,
                amount=8000000.0,
                transaction_type="CRYPTO_PURCHASE",
                source_account="ACC_USER_CRYPTO",
                destination_account="CRYPTO_EXCHANGE",
                agent_id="",
                location="Digital"
            ),
            TransactionLog(
                transaction_id="TXN_012_TRANSFER",
                timestamp=base_time + timedelta(minutes=15),
                amount=7800000.0,
                transaction_type="CRYPTO_TRANSFER",
                source_account="CRYPTO_EXCHANGE",
                destination_account="OFFSHORE_WALLET",
                agent_id="",
                location="Digital"
            )
        ],
        target_citation="CBK Guidance on Virtual Assets, Section 8.1 - VASP Due Diligence",
        statutory_authority="Central Bank of Kenya (CBK) Guidance on Virtual Assets",
        expected_reasoning=[
            "Identify large cryptocurrency purchase (KES 8M)",
            "Detect immediate transfer to offshore wallet",
            "Note lack of KYC on offshore wallet destination",
            "Recognize potential cryptocurrency-based money laundering",
            "Flag for VASP compliance review"
        ],
        expected_decision="FILE_STR - Cryptocurrency conversion to offshore requires STR"
    )
    cases.append(case_012)
    
    return cases


def get_case_by_id(case_id: str) -> SyntheticAuditCase:
    """Retrieve a specific synthetic audit case by ID."""
    cases = generate_synthetic_cases()
    for case in cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Case ID {case_id} not found")


def get_all_cases() -> List[SyntheticAuditCase]:
    """Return all synthetic audit cases."""
    return generate_synthetic_cases()


if __name__ == "__main__":
    # Generate and display all cases
    cases = generate_synthetic_cases()
    print(f"Generated {len(cases)} synthetic audit cases\n")
    
    for case in cases:
        print(f"Case ID: {case.case_id}")
        print(f"Category: {case.scenario_category}")
        print(f"Description: {case.description}")
        print(f"Target Citation: {case.target_citation}")
        print(f"Expected Decision: {case.expected_decision}")
        print(f"Number of Transactions: {len(case.evidence)}")
        print("-" * 80)
