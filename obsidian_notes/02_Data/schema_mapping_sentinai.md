# Schema Mapping — SentinAI

SentinAI maps transaction data fields to CBK-mandated SAR schema requirements.

## Core Field Mappings
| Transaction Field | CBK SAR Field | Description |
|---|---|---|
| sender_id | Subject ID | Unique identifier of transacting party |
| amount_value | Transaction Amount | Value of the mobile money transfer |
| timestamp | Transaction Date/Time | When the transaction occurred |
| sender_region | Geographic Zone | Region of transaction origin |
| receiver_id | Counterparty ID | Receiving party identifier |
| transaction_type | Product Type | M-Pesa product classification |

## CBK Compliance Fields
- Suspicion rationale (free text)
- Risk scoring tier (Low/Medium/High/Critical)
- Linked transactions (cross-references)
- Officer review notes
