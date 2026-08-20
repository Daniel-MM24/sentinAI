# Central Bank of Kenya AML/CFF Guidelines — Mobile Money Provisions

## 1. Regulatory Authority and Scope

The Central Bank of Kenya (CBK), under the Proceeds of Crime and Anti-Money Laundering Act (POCAMLA, 2009, as amended 2023) and the CBK Prudential Guideline 43 (PG/43), mandates that all payment service providers (PSPs) operating mobile money rails — including but not limited to M-Pesa (Safaricom), Airtel Money, T-Kash (Telkom Kenya), and Equitel (Equity Bank) — enforce tiered Anti-Money Laundering (AML) and Counter-Terrorist Financing (CFT) controls on all electronic wallet transactions.

PG/43 specifically classifies mobile money as a "Designated Non-Financial Business and Profession" (DNFBP) channel for the purposes of suspicious transaction reporting (STR) and currency transaction reporting (CTR).

## 2. Transactional Limits — CBK Mobile Money Directive (July 2022, updated January 2025)

### 2.1 Per-Transaction Ceiling

| Tier | KYC Level | Max Single Transaction (KES) | Max Daily Cumulative (KES) | Max Wallet Balance (KES) |
|------|-----------|------------------------------|----------------------------|--------------------------|
| 1    | Basic (No ID — SIM registration only) | 10,000 | 25,000 | 50,000 |
| 2    | Interim (Self-declared details) | 50,000 | 100,000 | 200,000 |
| 3    | Full KYC (National ID + Proof of Address) | 150,000 | 500,000 | 1,000,000 |
| 4    | Enhanced Due Diligence (EDD — Corporate/Institutional) | 500,000 | No daily cap (transaction velocity monitored) | No balance cap |

**Effective Regulatory Thresholds (directly enforceable by CBK):**
- Maximum per-transaction for a Tier-3 (retail) wallet: **KES 150,000**
- Maximum daily cumulative turnover for a Tier-3 wallet: **KES 500,000**
- Aggregated monthly velocity trigger: **KES 6,000,000** (any wallet exceeding this in a 30-day rolling window triggers automatic CBK reporting from the PSP)

### 2.2 Special Provisions for Agent Float and Liquidity Management

Mobile money agents (retail cash-in/cash-out points) are subject to a separate set of limits:
- Maximum agent-to-agent float transfer: KES 500,000 per transaction
- Agent daily aggregate cash-out: KES 1,000,000
- Agent e-float balance cap: KES 3,000,000
- Any agent whose cumulative daily float volume exceeds KES 2,000,000 must be flagged for suspicious activity review within 24 hours.

## 3. AML/CFT Transaction Monitoring Thresholds

### 3.1 Mandatory Reporting Triggers

Under CBK Guideline PG/43 Section 7 and POCAMLA Section 44, PSPs must file the following reports to the Financial Reporting Centre (FRC):

**Currency Transaction Report (CTR):** Any single transaction exceeding KES 1,000,000, or any series of related transactions aggregating to KES 1,000,000 or more within a single business day.

**Suspicious Transaction Report (STR):** Any transaction (regardless of amount) where the PSP has reasonable grounds to suspect:
- The funds are proceeds of a crime
- The transaction is structured to avoid CTR/STR thresholds
- The transaction has no apparent economic or lawful purpose
- The transaction involves a politically exposed person (PEP)
- The transaction involves a sanctioned entity or jurisdiction

**Threshold for enhanced monitoring (KES 100,000):** All transactions above KES 100,000 must be subject to manual review or automated scoring. This is the "Velocity Warning Trigger" — a turnover of KES 100,000 or more in a single day creates a presumption of enhanced risk and requires the PSP's AML system to log, score, and flag the account for review within 4 hours.

### 3.2 Know Your Customer (KYC) Enhanced Due Diligence Triggers

Enhanced Due Diligence (EDD) is mandatory when:
- A customer's aggregate monthly transactions exceed KES 6,000,000
- A customer transfers or receives more than KES 500,000 in a single transaction
- A customer conducts more than 20 transactions in a single calendar day (behavioral anomaly)
- A customer's geographic transaction pattern spans more than 3 counties within a 12-hour window
- A new account receives a first transaction exceeding KES 50,000 (must trigger intermediate KYC verification within 24 hours)

## 4. Mobile Money Money Laundering Typologies

### 4.1 Structuring (Smurfing)

**Definition:** Structuring, also known as smurfing, is the deliberate splitting of a single asset pool into multiple, smaller transfers each below the regulatory reporting threshold (KES 150,000) to evade CBK AML/CFT detection mechanisms.

**Operational Patterns in Kenyan Mobile Money:**

| Pattern ID | Description | Detection Signal |
|------------|-------------|-----------------|
| STRUC-001 | A single source wallet sends KES 140,000–149,000 to multiple distinct recipient wallets within a 60-minute window, where the aggregate sum exceeds KES 1,000,000 | Rapid-fire disbursements from a single MSISDN to 7+ distinct MSISDNs in ≤60 min |
| STRUC-002 | Multiple distinct sender wallets, all registered within the last 7 days, each send KES 140,000–149,999 to the same recipient wallet within a 2-hour window | Wave of new MSISDNs (age <7 days) converging on a single recipient with uniform amounts within a tight temporal window |
| STRUC-003 | Consecutive transactions of KES 140,000–149,999 between the same two wallets, separated by 45–90 minutes (i.e., the "cooling off" pattern), repeating 3+ times within 12 hours | Same payer-payee pair, amounts at 90–98% of the per-transaction ceiling, with temporal spacing suggesting deliberate evasion |
| STRUC-004 | Round-number deposits of KES 100,000 into an M-Pesa wallet from 10+ distinct agent cash-in points within a single 4-hour window, followed by a single KES 900,000 withdrawal | Multiple cash-in points (agents), each at the Enhanced Monitoring Threshold (KES 100,000), aggregated into a single beneficiary who then consolidates and extracts |
| STRUC-005 | Layering via P2P: Funds move from Wallet A to Wallet B to Wallet C to Wallet D, each transfer at KES 145,000, all occurring within a 90-minute window, with Wallet D immediately transferring to a bank account | Chain length ≥3 hops, each at 95%+ of the per-transaction cap, sub-90-minute end-to-end duration, terminating at a bank integration point |

### 4.2 Cuckoo Smurfing

A sophisticated typology where illicit funds are mixed with legitimate M-Pesa transfers. The criminal deposits illicit cash into a mobile money agent, then uses the agent's float to send a legitimate-looking P2P transfer to an unsuspecting third party who has received a separate legitimate request. Detection relies on correlating agent float inflation patterns with unexplained P2P spikes.

**Detection Signal:** An agent whose float increases by more than KES 500,000 within a 3-hour window and who simultaneously initiates more than 10 unique P2P transfers of KES 50,000 or more each. Flagged as Cuckoo_Smurfing_Risk if the ratio of inbound cash-in to outbound P2P exceeds 4:1 in any 6-hour window.

### 4.3 Trade-Based Money Laundering (TBML) via Merchant Payments

Merchants accepting M-Pesa payments can be used to layer funds through fake goods or services:
- Inflated merchant payment of KES 100,000–149,000 for goods with no verifiable inventory movement
- Circular payments: Merchant A → Customer → Merchant B → Merchant A, creating artificial turnover (KES 500,000+ cycle, repeated 3+ times in 7 days)

### 4.4 Ghost Agent Arbitrage

An agent registered with falsified documentation who operates above float caps and conducts cash-out services without CDD. Key indicator: an agent whose daily float volume exceeds KES 2,000,000 and whose transaction-to-customer ratio exceeds 50:1 (i.e., 50+ distinct customers per day).

## 5. Mandatory Analytical Actions for Transaction Monitoring Systems

Any AML transaction monitoring system operating on Kenyan mobile money rails must implement the following analytical actions:

### 5.1 Threshold Breach Detection
```
WHEN (transaction_amount > KES 150,000) OR (daily_cumulative > KES 500,000)
THEN AUTO_FLAG = "THRESHOLD_BREACH"
  ACTION: Lock wallet, block outbound P2P, escalate to AML officer within 2 hours
  NOTIFY: FRC within 7 calendar days if ≥KES 1,000,000 aggregate
```

### 5.2 Velocity Anomaly Detection
```
WHEN (txn_count_per_day > 20) OR (daily_aggregate_turnover > KES 100,000)
THEN AUTO_FLAG = "VELOCITY_WARNING"
  ACTION: Score account (1-100 risk score), queue for manual review
  ESCALATE: If risk score > 75, notify AML officer within 4 hours
```

### 5.3 Structuring Detection (Pattern-Based)
```
WHEN (amount BETWEEN KES 140,000 AND KES 149,999) AND 
     (same_sender_dest_same_day_count >= 3) AND 
     (aggregate_12h > KES 500,000)
THEN AUTO_FLAG = "STRUCTURING_SUSPECTED"
  ACTION: Freeze outbound txns >KES 10,000, generate STR draft
  CORRELATE: check receiver clustering (shared IMEI, geographic, temporal)
  NOTIFY: FRC STR within 7 calendar days
```

### 5.4 New Account Rapid Escalation Detection
```
WHEN (account_age < 7 days) AND (cumulative_volume > KES 500,000)
THEN AUTO_FLAG = "RAPID_ESCALATION"
  ACTION: Freeze wallet, demand EDD documentation within 48 hours
  IF (no_EDD_response after 48h): PERMANENT_FREEZE + STR filing
```

### 5.5 Agent Float Anomaly Detection
```
WHEN (agent_float_growth > KES 500,000 IN 3h) AND 
     (outbound_p2p_count > 10 AND each >= KES 50,000)
THEN AUTO_FLAG = "CUCKOO_SMURFING_RISK"
  ACTION: Block agent float top-up, trigger CDD reverification within 2 hours
```

### 5.6 Geographic Anomaly Detection
```
WHEN (sender_txn_originates_from > 3 counties IN 12h)
THEN AUTO_FLAG = "GEOGRAPHIC_ANOMALY"
  ACTION: Score as high-risk, require biometric re-verification before next txn
```

### 5.7 Round-Number Threshold Testing
```
WHEN (amount = KES 100,000) OR (amount = KES 140,000) OR (amount = KES 149,999)
  AND (frequency_per_week >= 5)
THEN AUTO_FLAG = "THRESHOLD_TESTING"
  ACTION: Flag for manual review, review historical pattern across 90-day lookback
```

## 6. Record-Keeping and Audit Requirements

PSPs must retain all transaction records for a minimum of **7 years** from the date of the transaction (POCAMLA Section 48). For flagged/suspicious transactions, the retention period extends to **10 years**.

Required fields for all transaction logs (minimal schema):
- Transaction ID, timestamp (ISO 8601), amount (KES), currency (KES)
- Sender MSISDN, Sender wallet tier, Sender ID verification method
- Receiver MSISDN, Receiver wallet tier
- Agent ID (if applicable), Agent geo-location (county + GPS)
- Transaction type (P2P, Cash-In, Cash-Out, Airtime, Merchant Payment, Bank Transfer, Float Transfer)
- Device IMEI, IP address (if OTT/APP), SIM registration match status
- Regulatory flag status (None / THRESHOLD_BREACH / VELOCITY_WARNING / STRUCTURING_SUSPECTED / PEP / SANCTION)

## 7. Penalties for Non-Compliance

CBK penalties for PSPs that fail to enforce AML limits or file reports:
- Tier 1 violation (failure to file CTR/STR): Fine of KES 5,000,000 per incident, plus suspension of PSP license for first repeat offense
- Tier 2 violation (inadequate KYC/EDD controls): Fine of KES 2,000,000 per incident, mandatory remedial action plan within 30 days
- Tier 3 violation (record-keeping failure): Fine of KES 1,000,000 per incident
- Individual penalties: Responsible persons face imprisonment of 3–14 years under POCAMLA Section 16 for knowingly facilitating money laundering

---

*Document compiled from CBK Prudential Guideline PG/43 (rev. 2025), POCAMLA (as amended 2023), CBK Mobile Money Directive July 2022/January 2025 update, and FRC Reporting Guidelines (2024). For authoritative text, refer to official CBK publications.*
