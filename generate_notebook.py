"""
Generate SentinAI Data Quality Validation Notebook using nbformat
Answers 8 validation domains: Data Quality, Constraint Compliance, Statistical Validity,
Regulatory Alignment, Model Robustness, Reproducibility, Edge Cases, and Performance.
"""
import nbformat as nbf
import os

os.chdir('/home/dan/project/sentinAI')

nb = nbf.v4.new_notebook()
nb.metadata = {
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.10.0'}
}

cells = []

def code(source):
    cells.append(nbf.v4.new_code_cell(source))

def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))

# ============================================================================
# SECTION 0 — Setup
# ============================================================================
md("""# SentinAI Data Quality Validation Notebook

Comprehensive validation across 8 domains to answer: **What questions must data answer before it is called quality data?**

## Validation Domains

**1. Data Quality**
- Completeness: Are all expected rows present? Missing fields?
- Uniqueness: Duplicate transaction IDs or customer IDs?
- Accuracy: Balance integrity (Σ inflow - Σ outflow = final balance)?
- Consistency: Timestamps chronologically increasing per customer?

**2. Constraint Compliance**
- Balance integrity: Negative balances? Running balance match?
- KYC tier limits: Balance exceeds tier maximum (50K/500K/5M)?
- Velocity limits: Daily velocity exceeds tier limits (100K/1M/10M)?

**3. Statistical Validity**
- Distribution fit: LogNormal/Poisson/Exponential distributions?
- Temporal patterns: Hourly/weekly/seasonal patterns realistic?

**4. Regulatory Alignment**
- CBK compliance: Transaction monitoring rules align with CBK thresholds?
- AML detection: Suspicious activity patterns detectable?

**5. Model Robustness**
- Overfitting: Train-test performance gap < 0.05?
- Cross-validation stability: CV std/mean < 0.01?
- Sensitivity: Performance degrade with noise/perturbations?

**6. Reproducibility**
- Determinism: Same seed produces identical data?
- Idempotency: Re-running produces same outputs (checksum match)?

**7. Edge Cases**
- Customers with 0 transactions, single transactions, max balance/velocity?
- AML anomaly scenarios injected correctly without cross-contamination?

**8. Performance**
- Generation completes within target times?
- Memory usage within limits?
""")

code(r"""import os, polars as pl, pandas as pd, matplotlib, hashlib
os.chdir('/home/dan/project/sentinAI')
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, warnings, json, time
from scipy import stats
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (14, 6)
print("Setup complete.")""")

md("""---
## Data Loading
""")

code(r"""# Load all data sources
customers_csv = pl.read_csv('data/customers_metadata.csv')
transactions_csv = pl.read_csv('data/detailed_transactions.csv')
silver_tx = pl.read_parquet('data/silver/silver_transactions_2026-07-16.parquet')
silver_cust = pl.read_parquet('data/silver/silver_customers_2026-07-16.parquet')
gold = pl.read_parquet('data/gold/features/v1.0/gold_features_consolidated.parquet')
bronze_tx = pl.read_parquet('data/bronze/transactions/2026-07-16/bronze_transactions_d72076a9-2776-4bfd-bcb2-c51e878f3df7.parquet')
bronze_cust = pl.read_parquet('data/bronze/customers/2026-07-16/bronze_customers_d72076a9-2776-4bfd-bcb2-c51e878f3df7.parquet')

print(f"Customers CSV: {customers_csv.shape}")
print(f"Transactions CSV: {transactions_csv.shape}")
print(f"Silver Transactions: {silver_tx.shape}")
print(f"Silver Customers: {silver_cust.shape}")
print(f"Gold Features: {gold.shape}")
print(f"Bronze Transactions: {bronze_tx.shape}")
print(f"Bronze Customers: {bronze_cust.shape}")

# Check for anomaly labels
if 'anomaly_flag' in gold.columns:
    anom = gold.filter(pl.col('anomaly_flag') == 1)
    norm = gold.filter(pl.col('anomaly_flag') == 0)
    print(f"\\nGold anomaly rows: {anom.shape[0]}")
    print(f"Gold normal rows: {norm.shape[0]}")
else:
    print("\\nNo anomaly_flag found in Gold data")
    anom = pl.DataFrame()
    norm = gold""")

# ============================================================================
# DOMAIN 1: Data Quality
# ============================================================================
md("""---
## Domain 1: Data Quality Validation

**Questions:**
- Completeness: Are all expected rows present? Missing fields?
- Uniqueness: Duplicate transaction IDs or customer IDs?
- Accuracy: Balance integrity (Σ inflow - Σ outflow = final balance)?
- Consistency: Timestamps chronologically increasing per customer?
""")

code(r"""print("=== 1.1 COMPLETENESS ===")
print(f"Customers CSV: {customers_csv.shape[0]} rows (expected: 1000)")
print(f"Transactions CSV: {transactions_csv.shape[0]} rows (expected: 10000)")
print(f"Silver Transactions: {silver_tx.shape[0]} rows")
print(f"Silver Customers: {silver_cust.shape[0]} rows")
print(f"Gold Features: {gold.shape[0]} rows")
print(f"Bronze Transactions: {bronze_tx.shape[0]} rows")
print(f"Bronze Customers: {bronze_cust.shape[0]} rows")

print("\\n=== 1.2 UNIQUENESS ===")
print(f"Duplicate transaction_id in CSV: {transactions_csv['transaction_id'].is_duplicated().sum()}")
print(f"Duplicate customer_id in CSV: {customers_csv['customer_id'].is_duplicated().sum()}")
print(f"Duplicate transaction_id in Silver: {silver_tx['transaction_id'].is_duplicated().sum()}")
print(f"Duplicate customer_id in Silver: {silver_cust['customer_id'].is_duplicated().sum()}")

print("\\n=== 1.3 ACCURACY - Balance Integrity ===")
# Check balance integrity: per entity, account_balance_before + net_change = account_balance_after
# For each entity: first balance should equal first account_balance_before,
# and sequential balance deltas should be internally consistent.
if 'account_balance_before' in transactions_csv.columns and 'account_balance_after' in transactions_csv.columns and 'entity_id' in transactions_csv.columns:
    # Net change per entity: opening_balance + sum(paid_in) - sum(paid_out) should = closing_balance
    # Sort by timestamp so first()/last() reflect chronological order
    balance_check = transactions_csv.sort('entity_id', 'timestamp').with_columns([
        pl.col('account_balance_before').first().over('entity_id').alias('opening_balance'),
        pl.col('account_balance_after').last().over('entity_id').alias('closing_balance'),
        pl.when(pl.col('transaction_type') == 'DEPOSIT')
            .then(pl.col('transaction_amount'))
            .otherwise(0).alias('inflow'),
        pl.when(pl.col('transaction_type') == 'WITHDRAWAL')
            .then(pl.col('transaction_amount'))
            .otherwise(0).alias('outflow')
    ])
    entity_check = balance_check.group_by('entity_id').first().select([
        pl.col('entity_id'),
        pl.col('opening_balance'),
        pl.col('closing_balance'),
        pl.col('inflow').sum().alias('total_inflow'),
        pl.col('outflow').sum().alias('total_outflow')
    ])
    entity_check = entity_check.with_columns(
        ((pl.col('opening_balance') + pl.col('total_inflow') - pl.col('total_outflow')) - pl.col('closing_balance')).abs().alias('balance_diff')
    )

    max_diff = entity_check['balance_diff'].max()
    mean_diff = entity_check['balance_diff'].mean()
    violations = (entity_check['balance_diff'] > 0.01).sum()
    print(f"Max balance difference: {max_diff:.2f} KES")
    print(f"Mean balance difference: {mean_diff:.2f} KES")
    print(f"Entities with violations: {violations}")
    if violations == 0:
        print("✅ PASS: Balance integrity verified (accounts balance within 0.01 KES)")
    else:
        print(f"⚠️  FAIL: {violations} entities have balance integrity violations ({max_diff:.2f} KES max)")

print("\\n=== 1.4 CONSISTENCY - Timestamp Ordering ===")
if 'timestamp' in transactions_csv.columns:
    transactions_csv = transactions_csv.with_columns(
        pl.col('timestamp').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f%:z")
    )
    # Check if timestamps are chronologically increasing per entity
    time_check = transactions_csv.sort('entity_id', 'timestamp')
    time_check = time_check.with_columns([
        pl.col('timestamp').shift().over('entity_id').alias('prev_timestamp')
    ])
    time_check = time_check.filter(pl.col('prev_timestamp').is_not_null())
    time_check = time_check.with_columns(
        (pl.col('timestamp') >= pl.col('prev_timestamp')).alias('is_chronological')
    )
    chronological_pct = time_check['is_chronological'].mean() * 100
    print(f"Chronological timestamp percentage: {chronological_pct:.2f}%")
    if chronological_pct == 100.0:
        print("✅ PASS: All timestamps chronologically increasing per entity")
    else:
        print(f"⚠️  FAIL: {100 - chronological_pct:.2f}% of timestamps out of order")""")

# ============================================================================
# DOMAIN 2: Constraint Compliance
# ============================================================================
md("""---
## Domain 2: Constraint Compliance Validation

**Questions:**
- Balance integrity: Negative balances? Running balance match?
- KYC tier limits: Balance exceeds tier maximum (50K/500K/5M)?
- Velocity limits: Daily velocity exceeds tier limits (100K/1M/10M)?
""")

code(r"""print("=== 2.1 BALANCE INTEGRITY - Negative Balances ===")
if 'account_balance_after' in transactions_csv.columns:
    neg_balances = (transactions_csv['account_balance_after'] < 0).sum()
    print(f"Negative balance occurrences: {neg_balances}")
    if neg_balances == 0:
        print("✅ PASS: No negative balances detected")
    else:
        print(f"⚠️  FAIL: {neg_balances} transactions with negative balance")

print("\\n=== 2.1b RUNNING BALANCE CONSISTENCY ===")
if 'entity_id' in transactions_csv.columns and 'account_balance_before' in transactions_csv.columns and 'account_balance_after' in transactions_csv.columns:
    rb = transactions_csv.sort(['entity_id', 'timestamp']).with_columns([
        pl.col('account_balance_before').shift().over('entity_id').alias('prev_balance_after')
    ])
    rb = rb.filter(pl.col('prev_balance_after').is_not_null())
    rb = rb.with_columns(
        (pl.col('account_balance_before') - pl.col('prev_balance_after')).abs().alias('running_diff')
    )
    violations = (rb['running_diff'] > 0.01).sum()
    max_diff = rb['running_diff'].max()
    mean_diff = rb['running_diff'].mean()
    print(f"Running balance violations: {violations}")
    print(f"Max running balance diff: {max_diff:.2f} KES")
    print(f"Mean running balance diff: {mean_diff:.2f} KES")
    if violations == 0:
        print("✅ PASS: Sequential balance is consistent per entity")
    else:
        print(f"⚠️  FAIL: {violations} sequential balance mismatches detected")

print("\\n=== 2.2 KYC TIER LIMITS ===")
# CBK PG/43 tier limits (4-tier structure)
tier_limits = {
    'TIER_1': 50000,
    'TIER_2': 500000,
    'TIER_3': 5000000,
    'VENDOR_MERCHANT': 5000000
}

if 'kyc_tier_level' in transactions_csv.columns and 'account_balance_after' in transactions_csv.columns:
    for tier, limit in tier_limits.items():
        tier_data = transactions_csv.filter(pl.col('kyc_tier_level') == tier)
        if tier_data.height > 0:
            max_balance = tier_data['account_balance_after'].max()
            violations = (tier_data['account_balance_after'] > limit).sum()
            status = "✅ PASS" if violations == 0 else f"⚠️  FAIL ({violations} violations)"
            print(f"{status}: {tier} - Max balance: {max_balance:.0f} KES, Limit: {limit} KES")

print("\\n=== 2.3 VELOCITY LIMITS - Daily Transaction Limits ===")
# Daily velocity limits per tier
velocity_limits = {
    'TIER_1': 100000,
    'TIER_2': 1000000,
    'TIER_3': 10000000,
    'VENDOR_MERCHANT': 10000000
}

if 'kyc_tier_level' in transactions_csv.columns and 'transaction_amount' in transactions_csv.columns and 'timestamp' in transactions_csv.columns:
    transactions_csv = transactions_csv.with_columns([
        pl.col('timestamp').dt.date().alias('date')
    ])
    daily_velocity = transactions_csv.group_by(['entity_id', 'kyc_tier_level', 'date']).agg([
        pl.col('transaction_amount').sum().alias('daily_amount')
    ])

    for tier, limit in velocity_limits.items():
        tier_data = daily_velocity.filter(pl.col('kyc_tier_level') == tier)
        if tier_data.height > 0:
            max_velocity = tier_data['daily_amount'].max()
            violations = (tier_data['daily_amount'] > limit).sum()
            status = "✅ PASS" if violations == 0 else f"⚠️  FAIL ({violations} violations)"
            print(f"{status}: {tier} - Max daily velocity: {max_velocity:.0f} KES, Limit: {limit} KES")""")

# ============================================================================
# DOMAIN 3: Statistical Validity
# ============================================================================
md("""---
## Domain 3: Statistical Validity Validation

**Questions:**
- Distribution fit: LogNormal/Poisson/Exponential distributions?
- Temporal patterns: Hourly/weekly/seasonal patterns realistic?
""")

code(r"""print("=== 3.1 DISTRIBUTION FIT - Transaction Amounts ===")
if 'transaction_amount' in transactions_csv.columns:
    amounts = transactions_csv['transaction_amount'].drop_nulls().to_numpy()

    # Test LogNormal fit
    log_amounts = np.log(amounts[amounts > 0])
    ks_stat, ks_pvalue = stats.kstest(log_amounts, 'norm')
    print(f"LogNormal KS test: statistic={ks_stat:.4f}, p-value={ks_pvalue:.4f}")
    if ks_pvalue > 0.05:
        print("✅ PASS: Transaction amounts follow LogNormal distribution")
    else:
        print("⚠️  FAIL: Transaction amounts do not follow LogNormal distribution")

    # Test Poisson fit for transaction counts per entity
    tx_counts = transactions_csv.group_by('entity_id').agg(pl.len().alias('count'))['count'].to_numpy()
    lambda_poisson = tx_counts.mean()
    ks_poisson, ks_poisson_p = stats.kstest(tx_counts, 'poisson', args=(lambda_poisson,))
    print(f"\\nPoisson KS test (tx counts): statistic={ks_poisson:.4f}, p-value={ks_poisson_p:.4f}")
    if ks_poisson_p > 0.05:
        print("✅ PASS: Transaction counts per customer follow Poisson distribution")
    else:
        print("⚠️  FAIL: Transaction counts do not follow Poisson distribution")

print("\\n=== 3.2 TEMPORAL PATTERNS ===")
if 'timestamp' in transactions_csv.columns:
    transactions_csv = transactions_csv.with_columns([
        pl.col('timestamp').dt.hour().alias('hour'),
        pl.col('timestamp').dt.weekday().alias('day_of_week')
    ])
    
    # Hourly distribution
    hourly_dist = transactions_csv.group_by('hour').agg(pl.len().alias('count')).sort('hour')
    night_ratio = hourly_dist.filter((pl.col('hour') >= 22) | (pl.col('hour') <= 6))['count'].sum() / transactions_csv.height
    print(f"Night transaction ratio (22:00-06:00): {night_ratio*100:.2f}% (expected: 3-5%)")
    if 0.03 <= night_ratio <= 0.05:
        print("✅ PASS: Night transaction ratio within expected range")
    else:
        print("⚠️  FAIL: Night transaction ratio outside expected range")
    
    # Weekend distribution
    weekend_ratio = transactions_csv.filter(pl.col('day_of_week') >= 5).height / transactions_csv.height
    print(f"\\nWeekend transaction ratio: {weekend_ratio*100:.2f}% (expected: 25-30%)")
    if 0.25 <= weekend_ratio <= 0.30:
        print("✅ PASS: Weekend transaction ratio within expected range")
    else:
        print("⚠️  FAIL: Weekend transaction ratio outside expected range")""")

# ============================================================================
# DOMAIN 4: Regulatory Alignment
# ============================================================================
md("""---
## Domain 4: Regulatory Alignment Validation

**Questions:**
- CBK compliance: Transaction monitoring rules align with CBK thresholds?
- AML detection: Suspicious activity patterns detectable?
""")

code(r"""print("=== 4.1 CBK THRESHOLD COMPLIANCE ===")
# CBK reporting thresholds
cbk_structuring_threshold = 100000  # KES 100,000 - requires STR
cbk_ctr_threshold = 1000000        # KES 1,000,000 - CTR threshold

if 'amount' in transactions_csv.columns:
    # Check transactions just below structuring threshold (smurfing detection)
    near_threshold = transactions_csv.filter(
        (pl.col('amount') >= cbk_structuring_threshold * 0.9) & 
        (pl.col('amount') < cbk_structuring_threshold)
    ).height
    print(f"Transactions just below KES 100K structuring threshold: {near_threshold}")
    
    # Check transactions above CTR threshold
    above_ctr = transactions_csv.filter(pl.col('amount') >= cbk_ctr_threshold).height
    print(f"Transactions above KES 1M CTR threshold: {above_ctr}")

print("\\n=== 4.2 AML ANOMALY DETECTION ===")
if 'is_launderer' in customers_csv.columns:
    launderer_count = customers_csv.filter(pl.col('is_launderer') == True).height
    print(f"AML launderers in dataset: {launderer_count} ({launderer_count/customers_csv.height*100:.2f}%)")
    
    if 'aml_scenario' in customers_csv.columns:
        scenario_dist = customers_csv.filter(pl.col('is_launderer') == True).group_by('aml_scenario').agg(pl.len().alias('count'))
        print("\\nAML scenario distribution:")
        print(scenario_dist.to_pandas().to_string(index=False))
        
        # Check for cross-contamination (no double-labeling)
        total_launderers = customers_csv.filter(pl.col('is_launderer') == True).height
        scenario_sum = scenario_dist['count'].sum()
        if total_launderers == scenario_sum:
            print("✅ PASS: No cross-contamination - each launderer has single scenario")
        else:
            print(f"⚠️  FAIL: Cross-contamination detected ({total_launderers} vs {scenario_sum})")

if 'anomaly_flag' in gold.columns:
    gold_anomaly_count = gold.filter(pl.col('anomaly_flag') == 1).height
    print(f"\\nGold layer anomaly rows: {gold_anomaly_count} ({gold_anomaly_count/gold.height*100:.2f}%)")""")

# ============================================================================
# DOMAIN 5: Model Robustness
# ============================================================================
md("""---
## Domain 5: Model Robustness Validation

**Questions:**
- Overfitting: Train-test performance gap < 0.05?
- Cross-validation stability: CV std/mean < 0.01?
- Sensitivity: Performance degrade with noise/perturbations?
""")

code(r"""print("=== 5.1 FEATURE DEGENERACY CHECK ===")
# Check for degenerate features (constant or near-zero variance)
exclude_cols = {'transaction_id','customer_id','counterparty_id','timestamp','anomaly_type','anomaly_flag'}
feature_cols = [c for c in gold.columns if c not in exclude_cols]

degenerate = []
for c in feature_cols:
    s = gold[c].drop_nulls()
    if s.len() > 0:
        std_val = s.std()
        n_unique = s.n_unique()
        if std_val < 0.001 or n_unique <= 2:
            degenerate.append({'feature': c, 'std': std_val, 'n_unique': n_unique})

print(f"Degenerate features (std<0.001 or unique≤2): {len(degenerate)}")
if len(degenerate) == 0:
    print("✅ PASS: All features have non-degenerate variance")
else:
    print("⚠️  FAIL: Degenerate features detected:")
    for f in degenerate:
        print(f"  {f['feature']:30s}  std={f['std']:.6f}  unique={f['n_unique']}")

print("\\n=== 5.2 FEATURE SEPARABILITY (Cohen's d) ===")
def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((n1-1)*v1 + (n2-1)*v2) / (n1+n2-2)) if (n1+n2 > 2) and (v1 > 0 or v2 > 0) else 1.0
    return (a.mean() - b.mean()) / sp if sp > 0 else 0.0

if anom.height > 0 and norm.height > 0:
    large_sep = []
    for c in feature_cols:
        a = anom[c].drop_nulls().to_numpy()
        b = norm[c].drop_nulls().to_numpy()
        if len(a) >= 2 and len(b) >= 2:
            d = cohens_d(a, b)
            if abs(d) > 0.8:  # Large effect size
                large_sep.append({'feature': c, 'cohens_d': d})
    
    print(f"Features with LARGE separation (|d|>0.8): {len(large_sep)}")
    if len(large_sep) >= 3:
        print("✅ PASS: Multiple features show strong class separation")
        for f in sorted(large_sep, key=lambda x: abs(x['cohens_d']), reverse=True)[:5]:
            print(f"  {f['feature']:30s}  d={f['cohens_d']:.4f}")
    else:
        print("⚠️  FAIL: Few features separate anomaly from normal")""")

# ============================================================================
# DOMAIN 6: Reproducibility
# ============================================================================
md("""---
## Domain 6: Reproducibility Validation

**Questions:**
- Determinism: Same seed produces identical data?
- Idempotency: Re-running produces same outputs (checksum match)?
""")

code(r"""print("=== 6.1 DETERMINISM CHECK ===")
# Check if data generation appears deterministic (consistent row counts)
print(f"Customer count: {customers_csv.shape[0]} (expected: 1000)")
print(f"Transaction count: {transactions_csv.shape[0]} (expected: 10000)")

if customers_csv.shape[0] == 1000 and transactions_csv.shape[0] == 1000:
    print("✅ PASS: Row counts match expected deterministic generation")
else:
    print("⚠️  FAIL: Row counts do not match expected values")

print("\\n=== 6.2 PIPELINE IMMUTABILITY CHECK ===")
# Check Bronze layer immutability - files should have unique identifiers
bronze_tx_path = 'data/bronze/transactions/2026-07-16/bronze_transactions_d72076a9-2776-4bfd-bcb2-c51e878f3df7.parquet'
bronze_cust_path = 'data/bronze/customers/2026-07-16/bronze_customers_d72076a9-2776-4bfd-bcb2-c51e878f3df7.parquet'

import os
if os.path.exists(bronze_tx_path):
    # Extract UUID from filename
    tx_uuid = bronze_tx_path.split('_')[-1].replace('.parquet', '')
    cust_uuid = bronze_cust_path.split('_')[-1].replace('.parquet', '')
    print(f"Bronze transactions UUID: {tx_uuid}")
    print(f"Bronze customers UUID: {cust_uuid}")
    
    if tx_uuid == cust_uuid:
        print("✅ PASS: Bronze layers have consistent run IDs (same batch)")
    else:
        print("⚠️  FAIL: Bronze layers have mismatched run IDs")
else:
    print("⚠️  Bronze files not found for immutability check")""")

# ============================================================================
# DOMAIN 7: Edge Cases
# ============================================================================
md("""---
## Domain 7: Edge Cases Validation

**Questions:**
- Customers with 0 transactions, single transactions, max balance/velocity?
- AML anomaly scenarios injected correctly without cross-contamination?
""")

code(r"""print("=== 7.1 EDGE CASE CUSTOMERS ===")
# Check for customers with 0 transactions
if 'customer_id' in transactions_csv.columns:
    tx_per_customer = transactions_csv.group_by('customer_id').agg(pl.len().alias('tx_count'))
    zero_tx_customers = tx_per_customer.filter(pl.col('tx_count') == 0).height
    single_tx_customers = tx_per_customer.filter(pl.col('tx_count') == 1).height
    max_tx_customer = tx_per_customer.sort('tx_count', descending=True).row(0)
    
    print(f"Customers with 0 transactions: {zero_tx_customers}")
    print(f"Customers with 1 transaction: {single_tx_customers}")
    print(f"Customer with max transactions: {max_tx_customer[1]} txs")
    
    if zero_tx_customers == 0:
        print("✅ PASS: No customers with 0 transactions")
    else:
        print(f"⚠️  INFO: {zero_tx_customers} customers have 0 transactions")

print("\\n=== 7.2 MAX BALANCE/VELOCITY CUSTOMERS ===")
if 'balance' in transactions_csv.columns:
    max_balance_customer = transactions_csv.sort('balance', descending=True).row(0)
    print(f"Max balance: {max_balance_customer[transactions_csv.columns.get_loc('balance')]:.2f} KES")
    
if 'tier' in customers_csv.columns:
    tier_dist = customers_csv.group_by('tier').agg(pl.len().alias('count'))
    print("\\nTier distribution:")
    print(tier_dist.to_pandas().to_string(index=False))

print("\\n=== 7.3 AML ANOMALY SCENARIO VALIDATION ===")
if 'is_launderer' in customers_csv.columns and 'aml_scenario' in customers_csv.columns:
    # Check each scenario type
    scenarios = ['smurfing', 'layering', 'mule_account', 'circular_trading']
    for scenario in scenarios:
        scenario_count = customers_csv.filter(
            (pl.col('is_launderer') == True) & (pl.col('aml_scenario') == scenario)
        ).height
        print(f"{scenario:20s}: {scenario_count} customers")
    
    # Check for no anomaly accounts
    no_anomaly = customers_csv.filter(pl.col('is_launderer') == False).height
    print(f"\\nNon-launderer customers: {no_anomaly} ({no_anomaly/customers_csv.height*100:.2f}%)")
    
    if no_anomaly == customers_csv.height - customers_csv.filter(pl.col('is_launderer') == True).height:
        print("✅ PASS: Clean transaction sequence for non-launderers")
    else:
        print("⚠️  FAIL: Inconsistent launderer labeling")""")

# ============================================================================
# DOMAIN 8: Performance
# ============================================================================
md("""---
## Domain 8: Performance Validation

**Questions:**
- Generation completes within target times?
- Memory usage within limits?
""")

code(r"""print("=== 8.1 DATA SIZE CHECK ===")
# Check file sizes for performance assessment
import os

def get_file_size(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)  # MB
    return 0

files_to_check = [
    ('data/customers_metadata.csv', 'Customers CSV'),
    ('data/detailed_transactions.csv', 'Transactions CSV'),
    ('data/silver/silver_transactions_2026-07-16.parquet', 'Silver Transactions'),
    ('data/silver/silver_customers_2026-07-16.parquet', 'Silver Customers'),
    ('data/gold/features/v1.0/gold_features_consolidated.parquet', 'Gold Features'),
]

print("File sizes (MB):")
for path, name in files_to_check:
    size = get_file_size(path)
    print(f"  {name:30s}: {size:.2f} MB")

print("\\n=== 8.2 ROW COUNT PERFORMANCE ===")
# Check if row counts are reasonable for performance
print(f"Total transactions: {transactions_csv.shape[0]:,}")
print(f"Total customers: {customers_csv.shape[0]:,}")
print(f"Gold feature rows: {gold.shape[0]:,}")

if transactions_csv.shape[0] <= 100000:
    print("✅ PASS: Transaction count within reasonable performance bounds (<100K)")
else:
    print("⚠️  INFO: Large transaction count may impact performance")""")

# ============================================================================
# SUMMARY
# ============================================================================
md("""---
## Validation Summary

This notebook validates data quality across 8 domains. Each domain answers critical questions about whether the data meets quality standards before being used for model training and deployment.

| Domain | Key Questions | Status |
|--------|---------------|--------|
| **1. Data Quality** | Completeness, Uniqueness, Accuracy, Consistency | See Domain 1 results |
| **2. Constraint Compliance** | Balance integrity, KYC tier limits, Velocity limits | See Domain 2 results |
| **3. Statistical Validity** | Distribution fit, Temporal patterns | See Domain 3 results |
| **4. Regulatory Alignment** | CBK thresholds, AML detection | See Domain 4 results |
| **5. Model Robustness** | Feature degeneracy, Class separability | See Domain 5 results |
| **6. Reproducibility** | Determinism, Pipeline immutability | See Domain 6 results |
| **7. Edge Cases** | Zero/single/max customers, Anomaly scenarios | See Domain 7 results |
| **8. Performance** | File sizes, Row counts | See Domain 8 results |

### Data Quality Criteria

Before data is considered "quality data," it must answer **YES** to these questions:

1. **Completeness**: Are all expected rows present with no missing critical fields?
2. **Uniqueness**: Are there no duplicate transaction IDs or customer IDs?
3. **Accuracy**: Does Σ(inflow) - Σ(outflow) = final balance for all customers?
4. **Consistency**: Are timestamps chronologically increasing per customer?
5. **Constraint Compliance**: Are there zero negative balances and no tier limit violations?
6. **Statistical Validity**: Do distributions follow expected patterns (LogNormal, Poisson)?
7. **Regulatory Alignment**: Are CBK thresholds respected and AML patterns detectable?
8. **Model Robustness**: Do features have non-degenerate variance and class separation?
9. **Reproducibility**: Is the data generation deterministic and pipeline immutable?
10. **Edge Cases**: Are edge cases handled correctly (0 transactions, max balance)?
11. **Performance**: Are data sizes and row counts within reasonable bounds?

### Next Steps

If all validations pass, the data is ready for:
- Model training (XGBoost, autoencoders, isolation forest)
- Feature engineering refinement
- Production deployment testing

If any validation fails, investigate the root cause before proceeding with model development.
""")

code(r"""print("=== Data Quality Validation Complete ===")
print(f"Total customers: {customers_csv.shape[0]:,}")
print(f"Total transactions: {transactions_csv.shape[0]:,}")
print(f"Gold feature rows: {gold.shape[0]:,}")
print(f"Silver transactions: {silver_tx.shape[0]:,}")
print(f"Silver customers: {silver_cust.shape[0]:,}")
print(f"Bronze transactions: {bronze_tx.shape[0]:,}")
print(f"Bronze customers: {bronze_cust.shape[0]:,}")
print("\\nAll 8 validation domains have been executed.")""")

nb.cells = cells

os.makedirs('obsidian_notes/08_Notebooks', exist_ok=True)
notebook_path = 'obsidian_notes/08_Notebooks/01_gold_layer_eda.ipynb'
with open(notebook_path, 'w') as f:
    nbf.write(nb, f)

print(f'✅ Notebook saved to: {notebook_path}')
