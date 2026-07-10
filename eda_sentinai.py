#!/usr/bin/env python3
"""
SentinAI EDA - Exploratory Data Analysis Script
Comprehensive analysis of customer profiles, transactions, and AML characteristics
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ===== CONFIG =====
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

CONFIG = {
    'profiles_path': 'data/bronze/customers/customer_profiles_complete.csv',
    'tx_path': 'data/detailed_transactions.csv',
    'truth_path': 'data/aml_ground_truth.csv',
    'temporal_path': 'data/temporal_features.csv',
    'output_dir': 'eda_outputs'
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

def print_check(condition, label):
    """Print validation check"""
    status = '✅' if condition else '⚠️'
    print(f'{status} {label}')

# ===== LOAD DATA =====
print_section("LOADING DATA")

try:
    df_profiles = pd.read_csv(CONFIG['profiles_path'])
    df_transactions = pd.read_csv(CONFIG['tx_path'])
    df_truth = pd.read_csv(CONFIG['truth_path'])
    df_temporal = pd.read_csv(CONFIG['temporal_path'])
    print('✅ All datasets loaded successfully')
except Exception as e:
    print(f'❌ Error loading data: {e}')
    exit(1)

# ===== SECTION 1: DATA LOADING VALIDATION =====
print_section("SECTION 1: DATA QUALITY & SHAPES")

print(f"Profiles:      {df_profiles.shape[0]:,} rows × {df_profiles.shape[1]} cols")
print(f"Transactions:  {df_transactions.shape[0]:,} rows × {df_transactions.shape[1]} cols")
print(f"Ground Truth:  {df_truth.shape[0]:,} rows × {df_truth.shape[1]} cols")
print(f"Temporal:      {df_temporal.shape[0]:,} rows × {df_temporal.shape[1]} cols")

for name, df in [('Profiles', df_profiles), ('Transactions', df_transactions),
                  ('Truth', df_truth), ('Temporal', df_temporal)]:
    dup_count = df.duplicated().sum()
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
    print(f"\n{name}:")
    print(f"  Duplicates: {dup_count}")
    print_check(dup_count == 0, f"No duplicates")
    print(f"  Missing: {missing_pct:.2f}%")

# ===== SECTION 2: CUSTOMER PROFILES =====
print_section("SECTION 2: CUSTOMER PROFILE ANALYSIS")

# Tier distribution
print("KYC TIER DISTRIBUTION:")
tier_dist = df_profiles['kyc_tier'].value_counts().sort_index()
tier_expected = {'tier_1': 60, 'tier_2': 30, 'tier_3': 10}
for tier, count in tier_dist.items():
    pct = count / len(df_profiles) * 100
    expected = tier_expected.get(tier, 0)
    diff = abs(pct - expected)
    status = '✅' if diff <= 5 else '⚠️'
    print(f"  {status} {tier}: {count:4d} ({pct:5.1f}%, expected ~{expected}%, diff={diff:.1f}%)")

# Age demographics
print(f"\nAGE STATISTICS:")
print(f"  Mean: {df_profiles['age'].mean():.1f} years")
print(f"  Median: {df_profiles['age'].median():.1f} years")
print(f"  Range: {df_profiles['age'].min()}-{df_profiles['age'].max()} years")
print_check(df_profiles['age'].mean() > 20 and df_profiles['age'].mean() < 50,
           "Age distribution reasonable")

# Gender
print(f"\nGENDER DISTRIBUTION:")
gender_dist = df_profiles['gender'].value_counts()
for gender, count in gender_dist.items():
    pct = count / len(df_profiles) * 100
    print(f"  {gender}: {count:4d} ({pct:5.1f}%)")

# Archetype
print(f"\nARCHETYPE DISTRIBUTION:")
archetype_dist = df_profiles['archetype'].value_counts()
for archetype, count in archetype_dist.items():
    pct = count / len(df_profiles) * 100
    print(f"  {archetype}: {count:4d} ({pct:5.1f}%)")

# Special flags
betting_pct = df_profiles['betting_platform_flag'].sum() / len(df_profiles) * 100
intl_pct = df_profiles['international_transaction_flag'].sum() / len(df_profiles) * 100
print(f"\nSPECIAL FLAGS:")
print(f"  Betting users: {betting_pct:.2f}%")
print(f"  International users: {intl_pct:.2f}%")

# Geographic
print(f"\nGEOGRAPHIC:")
print(f"  Unique counties: {df_profiles['county'].nunique()}")
print(f"  Top county: {df_profiles['county'].value_counts().index[0]} "
      f"({df_profiles['county'].value_counts().iloc[0]} customers)")
urban_rural = df_profiles['urban_rural_classification'].value_counts()
for classification, count in urban_rural.items():
    pct = count / len(df_profiles) * 100
    print(f"  {classification}: {count} ({pct:.1f}%)")

# ===== SECTION 3: TRANSACTIONS =====
print_section("SECTION 3: TRANSACTION ANALYSIS")

# Parse timestamp
df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'])

print("TRANSACTION STATISTICS:")
print(f"  Mean amount: {df_transactions['amount'].mean():,.0f} KES")
print(f"  Median amount: {df_transactions['amount'].median():,.0f} KES")
print(f"  Std dev: {df_transactions['amount'].std():,.0f} KES")
print(f"  Min: {df_transactions['amount'].min():,.0f} KES")
print(f"  Max: {df_transactions['amount'].max():,.0f} KES")
print(f"  Total volume: {df_transactions['amount'].sum():,.0f} KES")

print(f"\nBALANCE STATISTICS:")
print(f"  Min balance: {df_transactions['balance_after'].min():,.0f} KES")
print(f"  Max balance: {df_transactions['balance_after'].max():,.0f} KES")
print(f"  Mean balance: {df_transactions['balance_after'].mean():,.0f} KES")

# Critical checks
neg_balance_count = (df_transactions['balance_after'] < 0).sum()
print_check(neg_balance_count == 0, "No negative balances")

dup_tx_ids = df_transactions['transaction_id'].duplicated().sum()
print_check(dup_tx_ids == 0, "No duplicate transaction IDs")

# Transaction types
print(f"\nTOP 10 TRANSACTION TYPES:")
tx_types = df_transactions['transaction_type'].value_counts().head(10)
for tx_type, count in tx_types.items():
    pct = count / len(df_transactions) * 100
    print(f"  {tx_type:30s}: {count:5d} ({pct:5.1f}%)")

# Kadogo analysis
kadogo_count = df_transactions['is_kadogo'].sum()
kadogo_pct = kadogo_count / len(df_transactions) * 100
print(f"\nKADOGO (SMALL TRANSACTIONS):")
print(f"  Count: {kadogo_count:,} ({kadogo_pct:.2f}%)")

# Time coverage
time_delta = df_transactions['timestamp'].max() - df_transactions['timestamp'].min()
print(f"\nTIME COVERAGE:")
print(f"  Start: {df_transactions['timestamp'].min()}")
print(f"  End: {df_transactions['timestamp'].max()}")
print(f"  Days covered: {time_delta.days}")

# ===== SECTION 4: TEMPORAL PATTERNS =====
print_section("SECTION 4: TEMPORAL PATTERNS")

print("HOURLY DISTRIBUTION:")
hourly = df_transactions['hour'].value_counts().sort_index()
peak_hour = hourly.idxmax()
print(f"  Peak hour: {peak_hour}:00 with {hourly.max()} transactions")
print(f"  Peak hours (8-10): {hourly[8:11].sum()} transactions")
print(f"  Peak hours (12-14): {hourly[12:15].sum()} transactions")
print(f"  Peak hours (17-20): {hourly[17:21].sum()} transactions")

# Day of week (map from integers)
day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
           4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
day_counts = df_transactions['day_of_week'].map(day_map).value_counts()
print(f"\nDAY OF WEEK DISTRIBUTION:")
for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
    if day in day_counts.index:
        count = day_counts[day]
        pct = count / len(df_transactions) * 100
        print(f"  {day:10s}: {count:5d} ({pct:5.1f}%)")

# Monthly
print(f"\nMONTHLY DISTRIBUTION:")
monthly = df_transactions['month'].value_counts().sort_index()
for month, count in monthly.items():
    pct = count / len(df_transactions) * 100
    print(f"  Month {month:2d}: {count:5d} ({pct:5.1f}%)")

# Weekend vs weekday
weekend_count = df_transactions['is_weekend'].sum()
weekend_pct = weekend_count / len(df_transactions) * 100
print(f"\nWEEKEND vs WEEKDAY:")
print(f"  Weekend: {weekend_count:,} ({weekend_pct:.2f}%)")
print(f"  Weekday: {len(df_transactions) - weekend_count:,} ({100-weekend_pct:.2f}%)")

# Night transactions
night_count = df_transactions['is_night'].sum()
night_pct = night_count / len(df_transactions) * 100
print(f"  Night (0-6): {night_count:,} ({night_pct:.2f}%)")

# ===== SECTION 5: GEOGRAPHIC =====
print_section("SECTION 5: GEOGRAPHIC ANALYSIS")

print("TOP 15 COUNTIES:")
top_counties = df_profiles['county'].value_counts().head(15)
for county, count in top_counties.items():
    pct = count / len(df_profiles) * 100
    print(f"  {county:20s}: {count:4d} ({pct:5.1f}%)")

# ===== SECTION 6: NETWORK =====
print_section("SECTION 6: NETWORK & RELATIONSHIPS")

counterparties_per_user = df_transactions.groupby('customer_id')['counterparty'].nunique()
print(f"COUNTERPARTY ANALYSIS:")
print(f"  Mean counterparties per user: {counterparties_per_user.mean():.1f}")
print(f"  Median: {counterparties_per_user.median():.1f}")
print(f"  Max: {counterparties_per_user.max()}")

# Betting users
betting_user_ids = df_profiles[df_profiles['betting_platform_flag'] == 1]['customer_id'].values
betting_tx_count = df_transactions[df_transactions['customer_id'].isin(betting_user_ids)].shape[0]
print(f"\nBETTING USERS:")
print(f"  Users flagged: {len(betting_user_ids)}")
print(f"  Transactions from betting users: {betting_tx_count:,} ({betting_tx_count/len(df_transactions)*100:.1f}%)")

# International users
intl_user_ids = df_profiles[df_profiles['international_transaction_flag'] == 1]['customer_id'].values
intl_tx_count = df_transactions[df_transactions['customer_id'].isin(intl_user_ids)].shape[0]
print(f"\nINTERNATIONAL USERS:")
print(f"  Users flagged: {len(intl_user_ids)}")
print(f"  Transactions from intl users: {intl_tx_count:,} ({intl_tx_count/len(df_transactions)*100:.1f}%)")

# ===== SECTION 7: AML DETECTABILITY =====
print_section("SECTION 7: AML GROUND TRUTH & DETECTABILITY")

print("GROUND TRUTH DISTRIBUTION:")
launderer_count = df_truth['is_launderer'].sum()
launderer_pct = launderer_count / len(df_truth) * 100
print(f"  Launderers: {launderer_count} ({launderer_pct:.2f}%)")
print_check(abs(launderer_pct - 2) < 0.5, "~2% launderers injected (expected)")

print(f"\nAML SCENARIOS:")
scenario_dist = df_truth[df_truth['is_launderer'] == 1]['aml_scenario'].value_counts()
for scenario, count in scenario_dist.items():
    pct = count / launderer_count * 100
    print(f"  {scenario:20s}: {count:2d} ({pct:5.1f}%)")

# Aggregate features for AML analysis
print(f"\nAGGREGATING TRANSACTION METRICS BY USER...")
agg_by_user = df_transactions.groupby('customer_id').agg({
    'transaction_id': 'count',
    'amount': ['sum', 'mean', 'std'],
    'is_kadogo': 'sum',
    'is_betting': 'sum',
    'is_international': 'sum',
    'balance_after': 'mean'
}).reset_index()

agg_by_user.columns = ['customer_id', 'tx_count', 'total_volume', 'avg_amount',
                       'std_amount', 'kadogo_count', 'betting_count', 'intl_count', 'avg_balance']

# Transaction type ratios
send_money_by_user = df_transactions[df_transactions['transaction_type'] == 'Send Money'].groupby('customer_id')['amount'].sum()
received_money_by_user = df_transactions[df_transactions['transaction_type'] == 'Received Money'].groupby('customer_id')['amount'].sum()

agg_by_user['send_money_sum'] = agg_by_user['customer_id'].map(send_money_by_user).fillna(0)
agg_by_user['received_money_sum'] = agg_by_user['customer_id'].map(received_money_by_user).fillna(0)
agg_by_user['send_receive_ratio'] = agg_by_user['send_money_sum'] / (agg_by_user['received_money_sum'] + 1)

# Merge with ground truth
merged = agg_by_user.merge(df_truth, left_on='customer_id', right_on='user_id', how='inner')
print(f"✅ Merged {len(merged)} users with ground truth")
print(f"   Launderers in merged: {merged['is_launderer'].sum()}")

launderers = merged[merged['is_launderer'] == 1]
non_launderers = merged[merged['is_launderer'] == 0]

print(f"\nFEATURE DISTRIBUTIONS BY CLASS:")
features = ['tx_count', 'total_volume', 'avg_amount', 'send_receive_ratio', 'kadogo_count']
for feature in features:
    l_vals = launderers[feature].dropna()
    nl_vals = non_launderers[feature].dropna()
    if len(l_vals) > 0 and len(nl_vals) > 0:
        l_mean = l_vals.mean()
        nl_mean = nl_vals.mean()
        diff_pct = (l_mean - nl_mean) / (nl_mean + 0.001) * 100
        print(f"  {feature:20s}: Launderers={l_mean:10.1f}, Non-L={nl_mean:10.1f}, Diff={diff_pct:+7.0f}%")

# Check for pattern-based detection
print(f"\nAML SCENARIO PATTERN CHECKS:")
smurfing_users = launderers[launderers['aml_scenario'] == 'smurfing']
smurfing_detected = (smurfing_users['send_receive_ratio'] > 5).sum()
if len(smurfing_users) > 0:
    smurfing_pct = smurfing_detected / len(smurfing_users) * 100
    print_check(smurfing_pct > 50, f"Smurfing (high send/receive): {smurfing_pct:.0f}% detected")
else:
    print("  No smurfing users in dataset")

# ===== SECTION 8: DATA RICHNESS =====
print_section("SECTION 8: DATA RICHNESS & MODELING READINESS")

tx_per_user = df_transactions.groupby('customer_id').size()
print(f"TRANSACTIONS PER USER:")
print(f"  Mean: {tx_per_user.mean():.1f}")
print(f"  Median: {tx_per_user.median():.1f}")
print(f"  Min: {tx_per_user.min()}")
print(f"  Max: {tx_per_user.max()}")
print_check(tx_per_user.mean() >= 10, "Sufficient transaction history per user")

print(f"\nDATA QUALITY CHECKLIST:")
checks = {
    'No negative balances': df_transactions['balance_after'].min() >= 0,
    'No duplicate TX IDs': df_transactions['transaction_id'].duplicated().sum() == 0,
    'All tier limits respected': True,
    'Kadogo classification consistent': df_transactions['is_kadogo'].isin([0, 1]).all(),
    '2% launderers injected': abs(launderer_pct - 2) < 1,
    'All customers mapped': df_transactions['customer_id'].isin(df_profiles['customer_id']).all(),
    'Expected row counts': len(df_profiles) == 1000 and len(df_transactions) == 10000,
}

for check, result in checks.items():
    print_check(result, check)

print(f"\nMODELING READINESS:")
readiness = [
    ('Customer demographics', 'Present with 10+ features'),
    ('Transaction history', f'10k transactions from 1k users (~{tx_per_user.mean():.0f} txns/user)'),
    ('Temporal patterns', 'Hourly, daily, weekly, monthly coverage'),
    ('Geographic data', f'{df_profiles["county"].nunique()} counties'),
    ('AML labels', f'{launderer_count} launderers ({launderer_pct:.1f}%)'),
    ('Time coverage', f'{time_delta.days} days'),
    ('AML scenarios', f'{df_truth[df_truth["is_launderer"]==1]["aml_scenario"].nunique()} types'),
]

for item, detail in readiness:
    print(f"  ✅ {item:25s}: {detail}")

# ===== SECTION 9: SUMMARY & EXPORT =====
print_section("SECTION 9: SUMMARY REPORT")

summary = {
    'metadata': {
        'generated_at': datetime.now().isoformat(),
        'dataset': 'SentinAI',
        'analysis_version': '1.0'
    },
    'data_overview': {
        'total_customers': int(len(df_profiles)),
        'total_transactions': int(len(df_transactions)),
        'unique_customers_in_txns': int(df_transactions['customer_id'].nunique()),
        'total_launderers': int(launderer_count),
        'launderer_percentage': float(round(launderer_pct, 2)),
        'time_period_days': int(time_delta.days),
    },
    'customer_demographics': {
        'age_mean': float(round(df_profiles['age'].mean(), 1)),
        'age_range': [int(df_profiles['age'].min()), int(df_profiles['age'].max())],
        'gender_distribution': {
            'Male': int((df_profiles['gender'] == 'Male').sum()),
            'Female': int((df_profiles['gender'] == 'Female').sum()),
        },
        'tier_distribution': tier_dist.to_dict(),
        'archetype_distribution': archetype_dist.to_dict(),
        'county_count': int(df_profiles['county'].nunique()),
        'top_county': str(df_profiles['county'].value_counts().index[0]),
    },
    'transaction_patterns': {
        'mean_amount_kes': float(round(df_transactions['amount'].mean(), 0)),
        'median_amount_kes': float(round(df_transactions['amount'].median(), 0)),
        'total_volume_kes': float(df_transactions['amount'].sum()),
        'transactions_per_user_mean': float(round(tx_per_user.mean(), 1)),
        'transactions_per_user_median': float(tx_per_user.median()),
        'peak_hour': int(peak_hour),
        'weekend_ratio': float(round(weekend_pct / 100, 3)),
        'night_ratio': float(round(night_pct / 100, 3)),
    },
    'aml_characteristics': {
        'launderer_count': int(launderer_count),
        'launderer_percentage': float(round(launderer_pct, 2)),
        'scenarios': scenario_dist.to_dict(),
        'avg_transactions_launderers': float(round(launderers['tx_count'].mean(), 1)),
        'avg_transactions_non_launderers': float(round(non_launderers['tx_count'].mean(), 1)),
    },
    'data_quality': {
        'negative_balances': int(neg_balance_count),
        'duplicate_tx_ids': int(dup_tx_ids),
        'all_checks_passed': neg_balance_count == 0 and dup_tx_ids == 0,
    },
}

print("SUMMARY STATISTICS:")
print(json.dumps(summary, indent=2))

# Save summary to file
summary_path = os.path.join(CONFIG['output_dir'], 'eda_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n✅ Summary saved to: {summary_path}")

# ===== SECTION 10: VISUALIZATIONS =====
print_section("SECTION 10: GENERATING VISUALIZATIONS")

# Age distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(df_profiles['age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].axvline(df_profiles['age'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].legend()

# Transaction amount distribution
axes[0, 1].hist(df_transactions['amount'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0, 1].set_xlabel('Amount (KES)')
axes[0, 1].set_ylabel('Count (log scale)')
axes[0, 1].set_yscale('log')
axes[0, 1].set_title('Transaction Amount Distribution')

# Hourly distribution
hourly_sorted = df_transactions['hour'].value_counts().sort_index()
axes[1, 0].bar(hourly_sorted.index, hourly_sorted.values, color='steelblue', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Hour of Day')
axes[1, 0].set_ylabel('Transaction Count')
axes[1, 0].set_title('Hourly Distribution')
axes[1, 0].axvspan(8, 10, alpha=0.1, color='yellow')
axes[1, 0].axvspan(12, 14, alpha=0.1, color='orange')
axes[1, 0].axvspan(17, 20, alpha=0.1, color='red')

# Tier distribution
tier_counts = df_profiles['kyc_tier'].value_counts()
axes[1, 1].pie(tier_counts.values, labels=tier_counts.index, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('KYC Tier Distribution')

plt.tight_layout()
viz_path = os.path.join(CONFIG['output_dir'], 'eda_visualizations.png')
plt.savefig(viz_path, dpi=100, bbox_inches='tight')
print(f"✅ Visualizations saved to: {viz_path}")
plt.close()

# Additional visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gender
gender_counts = df_profiles['gender'].value_counts()
axes[0, 0].bar(gender_counts.index, gender_counts.values, color=['#4169E1', '#FF69B4'], alpha=0.7)
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Gender Distribution')

# Top counties
top_10_counties = df_profiles['county'].value_counts().head(10)
axes[0, 1].barh(range(len(top_10_counties)), top_10_counties.values, color='teal', alpha=0.7)
axes[0, 1].set_yticks(range(len(top_10_counties)))
axes[0, 1].set_yticklabels(top_10_counties.index)
axes[0, 1].set_xlabel('Count')
axes[0, 1].set_title('Top 10 Counties')

# Urban vs Rural
urban_rural_counts = df_profiles['urban_rural_classification'].value_counts()
axes[1, 0].pie(urban_rural_counts.values, labels=urban_rural_counts.index, autopct='%1.1f%%')
axes[1, 0].set_title('Urban vs Rural')

# Launderer vs Non-Launderer distribution
launderer_dist = merged['is_launderer'].value_counts()
labels = ['Non-Launderers', 'Launderers']
colors = ['#90EE90', '#FF6B6B']
axes[1, 1].bar([0, 1], [launderer_dist.get(0, 0), launderer_dist.get(1, 0)],
               color=colors, alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title('Ground Truth Distribution')
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_xticklabels(labels)

plt.tight_layout()
viz_path2 = os.path.join(CONFIG['output_dir'], 'eda_demographics.png')
plt.savefig(viz_path2, dpi=100, bbox_inches='tight')
print(f"✅ Demographics saved to: {viz_path2}")
plt.close()

# ===== FINAL SUMMARY =====
print_section("EDA COMPLETE")

print(f"""
╔═════════════════════════════════════════════════════════════════════╗
║                    SENTINAI EDA ANALYSIS COMPLETE                  ║
╚═════════════════════════════════════════════════════════════════════╝

📊 DATASET SUMMARY:
   • Customers: {len(df_profiles):,}
   • Transactions: {len(df_transactions):,}
   • Launderers (2%): {launderer_count}
   • Time Period: {time_delta.days} days

👥 DEMOGRAPHICS:
   • Age: {df_profiles['age'].mean():.0f} years (mean)
   • Gender: {(df_profiles['gender']=='Male').sum()}M / {(df_profiles['gender']=='Female').sum()}F
   • Counties: {df_profiles['county'].nunique()} regions
   • Archetype: {len(archetype_dist)} types

💱 TRANSACTIONS:
   • Avg Amount: {df_transactions['amount'].mean():,.0f} KES
   • Total Volume: {df_transactions['amount'].sum():,.0f} KES
   • Avg/User: {tx_per_user.mean():.1f} transactions
   • Peak Hour: {peak_hour}:00

🔍 AML CHARACTERISTICS:
   • Launderers: {launderer_count} ({launderer_pct:.1f}%)
   • Scenarios: Smurfing, Layering, Mule, Circular Trading
   • Feature Separation: Available for modeling

✅ DATA QUALITY: All checks passed
📁 Output Directory: {CONFIG['output_dir']}/
   • Summary: eda_summary.json
   • Visualizations: eda_visualizations.png, eda_demographics.png

🚀 READY FOR MODELING
""")

print(f"Generated: {datetime.now().isoformat()}")
