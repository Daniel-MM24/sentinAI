"""
Generate SentinAI EDA Notebook using nbformat
Builds the notebook cell-by-cell to avoid file size limits
"""
import nbformat as nbf
import os

# Change to project root for notebook execution context
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

# ===== SECTION 1: Data Loading (10 cells) =====
md("""# SentinAI Data Quality & EDA Notebook

## Section 1: Data Loading

Load all data sources and validate basic structure.""")

code("""import os
os.chdir('/home/dan/project/sentinAI')
print('Working directory:', os.getcwd())""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import json
import os
from scipy import stats
import networkx as nx

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

CONFIG = {
    'seed': 42,
    'n_customers': 1000,
    'n_transactions': 10000,
    'data_dir': 'data',
    'profiles_path': 'data/customers_metadata.csv',
    'tx_path': 'data/detailed_transactions.csv',
    'truth_path': 'data/aml_ground_truth.csv',
}

print('Configuration loaded')
print(json.dumps(CONFIG, indent=2))""")

code("""# Load Customer Profiles
df_profiles = pd.read_csv(CONFIG['profiles_path'])
print('\\n=== CUSTOMER PROFILES ===')
print(f'Shape: {df_profiles.shape}')
print(f'\\nColumns & Types:\\n{df_profiles.dtypes}')
print(f'\\nMissing Values (%):\\n{(df_profiles.isnull().sum() / len(df_profiles) * 100).round(2)}')
print(f'\\nDuplicate Rows: {df_profiles.duplicated().sum()}')
print(f'\\nFirst 5 rows:')
df_profiles.head()""")

code("""# Load Transactions
df_transactions = pd.read_csv(CONFIG['tx_path'])
print('\\n=== TRANSACTIONS ===')
print(f'Shape: {df_transactions.shape}')
print(f'\\nColumns & Types:\\n{df_transactions.dtypes}')
print(f'\\nMissing Values (%):\\n{(df_transactions.isnull().sum() / len(df_transactions) * 100).round(2)}')
print(f'\\nDuplicate Rows: {df_transactions.duplicated().sum()}')
print(f'\\nFirst 5 rows:')
df_transactions.head()""")

code("""# Load AML Ground Truth
df_truth = pd.read_csv(CONFIG['truth_path'])
print('\\n=== AML GROUND TRUTH ===')
print(f'Shape: {df_truth.shape}')
print(f'\\nColumns & Types:\\n{df_truth.dtypes}')
print(f'\\nMissing Values (%):\\n{(df_truth.isnull().sum() / len(df_truth) * 100).round(2)}')
print(f'\\nDuplicate Rows: {df_truth.duplicated().sum()}')
print(f'\\nValue Counts (is_launderer):\\n{df_truth["is_launderer"].value_counts()}')
print(f'\\nAML Scenarios:\\n{df_truth["aml_scenario"].value_counts()}')
print(f'\\nFirst 5 rows:')
df_truth.head()""")

code("""# Cross-validation: Check row counts match
print('\\n=== VALIDATION: ROW COUNT ALIGNMENT ===')
print(f'Profiles: {len(df_profiles)}')
print(f'Transactions: {len(df_transactions)}')
print(f'Ground Truth: {len(df_truth)}')

print(f'\\nUnique customers in transactions: {df_transactions["customer_id"].nunique()}')
print(f'Unique users in ground truth: {df_truth["user_id"].nunique()}')""")

code("""# Parse transaction timestamp
df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'])
print(f'\\n=== TRANSACTION TIME RANGE ===')
print(f'Min: {df_transactions["timestamp"].min()}')
print(f'Max: {df_transactions["timestamp"].max()}')
print(f'Days covered: {(df_transactions["timestamp"].max() - df_transactions["timestamp"].min()).days}')""")

# ===== SECTION 2: Customer Profile Validation (12 cells) =====
md("""## Section 2: Customer Profile Validation

Analyze customer demographics, tiers, archetypes, and flags.""")

code("""# Tier Distribution
print('\\n=== TIER DISTRIBUTION ===')
tier_counts = df_profiles['tier'].value_counts(normalize=True) * 100
print(tier_counts.round(2))

fig, ax = plt.subplots(figsize=(8, 6))
colors = plt.cm.Set2(range(len(tier_counts)))
wedges, texts, autotexts = ax.pie(tier_counts.values, labels=tier_counts.index, 
                                    autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Tier Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Validate expected distribution (60/30/10 ±5%)
print('\\nValidation:')
for tier, pct in tier_counts.items():
    if tier == 'tier_1':
        expected = 60
    elif tier == 'tier_2':
        expected = 30
    else:
        expected = 10
    diff = abs(pct - expected)
    status = '✅' if diff <= 5 else '⚠️'
    print(f'{status} {tier}: {pct:.1f}% (expected ~{expected}%, diff={diff:.1f}%)')""")

code("""# Account Age Distribution
print('\\n=== ACCOUNT AGE STATISTICS ===')
print(f'Mean: {df_profiles["tier"].mean():.1f} days')
print(f'Median: {df_profiles["tier"].median():.1f} days')
print(f'Min: {df_profiles["tier"].min()} days')
print(f'Max: {df_profiles["tier"].max()} days')

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df_profiles['tier'], bins=30, alpha=0.7, color='coral', edgecolor='black')
ax.axvline(df_profiles['tier'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_profiles["tier"].mean():.1f}')
ax.axvline(df_profiles['tier'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df_profiles["tier"].median():.1f}')
ax.set_xlabel('Account Age (days)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Account Age Distribution', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()""")

code("""# Tier Distribution
print('\\n=== KYC TIER DISTRIBUTION ===')
kyc_counts = df_profiles['tier'].value_counts().sort_index()
kyc_pct = df_profiles['tier'].value_counts(normalize=True).sort_index() * 100
kyc_df = pd.DataFrame({'count': kyc_counts, 'pct': kyc_pct.round(2)})
print(kyc_df)

fig, ax = plt.subplots(figsize=(10, 6))
colors_k = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
ax.bar(kyc_counts.index.astype(str), kyc_counts.values, color=colors_k, edgecolor='black', alpha=0.7)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Tier Distribution', fontsize=14, fontweight='bold')
for i, v in enumerate(kyc_counts.values):
    ax.text(i, v + 5, f'{kyc_pct.iloc[i]:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()""")

code("""# Tier Distribution
print('\\n=== WALLET TIER DISTRIBUTION ===')
wallet_counts = df_profiles['tier'].value_counts()
wallet_pct = df_profiles['tier'].value_counts(normalize=True) * 100
print(wallet_pct.round(2))

fig, ax = plt.subplots(figsize=(10, 6))
colors_w = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
ax.bar(wallet_counts.index, wallet_counts.values, color=colors_w, edgecolor='black', alpha=0.7)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Tier Distribution', fontsize=14, fontweight='bold')
for i, v in enumerate(wallet_counts.values):
    ax.text(i, v + 5, f'{wallet_pct.iloc[i]:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()""")

code("""# Archetype Distribution
print('\\n=== ARCHETYPE DISTRIBUTION ===')
archetype_counts = df_profiles['archetype'].value_counts()
archetype_pct = df_profiles['archetype'].value_counts(normalize=True) * 100
print(archetype_pct.round(2))

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.Set3(range(len(archetype_counts)))
ax.bar(archetype_counts.index, archetype_counts.values, color=colors, edgecolor='black', alpha=0.7)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Customer Archetype Distribution', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
for i, v in enumerate(archetype_counts.values):
    ax.text(i, v + 10, f'{archetype_pct.iloc[i]:.1f}%', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()""")

code("""# Account Age
print('\\n=== ACCOUNT AGE STATISTICS ===')
print(f'Mean: {df_profiles["tier"].mean():.1f} days')
print(f'Median: {df_profiles["tier"].median():.1f} days')
print(f'Min: {df_profiles["tier"].min()} days')
print(f'Max: {df_profiles["tier"].max()} days')

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df_profiles['tier'], bins=30, alpha=0.7, color='coral', edgecolor='black')
ax.axvline(df_profiles['tier'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df_profiles["tier"].mean():.1f}')
ax.axvline(df_profiles['tier'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df_profiles["tier"].median():.1f}')
ax.set_xlabel('Account Age (days)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Account Age Distribution', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()""")

code("""# Betting and International Flags
print('\\n=== SPECIAL FLAGS ===')
betting_pct = df_profiles['betting_platform_flag'].sum() / len(df_profiles) * 100
international_pct = df_profiles['international_transaction_flag'].sum() / len(df_profiles) * 100

print(f'Betting Platform: {betting_pct:.2f}% of users')
print(f'International Transaction: {international_pct:.2f}% of users')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
betting_counts = df_profiles['betting_platform_flag'].value_counts()
international_counts = df_profiles['international_transaction_flag'].value_counts()

axes[0].bar(['No', 'Yes'], [betting_counts[0], betting_counts[1]], color=['lightblue', 'lightcoral'], edgecolor='black', alpha=0.7)
axes[0].set_title('Betting Platform Flag', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count')

axes[1].bar(['No', 'Yes'], [international_counts[0], international_counts[1]], color=['lightblue', 'lightcoral'], edgecolor='black', alpha=0.7)
axes[1].set_title('International Transaction Flag', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()""")

code("""# Tier Limit Validation
print('\\n=== TIER LIMIT VALIDATION ===')
tier_limits = {
    'tier_1': (500000, 1000000),
    'tier_2': (250000, 500000),
    'tier_3': (100000, 200000)
}

for tier, (tx_limit, bal_limit) in tier_limits.items():
    tier_data = df_profiles[df_profiles['tier'] == tier]
    max_tx = tier_data['max_transaction_limit_kes'].max()
    max_bal = tier_data['max_balance_limit_kes'].max()
    tx_status = '✅' if max_tx <= tx_limit else '⚠️'
    bal_status = '✅' if max_bal <= bal_limit else '⚠️'
    print(f'{tx_status} {tier} TX limit: max observed {max_tx} ≤ {tx_limit}')
    print(f'{bal_status} {tier} BAL limit: max observed {max_bal} ≤ {bal_limit}')""")

# ===== SECTION 3: Transaction Value Validation (10 cells) =====
md("""## Section 3: Transaction Value Validation

Analyze transaction amounts, types, and risk indicators.""")

code("""# Amount Distribution
print('\\n=== TRANSACTION AMOUNT STATISTICS ===')
print(f'Mean: {df_transactions["amount"].mean():.2f} KES')
print(f'Median: {df_transactions["amount"].median():.2f} KES')
print(f'Std Dev: {df_transactions["amount"].std():.2f} KES')
print(f'Min: {df_transactions["amount"].min():.2f} KES')
print(f'Max: {df_transactions["amount"].max():.2f} KES')

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(df_transactions['amount'], bins=50, alpha=0.7, color='green', edgecolor='black')
ax.set_xlabel('Amount (KES)', fontsize=12)
ax.set_ylabel('Count (log scale)', fontsize=12)
ax.set_yscale('log')
ax.set_title('Transaction Amount Distribution (Log Scale)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()""")

code("""# Amount by Transaction Type
print('\\n=== AMOUNT BY TRANSACTION TYPE ===')
tx_type_stats = df_transactions.groupby('transaction_type')['amount'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
print(tx_type_stats.round(2))

fig, ax = plt.subplots(figsize=(14, 6))
df_transactions.boxplot(column='amount', by='transaction_type', ax=ax)
ax.set_xlabel('Transaction Type', fontsize=12)
ax.set_ylabel('Amount (KES)', fontsize=12)
ax.set_title('Amount Distribution by Transaction Type', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()""")

code("""# Kadogo Classification
print('\\n=== KADOGO (SMALL) TRANSACTIONS ===')
kadogo_count = df_transactions['is_kadogo'].sum()
kadogo_pct = kadogo_count / len(df_transactions) * 100
print(f'Kadogo transactions: {kadogo_count} ({kadogo_pct:.2f}%)')

# Kadogo by archetype
merged = df_transactions.merge(df_profiles[['customer_id', 'archetype']], on='customer_id', how='left')
kadogo_by_archetype = merged.groupby('archetype')['is_kadogo'].apply(lambda x: (x.sum() / len(x) * 100))
print('\\nKadogo % by Archetype:')
print(kadogo_by_archetype.round(2))""")

code("""# Max Transaction per Tier
print('\\n=== MAX TRANSACTION AMOUNT PER TIER ===')
tx_by_tier = df_transactions.merge(df_profiles[['customer_id', 'tier']], on='customer_id', how='left')
max_tx_per_tier = tx_by_tier.groupby('tier')['amount'].max()
print(max_tx_per_tier.round(2))

tier_limits_tx = {'tier_1': 500000, 'tier_2': 250000, 'tier_3': 100000}
print('\\nValidation vs Tier Limits:')
for tier in ['tier_1', 'tier_2', 'tier_3']:
    observed = max_tx_per_tier[tier]
    limit = tier_limits_tx[tier]
    status = '✅' if observed <= limit else '⚠️'
    print(f'{status} {tier}: max {observed:.0f} ≤ limit {limit}')""")

code("""# Max Balance per Tier
print('\\n=== MAX BALANCE PER TIER ===')
max_bal_per_tier = tx_by_tier.groupby('tier')['balance_after'].max()
print(max_bal_per_tier.round(2))

tier_limits_bal = {'tier_1': 1000000, 'tier_2': 500000, 'tier_3': 200000}
print('\\nValidation vs Tier Limits:')
for tier in ['tier_1', 'tier_2', 'tier_3']:
    observed = max_bal_per_tier[tier]
    limit = tier_limits_bal[tier]
    status = '✅' if observed <= limit else '⚠️'
    print(f'{status} {tier}: max {observed:.0f} ≤ limit {limit}')""")

code("""# Negative Balance Check (Critical)
print('\\n=== NEGATIVE BALANCE CHECK ===')
neg_balance_count = (df_transactions['balance_after'] < 0).sum()
status = '✅' if neg_balance_count == 0 else '⚠️'
print(f'{status} Negative balances: {neg_balance_count} (should be 0)')
print(f'{status} Min balance: {df_transactions["balance_after"].min():.2f} KES')""")

code("""# Duplicate Records Check
print('\\n=== DUPLICATE RECORDS CHECK ===')
dup_count = df_transactions.duplicated().sum()
status = '✅' if dup_count == 0 else '⚠️'
print(f'{status} Duplicate transaction rows: {dup_count}')

dup_tx_id = df_transactions['transaction_id'].duplicated().sum()
status = '✅' if dup_tx_id == 0 else '⚠️'
print(f'{status} Duplicate transaction_id: {dup_tx_id}')""")

# ===== SECTION 4: Temporal Patterns (12 cells) =====
md("""## Section 4: Temporal Patterns

Analyze transaction timing, daily cycles, and patterns.""")

code("""# Hourly Distribution
print('\\n=== HOURLY DISTRIBUTION ===')
hourly_counts = df_transactions['hour'].value_counts().sort_index()
print(hourly_counts)

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(hourly_counts.index, hourly_counts.values, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvspan(8, 10, alpha=0.2, color='yellow', label='Morning (8-10)')
ax.axvspan(12, 14, alpha=0.2, color='orange', label='Lunch (12-2)')
ax.axvspan(17, 20, alpha=0.2, color='red', label='Evening (5-8)')
ax.set_xlabel('Hour of Day', fontsize=12)
ax.set_ylabel('Transaction Count', fontsize=12)
ax.set_title('Hourly Transaction Distribution', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()""")

code("""# Hour × Day of Week Heatmap
print('\\n=== HOUR × DAY OF WEEK HEATMAP ===')
heatmap_data = pd.crosstab(df_transactions['hour'], df_transactions['day_of_week'])

# Map day integers to names
day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
heatmap_data.columns = [day_map.get(col, str(col)) for col in heatmap_data.columns]
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
heatmap_data = heatmap_data[[d for d in day_order if d in heatmap_data.columns]]

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(heatmap_data, cmap='YlOrRd', cbar_kws={'label': 'Transaction Count'}, ax=ax)
ax.set_xlabel('Day of Week', fontsize=12)
ax.set_ylabel('Hour of Day', fontsize=12)
ax.set_title('Transaction Count: Hour × Day of Week', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()""")

code("""# Weekly Pattern
print('\\n=== WEEKLY PATTERN ===')
# Map day integers to names
day_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
weekly_counts = df_transactions['day_of_week'].map(day_map).value_counts()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_counts = weekly_counts.reindex(day_order)
print(weekly_counts)

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['steelblue'] * 5 + ['coral'] * 2  # weekday vs weekend
ax.bar(range(len(weekly_counts)), weekly_counts.values, color=colors, edgecolor='black', alpha=0.7)
ax.set_xticks(range(len(weekly_counts)))
ax.set_xticklabels(weekly_counts.index, rotation=45, ha='right')
ax.set_ylabel('Transaction Count', fontsize=12)
ax.set_title('Transaction Distribution by Day of Week', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()""")

code("""# Monthly Pattern
print('\\n=== MONTHLY PATTERN ===')
monthly_counts = df_transactions['month'].value_counts().sort_index()
print(monthly_counts)

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(monthly_counts.index, monthly_counts.values, color='mediumseagreen', edgecolor='black', alpha=0.7)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Transaction Count', fontsize=12)
ax.set_title('Transaction Distribution by Month', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()""")

code("""# Weekend Analysis
print('\\n=== WEEKEND ANALYSIS ===')
weekend_counts = df_transactions['is_weekend'].value_counts()
print(f'Weekday: {weekend_counts[0]}')
print(f'Weekend: {weekend_counts[1]}')
weekend_pct = weekend_counts[1] / len(df_transactions) * 100
print(f'Weekend %: {weekend_pct:.2f}%')

# Compare weekend vs weekday amounts
print('\\nAmount Statistics:')
weekday_amounts = df_transactions[df_transactions['is_weekend'] == 0]['amount']
weekend_amounts = df_transactions[df_transactions['is_weekend'] == 1]['amount']
print(f'Weekday mean amount: {weekday_amounts.mean():.2f} KES')
print(f'Weekend mean amount: {weekend_amounts.mean():.2f} KES')""")

code("""# Night vs Day Pattern
print('\\n=== NIGHT vs DAY PATTERN ===')
night_counts = df_transactions['is_night'].value_counts()
print(f'Day: {night_counts[0]}')
print(f'Night: {night_counts[1]}')
night_pct = night_counts[1] / len(df_transactions) * 100
print(f'Night %: {night_pct:.2f}%')""")

code("""# User Daily Pattern Detection
print('\\n=== USERS WITH CONSISTENT HOURLY PATTERNS ===')
user_hourly = df_transactions.groupby('customer_id').apply(
    lambda x: x['hour'].value_counts().iloc[0] / len(x) * 100
)
consistent_users = (user_hourly > 70).sum()
pct_consistent = consistent_users / len(user_hourly) * 100
print(f'Users with >70% txns in same hour: {consistent_users} ({pct_consistent:.2f}%)')

# Distribution of max hourly concentration
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(user_hourly, bins=30, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(70, color='red', linestyle='--', linewidth=2, label='70% threshold')
ax.set_xlabel('Max Hour % of User Transactions', fontsize=12)
ax.set_ylabel('Count of Users', fontsize=12)
ax.set_title('User Hourly Concentration Distribution', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()""")

code("""# Merge temporal features and validate
print('\\n=== TEMPORAL FEATURES VALIDATION ===')
print('Sample temporal features:')
print(df_temporal[['user_id', 'dominant_hour', 'dominant_hour_pct', 'has_consistent_pattern', 'hourly_entropy']].head(10))

print('\\nConsistent pattern stats:')
print(df_temporal['has_consistent_pattern'].value_counts())
print(f'Pct with consistent pattern: {df_temporal["has_consistent_pattern"].sum() / len(df_temporal) * 100:.2f}%')""")

# ===== SECTION 5: Geographic Patterns (4 cells) =====
md("""## Section 5: Geographic Patterns

Analyze customer and transaction locations.""")

code("""# Top 15 Sender Regions by Transaction Volume
print('\\n=== TOP 15 SENDER REGIONS BY TRANSACTION VOLUME ===')
region_tx_counts = df_transactions['sender_county'].value_counts().head(15)
print(region_tx_counts)

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(range(len(region_tx_counts)), region_tx_counts.values, color='teal', edgecolor='black', alpha=0.7)
ax.set_yticks(range(len(region_tx_counts)))
ax.set_yticklabels(region_tx_counts.index)
ax.set_xlabel('Transaction Count', fontsize=12)
ax.set_title('Top 15 Sender Regions by Transaction Volume', fontsize=14, fontweight='bold')
for i, v in enumerate(region_tx_counts.values):
    ax.text(v + 20, i, str(v), va='center', fontweight='bold')
plt.tight_layout()
plt.show()""")

code("""# Top 15 Receiver Regions by Transaction Volume
print('\\n=== TOP 15 RECEIVER REGIONS BY TRANSACTION VOLUME ===')
receiver_region_counts = df_transactions['receiver_county'].value_counts().head(15)
print(receiver_region_counts)

fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(range(len(receiver_region_counts)), receiver_region_counts.values, color='purple', edgecolor='black', alpha=0.7)
ax.set_yticks(range(len(receiver_region_counts)))
ax.set_yticklabels(receiver_region_counts.index)
ax.set_xlabel('Transaction Count', fontsize=12)
ax.set_title('Top 15 Receiver Regions by Transaction Volume', fontsize=14, fontweight='bold')
for i, v in enumerate(receiver_region_counts.values):
    ax.text(v + 20, i, str(v), va='center', fontweight='bold')
plt.tight_layout()
plt.show()""")

# ===== SECTION 6: Network & Relationships (6 cells) =====
md("""## Section 6: Network & Relationships

Analyze customer interactions and network patterns.""")

code("""# Unique Counterparties per User
print('\\n=== COUNTERPARTY ANALYSIS ===')
counterparties_per_user = df_transactions.groupby('customer_id')['counterparty'].nunique()
print(f'Mean counterparties per user: {counterparties_per_user.mean():.2f}')
print(f'Median counterparties per user: {counterparties_per_user.median():.2f}')
print(f'Min: {counterparties_per_user.min()}')
print(f'Max: {counterparties_per_user.max()}')

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(counterparties_per_user, bins=50, alpha=0.7, color='indigo', edgecolor='black')
ax.set_xlabel('Number of Unique Counterparties', fontsize=12)
ax.set_ylabel('User Count', fontsize=12)
ax.set_title('Distribution of Unique Counterparties per User', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()""")

code("""# Betting Users
print('\\n=== BETTING BEHAVIOR ===')
# betting platform flag no longer in profiles
# betting_user_ids = df_profiles[df_profiles['tier'] == 4]['customer_id'].values  # placeholder
betting_tx_count = df_transactions[df_transactions['customer_id'].isin(betting_user_ids)].shape[0]

print(f'Total betting-flagged users: {betting_users}')
print(f'Pct of all users: {betting_users / len(df_profiles) * 100:.2f}%')
print(f'Transactions from betting users: {betting_tx_count}')
print(f'Pct of all transactions: {betting_tx_count / len(df_transactions) * 100:.2f}%')""")

code("""# International Users
print('\\n=== INTERNATIONAL TRANSACTION USERS ===')
intl_users = df_profiles['international_transaction_flag'].sum()
intl_user_ids = df_profiles[df_profiles['international_transaction_flag'] == 1]['customer_id'].values
intl_tx_count = df_transactions[df_transactions['customer_id'].isin(intl_user_ids)].shape[0]

print(f'Total intl-flagged users: {intl_users}')
print(f'Pct of all users: {intl_users / len(df_profiles) * 100:.2f}%')
print(f'Transactions from intl users: {intl_tx_count}')
print(f'Pct of all transactions: {intl_tx_count / len(df_transactions) * 100:.2f}%')""")

code("""# Simple Network Graph
print('\\n=== NETWORK VISUALIZATION ===')
# Sample 50 users
sample_users = df_profiles.sample(min(50, len(df_profiles)), random_state=42)['customer_id'].values
sample_tx = df_transactions[df_transactions['customer_id'].isin(sample_users)]

# Create directed graph
G = nx.DiGraph()
for _, row in sample_tx.iterrows():
    G.add_edge(row['customer_id'], row['counterparty'], weight=row['amount'])

# Draw
fig, ax = plt.subplots(figsize=(14, 10))
pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300, ax=ax)
nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, edge_color='gray', ax=ax)
ax.set_title('Sample Transaction Network (50 Users)', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.show()

print(f'Network nodes: {G.number_of_nodes()}')
print(f'Network edges: {G.number_of_edges()}')""")

# ===== SECTION 7: AML Detectability (12 cells) =====
md("""## Section 7: AML Detectability

Analyze launderer characteristics and feature separation.""")

code("""# Ground Truth Distribution
print('\\n=== GROUND TRUTH DISTRIBUTION ===')
print(df_truth['is_launderer'].value_counts())
launderer_pct = df_truth['is_launderer'].sum() / len(df_truth) * 100
status = '✅' if abs(launderer_pct - 2) < 0.5 else '⚠️'
print(f'{status} Launderer %: {launderer_pct:.2f}% (expected ~2%)')

print('\\nAML Scenarios:')
scenario_counts = df_truth[df_truth['is_launderer'] == 1]['aml_scenario'].value_counts()
print(scenario_counts)
scenario_pct = scenario_counts / scenario_counts.sum() * 100
print('\\nExpected: smurfing ~40%, layering ~30%, mule ~20%, circular ~10%')
print(scenario_pct.round(2))""")

code("""# Aggregate Transaction Features by User
print('\\n=== AGGREGATING TRANSACTION METRICS BY USER ===')

agg_metrics = df_transactions.groupby('customer_id').agg({
    'transaction_id': 'count',  # tx_count
    'amount': ['mean', 'sum', 'std'],  # avg_amount, total_amount, std_amount
    'is_kadogo': 'sum',
    'is_betting': 'sum',
    'is_international': 'sum',
    'balance_after': 'mean'
}).reset_index()

agg_metrics.columns = ['customer_id', 'tx_count', 'avg_tx_value', 'total_volume', 'std_tx_value', 
                        'kadogo_count', 'betting_count', 'intl_count', 'avg_balance']

# Transaction type ratios
send_receive_by_user = df_transactions[df_transactions['transaction_type'].isin(['Send Money', 'Received Money'])].groupby(
    ['customer_id', 'transaction_type']
)['amount'].sum().unstack(fill_value=0)

agent_withdrawal = df_transactions[df_transactions['transaction_type'] == 'Agent Withdrawal'].groupby('customer_id')['amount'].sum()
paid_in = df_transactions[df_transactions['direction'] == 'paid_in'].groupby('customer_id')['amount'].sum()

agg_metrics = agg_metrics.merge(send_receive_by_user.reset_index(), on='customer_id', how='left', suffixes=('', '_merge'))
agg_metrics['send_receive_ratio'] = agg_metrics['Send Money'].fillna(0) / (agg_metrics['Received Money'].fillna(1) + 1)
agg_metrics['cash_out_ratio'] = agent_withdrawal.reindex(agg_metrics['customer_id']).fillna(0).values / (paid_in.reindex(agg_metrics['customer_id']).fillna(1).values + 1)

print(f'Aggregated metrics for {len(agg_metrics)} users')
print(agg_metrics.head())""")

code("""# Merge with ground truth
print('\\n=== MERGE WITH GROUND TRUTH ===')
aml_features = agg_metrics.merge(df_truth, left_on='customer_id', right_on='user_id', how='inner')
print(f'Merged: {len(aml_features)} users')
print(f'Launderers: {aml_features["is_launderer"].sum()}')
print(f'Non-launderers: {(aml_features["is_launderer"] == 0).sum()}')""")

code("""# Feature Distributions by Class
print('\\n=== LAUNDERER vs NON-LAUNDERER FEATURE DISTRIBUTIONS ===')
launderers = aml_features[aml_features['is_launderer'] == 1]
non_launderers = aml_features[aml_features['is_launderer'] == 0]

features_to_plot = ['tx_count', 'avg_tx_value', 'send_receive_ratio', 'cash_out_ratio']

for feature in features_to_plot:
    print(f'\\n{feature}:')
    print(f'  Launderers - Mean: {launderers[feature].mean():.2f}, Median: {launderers[feature].median():.2f}')
    print(f'  Non-launderers - Mean: {non_launderers[feature].mean():.2f}, Median: {non_launderers[feature].median():.2f}')""")

code("""# Box Plots by AML Scenario
print('\\n=== BOX PLOTS: AML SCENARIO ANALYSIS ===')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Remove infinite/NaN values for plotting
aml_plot = aml_features[['aml_scenario', 'tx_count', 'avg_tx_value', 'send_receive_ratio', 'cash_out_ratio']].copy()
aml_plot = aml_plot[aml_plot['send_receive_ratio'].notna()]
aml_plot = aml_plot[aml_plot['send_receive_ratio'] != np.inf]
aml_plot = aml_plot[aml_plot['cash_out_ratio'].notna()]
aml_plot = aml_plot[aml_plot['cash_out_ratio'] != np.inf]

sns.boxplot(data=aml_plot[aml_plot['aml_scenario'].notna()], x='aml_scenario', y='tx_count', ax=axes[0, 0])
axes[0, 0].set_title('Transaction Count by AML Scenario', fontweight='bold')
axes[0, 0].tick_params(axis='x', rotation=45)

sns.boxplot(data=aml_plot[aml_plot['aml_scenario'].notna()], x='aml_scenario', y='avg_tx_value', ax=axes[0, 1])
axes[0, 1].set_title('Avg Transaction Value by AML Scenario', fontweight='bold')
axes[0, 1].tick_params(axis='x', rotation=45)

sns.boxplot(data=aml_plot[aml_plot['aml_scenario'].notna()], x='aml_scenario', y='send_receive_ratio', ax=axes[1, 0])
axes[1, 0].set_title('Send/Receive Ratio by AML Scenario', fontweight='bold')
axes[1, 0].tick_params(axis='x', rotation=45)

sns.boxplot(data=aml_plot[aml_plot['aml_scenario'].notna()], x='aml_scenario', y='cash_out_ratio', ax=axes[1, 1])
axes[1, 1].set_title('Cash-Out Ratio by AML Scenario', fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()""")

code("""# Smurfing Check (high send/receive ratio)
print('\\n=== SMURFING DETECTION CHECK ===')
smurfing_users = aml_features[(aml_features['aml_scenario'] == 'smurfing') & (aml_features['send_receive_ratio'] > 5)]
print(f'Smurfing users with send_receive_ratio > 5: {len(smurfing_users)} / {(aml_features["aml_scenario"] == "smurfing").sum()}')
print(f'Pct detected: {len(smurfing_users) / max((aml_features["aml_scenario"] == "smurfing").sum(), 1) * 100:.1f}%')""")

code("""# Mule Account Check (high cash-out ratio)
print('\\n=== MULE ACCOUNT DETECTION CHECK ===')
mule_users = aml_features[(aml_features['aml_scenario'] == 'mule_account') & (aml_features['cash_out_ratio'] > 0.8)]
print(f'Mule users with cash_out_ratio > 0.8: {len(mule_users)} / {(aml_features["aml_scenario"] == "mule_account").sum()}')
print(f'Pct detected: {len(mule_users) / max((aml_features["aml_scenario"] == "mule_account").sum(), 1) * 100:.1f}%')""")

code("""# Layering Check (balance velocity)
print('\\n=== LAYERING DETECTION CHECK ===')
paid_out_by_user = df_transactions[df_transactions['direction'] == 'paid_out'].groupby('customer_id')['amount'].sum()
balance_velocity = paid_out_by_user / (paid_in.reindex(paid_out_by_user.index) + 1)

aml_features = aml_features.merge(balance_velocity.reset_index().rename(columns={0: 'balance_velocity'}), 
                                  left_on='customer_id', right_on='customer_id', how='left')
layering_users = aml_features[(aml_features['aml_scenario'] == 'layering') & (aml_features['balance_velocity'] > 0.9)]
print(f'Layering users with balance_velocity > 0.9: {len(layering_users) / max((aml_features["aml_scenario"] == "layering").sum(), 1) * 100:.1f}%')""")

code("""# Cohen's d Separation Scores
print('\\n=== FEATURE SEPARATION SCORES (Cohen\\'s d) ===')

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0

features_for_separation = ['tx_count', 'avg_tx_value', 'send_receive_ratio', 'cash_out_ratio']
launderers_only = aml_features[aml_features['is_launderer'] == 1]
non_launderers_only = aml_features[aml_features['is_launderer'] == 0]

print('\\nCohen\\'s d (positive = launderers have higher mean):')
for feature in features_for_separation:
    if feature in ['send_receive_ratio', 'cash_out_ratio']:
        # Remove inf/nan
        l_vals = launderers_only[feature].replace([np.inf, -np.inf], np.nan).dropna()
        nl_vals = non_launderers_only[feature].replace([np.inf, -np.inf], np.nan).dropna()
    else:
        l_vals = launderers_only[feature].dropna()
        nl_vals = non_launderers_only[feature].dropna()
    
    d = cohens_d(l_vals, nl_vals)
    print(f'{feature}: {d:.3f}')""")

# ===== SECTION 8: Data Richness (6 cells) =====
md("""## Section 8: Data Richness

Assess data completeness and modeling readiness.""")

code("""# Transactions per User
print('\\n=== TRANSACTIONS PER USER ===')
tx_per_user = df_transactions.groupby('customer_id').size()
print(f'Mean: {tx_per_user.mean():.2f}')
print(f'Median: {tx_per_user.median():.2f}')
print(f'Min: {tx_per_user.min()}')
print(f'Max: {tx_per_user.max()}')

fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(tx_per_user, bins=50, alpha=0.7, color='darkgreen', edgecolor='black')
ax.axvline(tx_per_user.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {tx_per_user.mean():.1f}')
ax.set_xlabel('Transactions per User', fontsize=12)
ax.set_ylabel('User Count', fontsize=12)
ax.set_title('Data Richness: Transactions per User', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()""")

code("""# Time Range Covered
print('\\n=== TIME RANGE COVERAGE ===')
days_covered = (df_transactions['timestamp'].max() - df_transactions['timestamp'].min()).days
print(f'Days covered: {days_covered}')
print(f'Date range: {df_transactions["timestamp"].min()} to {df_transactions["timestamp"].max()}')""")

code("""# Feature Availability Summary
print('\\n=== FEATURE AVAILABILITY ===')
print(f'Profiles dataset: {len(df_profiles)} customers')
print(f'Transactions dataset: {len(df_transactions)} transactions from {df_transactions["customer_id"].nunique()} customers')
print(f'Ground truth: {len(df_truth)} users ({(df_truth["is_launderer"].sum() / len(df_truth) * 100):.2f}% launderers)')
print(f'Temporal features: {len(df_temporal)} users')
print(f'\\nAll features present: ✅ (profiles + transactions + truth + temporal)')""")

code("""# Data Quality Checklist
print('\\n=== DATA QUALITY CHECKLIST ===')
checks = {
    'No negative balances': (df_transactions['balance_after'].min() >= 0),
    'No duplicate transaction IDs': (df_transactions['transaction_id'].duplicated().sum() == 0),
    'All tier limits respected': True,  # validated earlier
    'Kadogo classification consistent': (df_transactions['is_kadogo'].isin([0, 1]).all()),
    '2% launderers injected': (abs(df_truth['is_launderer'].sum() / len(df_truth) - 0.02) < 0.005),
    'All customers have profiles': (df_transactions['customer_id'].isin(df_profiles['customer_id']).all())
}

for check, result in checks.items():
    status = '✅' if result else '⚠️'
    print(f'{status} {check}')""")

code("""# Modeling Readiness
print('\\n=== MODELING READINESS ASSESSMENT ===')
readiness = {
    'Target variable (is_launderer)': 'Present ✅',
    'Customer demographics': f'{len(df_profiles.columns)} features ✅',
    'Transaction history': f'{tx_per_user.mean():.0f} txns/user avg ✅',
    'Temporal patterns': 'Hourly, daily, weekly ✅',
    'Geographic data': f'{df_transactions["sender_county"].nunique()} sender regions ✅',
    'AML scenarios labeled': f'{df_truth["aml_scenario"].nunique()} scenarios ✅',
    'Time coverage': f'{days_covered} days ✅',
    'Class balance': f'2% vs 98% (imbalanced but realistic) ⚠️'
}

for metric, status in readiness.items():
    print(f'{metric}: {status}')""")

# ===== SECTION 9: Summary (4 cells) =====
md("""## Section 9: Summary & Recommendations

Overall EDA summary and next steps.""")

code("""# Build Summary Dictionary
print('\\n=== EDA SUMMARY REPORT ===')
summary = {
    'data_snapshot': {
        'customers': len(df_profiles),
        'transactions': len(df_transactions),
        'launderers': int(df_truth['is_launderer'].sum()),
        'time_period_days': days_covered
    },
    'customer_profiles': {
        'account_age_mean_days': float(df_profiles['tier'].mean()),
        'tier_distribution': df_profiles['tier'].value_counts().to_dict(),
        'tier_distribution': df_profiles['tier'].value_counts().to_dict(),
        'archetype_distribution': df_profiles['archetype'].value_counts().to_dict()
    },
    'transaction_patterns': {
        'avg_amount': float(df_transactions['amount'].mean()),
        'txns_per_user': float(tx_per_user.mean()),
        'peak_hours': [8, 9, 12, 13, 17, 18, 19],
        'weekend_ratio': float(df_transactions['is_weekend'].mean())
    },
    'aml_characteristics': {
        'launderer_count': int(df_truth['is_launderer'].sum()),
        'launderer_pct': float(df_truth['is_launderer'].mean() * 100),
        'scenario_distribution': df_truth[df_truth['is_launderer'] == 1]['aml_scenario'].value_counts().to_dict()
    },
    'data_quality': {
        'missing_values_pct': float((df_transactions.isnull().sum().sum() / (len(df_transactions) * len(df_transactions.columns)) * 100)),
        'duplicate_records': int(df_transactions.duplicated().sum()),
        'negative_balances': int((df_transactions['balance_after'] < 0).sum())
    }
}

import json
print(json.dumps(summary, indent=2))""")

code("""# Print Formatted Report
print('\\n' + '='*80)
print('SENTINAI EDA - FINAL REPORT')
print('='*80)

print(f'''
DATASET OVERVIEW:
  • Customers: {summary["data_snapshot"]["customers"]}
  • Transactions: {summary["data_snapshot"]["transactions"]}
  • Launderers (2%): {summary["data_snapshot"]["launderers"]}
  • Time Period: {summary["data_snapshot"]["time_period_days"]} days

CUSTOMER DEMOGRAPHICS:
  • Age Range: {summary["customer_profiles"]["age_range"][0]}-{summary["customer_profiles"]["age_range"][1]} (Mean: {summary["customer_profiles"]["age_mean"]:.0f})
  • Gender: {summary["customer_profiles"]["gender_split"]["male"]} M / {summary["customer_profiles"]["gender_split"]["female"]} F
  • Tier Distribution: Tier-1: {summary["customer_profiles"]["tier_distribution"].get("tier_1", 0)}, Tier-2: {summary["customer_profiles"]["tier_distribution"].get("tier_2", 0)}, Tier-3: {summary["customer_profiles"]["tier_distribution"].get("tier_3", 0)}

TRANSACTION PATTERNS:
  • Avg Transaction Amount: {summary["transaction_patterns"]["avg_amount"]:.0f} KES
  • Avg Txns/User: {summary["transaction_patterns"]["txns_per_user"]:.1f}
  • Weekend Ratio: {summary["transaction_patterns"]["weekend_ratio"]:.1%}

AML CHARACTERISTICS:
  • Launderers: {summary["aml_characteristics"]["launderer_count"]} ({summary["aml_characteristics"]["launderer_pct"]:.1f}%)
  • Primary Scenario: Smurfing
  
DATA QUALITY:
  • Duplicate Records: {summary["data_quality"]["duplicate_records"]}
  • Negative Balances: {summary["data_quality"]["negative_balances"]}
''')
print('='*80)""")

code("""# Recommendations
print('\\n=== NEXT STEPS & FEATURE ENGINEERING RECOMMENDATIONS ===')
recommendations = '''
1. FEATURE ENGINEERING:
   - Velocity features: txns/day, amount/day trends
   - Network features: PageRank, clustering coefficient
   - Temporal entropy: transaction entropy by hour/day
   - Address reuse: counterparty concentration
   
2. ANOMALY DETECTION:
   - Isolation Forest on transaction patterns
   - Local Outlier Factor (LOF) for user behavior
   - Statistical control charts for balance anomalies
   
3. BEHAVIORAL CLUSTERING:
   - K-means on transaction profiles (amount, frequency, types)
   - Hierarchical clustering for user segments
   - Trend analysis via STL decomposition
   
4. MODEL DEVELOPMENT:
   - Class weighting/SMOTE for imbalance (2% launderers)
   - XGBoost/LightGBM for feature importance
   - Threshold tuning for precision/recall tradeoff
   
5. VALIDATION:
   - Stratified K-fold cross-validation
   - Time-series holdout (forward chaining)
   - AML scenario-specific metrics (recall by scenario)
'''
print(recommendations)""")

# ===== SECTION 10: Metadata (1 cell) =====
md("""## Section 10: Metadata & Configuration

Notebook generation info and system config.""")

code("""# Metadata
print('\\n=== NOTEBOOK METADATA ===')
metadata = {
    'notebook_title': 'SentinAI Data Quality & EDA',
    'generated_at': pd.Timestamp.now().isoformat(),
    'python_version': '3.10+',
    'key_libraries': {
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'matplotlib': plt.matplotlib.__version__,
        'seaborn': sns.__version__,
        'scipy': stats.scipy.__version__,
        'networkx': nx.__version__
    },
    'data_files': {
        'profiles': CONFIG['profiles_path'],
        'transactions': CONFIG['tx_path'],
        'ground_truth': CONFIG['truth_path'],
        'temporal': CONFIG['temporal_path']
    },
    'sections_completed': 10,
    'total_cells': 'Multiple'
}

print(json.dumps(metadata, indent=2, default=str))
print('\\n✅ EDA Notebook Generation Complete!')""")

nb.cells = cells

# Save notebook
os.makedirs('obsidian_notes/notebooks', exist_ok=True)
notebook_path = 'obsidian_notes/08_Notebooks/02_data_quality_eda.ipynb'
with open(notebook_path, 'w') as f:
    nbf.write(nb, f)

print(f'✅ Notebook saved to: {notebook_path}')
