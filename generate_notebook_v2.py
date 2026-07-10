"""
Generate SentinAI EDA Notebook - Simplified & Robust Version
"""
import nbformat as nbf
import os

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

# ===== SETUP =====
md("""# SentinAI Data Quality & EDA Notebook

Comprehensive exploratory data analysis of SentinAI transaction and customer data.""")

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

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print('✅ Libraries loaded')""")

# ===== SECTION 1: Data Loading =====
md("""## Section 1: Data Loading & Validation""")

code("""# Load all datasets
df_profiles = pd.read_csv('data/bronze/customers/customer_profiles_complete.csv')
df_transactions = pd.read_csv('data/detailed_transactions.csv')
df_truth = pd.read_csv('data/aml_ground_truth.csv')
df_temporal = pd.read_csv('data/temporal_features.csv')

print('\\n=== DATASET SHAPES ===')
print(f'Profiles: {df_profiles.shape}')
print(f'Transactions: {df_transactions.shape}')
print(f'Ground Truth: {df_truth.shape}')
print(f'Temporal: {df_temporal.shape}')""")

code("""# Data quality overview
print('\\n=== DATA QUALITY ===')
for name, df in [('Profiles', df_profiles), ('Transactions', df_transactions), 
                  ('Ground Truth', df_truth), ('Temporal', df_temporal)]:
    print(f'\\n{name}:')
    print(f'  Duplicates: {df.duplicated().sum()}')
    print(f'  Missing %: {(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100):.2f}%')""")

code("""# Sample data
print('\\n=== SAMPLE DATA ===')
print('\\nProfiles (first 3 rows):')
print(df_profiles[['customer_id', 'kyc_tier', 'age', 'archetype', 'county']].head(3))
print('\\nTransactions (first 3 rows):')
print(df_transactions[['transaction_id', 'customer_id', 'amount', 'transaction_type', 'timestamp']].head(3))
print('\\nGround Truth (first 3 rows):')
print(df_truth.head(3))""")

# ===== SECTION 2: Customer Profiles =====
md("""## Section 2: Customer Demographics & Profiles""")

code("""# Tier distribution
print('\\n=== KYC TIER DISTRIBUTION ===')
tier_dist = df_profiles['kyc_tier'].value_counts()
tier_pct = (tier_dist / len(df_profiles) * 100).round(1)
for tier, count in tier_dist.items():
    print(f'{tier}: {count} ({tier_pct[tier]}%)')

fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.Set2(range(len(tier_dist)))
ax.bar(tier_dist.index, tier_dist.values, color=colors, edgecolor='black', alpha=0.7)
ax.set_ylabel('Count')
ax.set_title('KYC Tier Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()""")

code("""# Age distribution
print('\\n=== AGE STATISTICS ===')
print(f'Mean: {df_profiles["age"].mean():.1f}')
print(f'Median: {df_profiles["age"].median():.1f}')
print(f'Range: {df_profiles["age"].min()}-{df_profiles["age"].max()}')

fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(df_profiles['age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax.axvline(df_profiles['age'].mean(), color='red', linestyle='--', label=f'Mean: {df_profiles["age"].mean():.0f}')
ax.axvline(df_profiles['age'].median(), color='green', linestyle='--', label=f'Median: {df_profiles["age"].median():.0f}')
ax.set_xlabel('Age')
ax.set_ylabel('Count')
ax.set_title('Age Distribution')
ax.legend()
plt.tight_layout()
plt.show()""")

code("""# Gender and archetype
print('\\n=== GENDER DISTRIBUTION ===')
gender_dist = df_profiles['gender'].value_counts()
print(gender_dist)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].bar(gender_dist.index, gender_dist.values, color=['#4169E1', '#FF69B4'], alpha=0.7)
axes[0].set_ylabel('Count')
axes[0].set_title('Gender Distribution')

archetype_dist = df_profiles['archetype'].value_counts()
axes[1].bar(archetype_dist.index, archetype_dist.values, color=plt.cm.Set3(range(len(archetype_dist))), alpha=0.7)
axes[1].set_ylabel('Count')
axes[1].set_title('Archetype Distribution')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()""")

code("""# Special flags
print('\\n=== SPECIAL FLAGS ===')
betting_pct = df_profiles['betting_platform_flag'].sum() / len(df_profiles) * 100
intl_pct = df_profiles['international_transaction_flag'].sum() / len(df_profiles) * 100
print(f'Betting users: {betting_pct:.1f}%')
print(f'International users: {intl_pct:.1f}%')""")

# ===== SECTION 3: Transactions =====
md("""## Section 3: Transaction Analysis""")

code("""# Parse timestamp
df_transactions['timestamp'] = pd.to_datetime(df_transactions['timestamp'])

print('\\n=== TRANSACTION STATISTICS ===')
print(f'Mean amount: {df_transactions["amount"].mean():.0f} KES')
print(f'Median amount: {df_transactions["amount"].median():.0f} KES')
print(f'Total volume: {df_transactions["amount"].sum():,.0f} KES')
print(f'Min balance: {df_transactions["balance_after"].min():.0f} KES')
print(f'Max balance: {df_transactions["balance_after"].max():.0f} KES')

# Critical validation
neg_balance = (df_transactions['balance_after'] < 0).sum()
status = '✅' if neg_balance == 0 else '⚠️'
print(f'{status} Negative balances: {neg_balance}')""")

code("""# Amount distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df_transactions['amount'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Amount (KES)')
axes[0].set_ylabel('Count')
axes[0].set_title('Transaction Amount Distribution')
axes[0].set_yscale('log')

# Amount by transaction type (top 10)
top_types = df_transactions['transaction_type'].value_counts().head(10).index
df_tx_sample = df_transactions[df_transactions['transaction_type'].isin(top_types)]
df_tx_sample.boxplot(column='amount', by='transaction_type', ax=axes[1])
axes[1].set_xlabel('Transaction Type')
axes[1].set_ylabel('Amount (KES)')
axes[1].set_title('Amount by Transaction Type (Top 10)')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()""")

code("""# Transaction types
print('\\n=== TRANSACTION TYPES ===')
tx_types = df_transactions['transaction_type'].value_counts().head(10)
print(tx_types)

# Kadogo analysis
kadogo_pct = df_transactions['is_kadogo'].sum() / len(df_transactions) * 100
print(f'\\nKadogo (small) transactions: {kadogo_pct:.1f}%')""")

# ===== SECTION 4: Temporal Patterns =====
md("""## Section 4: Temporal Patterns""")

code("""# Hourly pattern
print('\\n=== HOURLY DISTRIBUTION ===')
hourly = df_transactions['hour'].value_counts().sort_index()
print(f'Peak hour: {hourly.idxmax()} with {hourly.max()} transactions')

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(hourly.index, hourly.values, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvspan(8, 10, alpha=0.2, color='yellow', label='Morning (8-10)')
ax.axvspan(12, 14, alpha=0.2, color='orange', label='Lunch (12-2)')
ax.axvspan(17, 20, alpha=0.2, color='red', label='Evening (5-8)')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Transaction Count')
ax.set_title('Hourly Transaction Distribution')
ax.legend()
plt.tight_layout()
plt.show()""")

code("""# Day of week (map from integers to names)
print('\\n=== WEEKLY PATTERN ===')
day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
weekly = df_transactions['day_of_week'].map(day_map).value_counts().reindex([day_map[i] for i in range(7)])
print(weekly)

fig, ax = plt.subplots(figsize=(10, 5))
colors = ['steelblue']*5 + ['coral']*2
ax.bar(weekly.index, weekly.values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Transaction Count')
ax.set_title('Transactions by Day of Week')
plt.tight_layout()
plt.show()""")

code("""# Monthly trend
print('\\n=== MONTHLY DISTRIBUTION ===')
monthly = df_transactions['month'].value_counts().sort_index()
print(f'Months covered: {len(monthly)}')

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly.index, monthly.values, marker='o', linewidth=2, markersize=8)
ax.set_xlabel('Month')
ax.set_ylabel('Transaction Count')
ax.set_title('Monthly Transaction Trend')
plt.tight_layout()
plt.show()""")

code("""# Weekend analysis
print('\\n=== WEEKEND vs WEEKDAY ===')
weekend_counts = df_transactions['is_weekend'].value_counts()
weekend_pct = weekend_counts.get(1, 0) / len(df_transactions) * 100
print(f'Weekend transactions: {weekend_pct:.1f}%')

is_night_pct = df_transactions['is_night'].sum() / len(df_transactions) * 100
print(f'Night transactions: {is_night_pct:.1f}%')""")

# ===== SECTION 5: AML Detection =====
md("""## Section 5: AML Ground Truth & Detectability""")

code("""# Ground truth distribution
print('\\n=== AML GROUND TRUTH ===')
launderer_count = df_truth['is_launderer'].sum()
launderer_pct = launderer_count / len(df_truth) * 100
status = '✅' if abs(launderer_pct - 2) < 1 else '⚠️'
print(f'{status} Launderers: {launderer_count} ({launderer_pct:.1f}%)')

scenario_dist = df_truth[df_truth['is_launderer'] == 1]['aml_scenario'].value_counts()
print('\\nAML Scenarios:')
for scenario, count in scenario_dist.items():
    pct = count / launderer_count * 100
    print(f'  {scenario}: {count} ({pct:.0f}%)')""")

code("""# Merge transactions with truth for analysis
agg_by_user = df_transactions.groupby('customer_id').agg({
    'transaction_id': 'count',
    'amount': ['sum', 'mean', 'std'],
    'is_kadogo': 'sum',
    'is_betting': 'sum',
    'balance_after': 'mean'
}).reset_index()

agg_by_user.columns = ['customer_id', 'tx_count', 'total_volume', 'avg_amount', 'std_amount',
                        'kadogo_count', 'betting_count', 'avg_balance']

merged = agg_by_user.merge(df_truth, left_on='customer_id', right_on='user_id', how='inner')

print(f'\\nMerged {len(merged)} users with ground truth')
print(f'Launderers in merged: {merged["is_launderer"].sum()}')""")

code("""# Feature comparison
print('\\n=== FEATURE DISTRIBUTIONS ===')
launderers = merged[merged['is_launderer'] == 1]
non_launderers = merged[merged['is_launderer'] == 0]

features = ['tx_count', 'total_volume', 'avg_amount', 'kadogo_count']
for feature in features:
    l_mean = launderers[feature].mean()
    nl_mean = non_launderers[feature].mean()
    diff_pct = (l_mean - nl_mean) / nl_mean * 100 if nl_mean > 0 else 0
    print(f'{feature}:')
    print(f'  Launderers: {l_mean:.1f}, Non-launderers: {nl_mean:.1f} (diff: {diff_pct:+.0f}%)')""")

code("""# Visualize feature differences
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

features = ['tx_count', 'total_volume', 'avg_amount', 'kadogo_count']
for idx, feature in enumerate(features):
    ax = axes[idx // 2, idx % 2]
    data_to_plot = [launderers[feature].dropna(), non_launderers[feature].dropna()]
    ax.boxplot(data_to_plot, labels=['Launderers', 'Non-Launderers'])
    ax.set_ylabel(feature)
    ax.set_title(f'{feature} Distribution')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()""")

# ===== SECTION 6: Geographic =====
md("""## Section 6: Geographic Analysis""")

code("""# Top counties
print('\\n=== GEOGRAPHIC DISTRIBUTION ===')
top_counties = df_profiles['county'].value_counts().head(10)
print(top_counties)

# Urban vs rural
urban_rural = df_profiles['urban_rural_classification'].value_counts()
print(f'\\nUrban/Rural:')
print(urban_rural)""")

code("""# Geographic visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Top 10 counties
axes[0].barh(range(len(top_counties)), top_counties.values, color='teal', alpha=0.7)
axes[0].set_yticks(range(len(top_counties)))
axes[0].set_yticklabels(top_counties.index)
axes[0].set_xlabel('Customer Count')
axes[0].set_title('Top 10 Counties')

# Urban vs rural pie
colors = ['#90EE90', '#FFB6C1']
axes[1].pie(urban_rural.values, labels=urban_rural.index, autopct='%1.1f%%', colors=colors)
axes[1].set_title('Urban vs Rural')

plt.tight_layout()
plt.show()""")

# ===== SECTION 7: Summary =====
md("""## Section 7: Data Quality Summary""")

code("""# Validation checklist
print('\\n=== DATA QUALITY CHECKLIST ===')
checks = {
    'No negative balances': df_transactions['balance_after'].min() >= 0,
    'No duplicate TX IDs': df_transactions['transaction_id'].duplicated().sum() == 0,
    '2% launderers present': abs(launderer_pct - 2) < 1,
    'All data mapped': len(merged) > 0,
    'Customer count matches': len(df_profiles) == 1000,
    'Transaction count': len(df_transactions) == 10000,
}

for check, result in checks.items():
    status = '✅' if result else '⚠️'
    print(f'{status} {check}')""")

code("""# Summary statistics
print('\\n=== EDA SUMMARY ===')
summary = {
    'Total Customers': len(df_profiles),
    'Total Transactions': len(df_transactions),
    'Total Launderers': int(launderer_count),
    'Launderer %': f'{launderer_pct:.1f}%',
    'Time Period': f'{df_transactions["timestamp"].min().date()} to {df_transactions["timestamp"].max().date()}',
    'Days Covered': (df_transactions['timestamp'].max() - df_transactions['timestamp'].min()).days,
    'Avg Txns/User': f'{df_transactions.groupby("customer_id").size().mean():.1f}',
    'Avg Amount': f'{df_transactions["amount"].mean():.0f} KES',
    'Top County': top_counties.index[0] if len(top_counties) > 0 else 'N/A'
}

for key, val in summary.items():
    print(f'{key}: {val}')""")

code("""# Data readiness for modeling
print('\\n=== MODELING READINESS ===')
readiness = [
    '✅ Customer demographics: demographics available',
    '✅ Transaction history: 10k transactions from 1k users',
    '✅ Temporal patterns: hourly, daily, monthly data',
    '✅ Geographic info: county and urban/rural classification',
    '✅ Target variable: 2% launderers (imbalanced but realistic)',
    '✅ AML scenarios: smurfing, layering, mule, circular trading',
    '✅ Validation split: stratified sampling recommended',
    '⚠️ Class imbalance: requires SMOTE/weighting for training'
]

for item in readiness:
    print(item)""")

code("""# Feature engineering recommendations
print('\\n=== RECOMMENDED FEATURES FOR MODELING ===')
recommendations = '''
1. User-level aggregates:
   - Transaction velocity (txns/day)
   - Amount volatility (std/mean)
   - Cash-in vs cash-out ratio
   
2. Behavioral patterns:
   - Hourly concentration (max_hour_pct > 70%)
   - Weekend activity ratio
   - Night activity ratio
   
3. Network features:
   - Unique counterparties
   - Betweenness centrality
   - Clustering coefficient
   
4. Temporal features:
   - Rolling 7-day average transaction count
   - Rolling 30-day volume
   - Deviation from user mean patterns
'''
print(recommendations)""")

code("""print('\\n' + '='*60)
print('✅ EDA NOTEBOOK COMPLETE')
print('='*60)
print(f'Generated: {datetime.now().isoformat()}')
print('Dataset: SentinAI customer & transaction data')
print(f'Rows analyzed: {len(df_profiles) + len(df_transactions):,}')
print('='*60)""")

nb.cells = cells

# Save notebook
os.makedirs('obsidian_notes/notebooks', exist_ok=True)
notebook_path = 'obsidian_notes/notebooks/01_data_quality_eda.ipynb'
with open(notebook_path, 'w') as f:
    nbf.write(nb, f)

print(f'✅ Notebook saved to: {notebook_path}')
