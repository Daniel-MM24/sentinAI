#!/bin/bash
# Comprehensive pipeline execution and validation script
# This script runs the full medallion pipeline (Bronze → Silver → Gold) and validates the output

set -e  # Exit on error

echo "=========================================="
echo "SentinAI Pipeline Execution & Validation"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Step 1: Clean existing data
echo "Step 1: Cleaning existing data directories..."
poetry run python -c "
from pathlib import Path
import shutil
data_dir = Path('data')
for layer in ['bronze', 'silver', 'gold']:
    layer_dir = data_dir / layer
    if layer_dir.exists():
        shutil.rmtree(layer_dir)
        print(f'Cleaned {layer} layer')
    layer_dir.mkdir(parents=True, exist_ok=True)
print_success('Data directories cleaned')
"
echo ""

# Step 2: Run Bronze layer
echo "Step 2: Running Bronze layer..."
poetry run python scripts/run_bronze.py --fast-mode
print_success "Bronze layer completed"
echo ""

# Step 3: Run Silver layer
echo "Step 3: Running Silver layer..."
poetry run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from src.data.medallion_stages import run_silver_stage
from datetime import datetime

result = run_silver_stage(
    partition_key=datetime.now().strftime('%Y-%m-%d'),
    bronze_base_path='data/bronze',
    silver_base_path='data/silver',
)
print(f'Silver layer completed: {result.transaction_count} transactions, {result.customer_count} customers')
"
print_success "Silver layer completed"
echo ""

# Step 4: Run Gold layer
echo "Step 4: Running Gold layer..."
poetry run python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from src.data.medallion_stages import run_gold_stage
from datetime import datetime

result = run_gold_stage(
    partition_key=datetime.now().strftime('%Y-%m-%d'),
    silver_base_path='data/silver',
)
print(f'Gold layer completed: {result.gold_uri}')
"
print_success "Gold layer completed"
echo ""

# Step 5: Run comprehensive validation tests
echo "Step 5: Running comprehensive validation tests..."
poetry run python tests/validate_streamlined_schema.py
print_success "Validation tests completed"
echo ""

# Step 6: Generate validation report
echo "Step 6: Generating validation report..."
poetry run python -c "
import polars as pl
from pathlib import Path
from datetime import datetime

print('=' * 50)
print('PIPELINE VALIDATION REPORT')
print('=' * 50)
print(f'Generated: {datetime.now().isoformat()}')
print('')

# Bronze layer validation
print('BRONZE LAYER:')
from src.data.bronze import BronzeLayer
bronze_layer = BronzeLayer(bronze_base_path='data/bronze')
partition_key = datetime.now().strftime('%Y-%m-%d')
customers_df, transactions_df = bronze_layer.read_normalized_bronze_partition(partition_key)

print(f'  Customers: {customers_df.height}')
print(f'  Transactions: {transactions_df.height}')
print(f'  Customer columns: {len(customers_df.columns)}')
print(f'  Transaction columns: {len(transactions_df.columns)}')
print(f'  Customer columns: {list(customers_df.columns[:5])}...')
print(f'  Transaction columns: {list(transactions_df.columns[:5])}...')
print('')

# Silver layer validation
print('SILVER LAYER:')
silver_path = Path('data/silver/silver_transactions_2026-08-05.parquet')
if silver_path.exists():
    silver_df = pl.read_parquet(silver_path)
    print(f'  Transactions: {silver_df.height}')
    print(f'  Columns: {len(silver_df.columns)}')
    
    temporal_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_night']
    present_temporal = [f for f in temporal_features if f in silver_df.columns]
    print(f'  Temporal features present: {len(present_temporal)}/{len(temporal_features)}')
    print(f'  Present: {present_temporal}')
else:
    print('  Silver data not found')
print('')

# Gold layer validation
print('GOLD LAYER:')
gold_dir = Path('data/gold/features/v1.0')
customer_features_path = gold_dir / f'customer_features_{partition_key}.parquet'
gold_features_path = gold_dir / 'gold_features_consolidated.parquet'

if customer_features_path.exists():
    customer_features_df = pl.read_parquet(customer_features_path)
    print(f'  Customer features: {customer_features_df.height} customers')
    print(f'  Customer feature columns: {len(customer_features_df.columns)}')
    
    engineered_features = ['tx_count_7d', 'volume_7d', 'night_tx_ratio', 'rapid_tx_ratio']
    present_engineered = [f for f in engineered_features if f in customer_features_df.columns]
    print(f'  Engineered features present: {len(present_engineered)}/{len(engineered_features)}')
    print(f'  Present: {present_engineered}')
else:
    print('  Customer features not found')

if gold_features_path.exists():
    gold_features_df = pl.read_parquet(gold_features_path)
    print(f'  Gold features: {gold_features_df.height} transactions')
    print(f'  Gold feature columns: {len(gold_features_df.columns)}')
else:
    print('  Gold features not found')
print('')

# Data quality checks
print('DATA QUALITY CHECKS:')
# Check for nulls in key columns
key_columns = ['customer_id', 'transaction_id', 'amount', 'timestamp']
null_counts = {}
for col in key_columns:
    if col in transactions_df.columns:
        null_counts[col] = transactions_df[col].null_count()

print('  Null counts in key columns:')
for col, count in null_counts.items():
    status = '✓' if count == 0 else '✗'
    print(f'    {status} {col}: {count}')

# Check anomaly rate
if 'anomaly_flag' in transactions_df.columns:
    anomaly_rate = (transactions_df['anomaly_flag'].sum() / transactions_df.height) * 100
    print(f'  Anomaly rate: {anomaly_rate:.2f}%')

print('')
print('=' * 50)
print('VALIDATION COMPLETE')
print('=' * 50)
"
print_success "Validation report generated"
echo ""

echo "=========================================="
echo "Pipeline execution and validation complete!"
echo "=========================================="
