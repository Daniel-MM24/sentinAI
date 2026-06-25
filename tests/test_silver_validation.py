"""
Validation test suite for src/datasets/silver.py and src/datasets/schemas.py.

Performs:
1. Syntax validation (AST parse)
2. Import validation (all external dependencies resolve)
3. Unit tests for SilverLayer logic

Run: python tests/test_silver_validation.py
"""

import ast
import sys
import os
import tempfile
import logging

# ---------------------------------------------------------------------------
# Phase 1: Syntax validation
# ---------------------------------------------------------------------------
print("=" * 60)
print("PHASE 1: Syntax Validation")
print("=" * 60)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files_to_check = [
    os.path.join(ROOT, "src", "datasets", "silver.py"),
    os.path.join(ROOT, "src", "datasets", "schemas.py"),
    os.path.join(ROOT, "src", "datasets", "__init__.py"),
]

for fpath in files_to_check:
    try:
        with open(fpath) as f:
            ast.parse(f.read(), filename=fpath)
        print(f"  ✓ {os.path.basename(fpath)}: syntax OK")
    except SyntaxError as e:
        print(f"  ✗ {os.path.basename(fpath)}: SYNTAX ERROR — {e}")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Phase 2: Import validation
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("PHASE 2: Import Validation")
print("=" * 60)

sys.path.insert(0, ROOT)

import_errors = []

deps = {
    "polars": "polars",
    "great_expectations": "great-expectations",
    "great_expectations.core": "great-expectations",
    "openlineage.client": "openlineage-python",
    "openlineage.client.run": "openlineage-python",
    "rapidfuzz": "rapidfuzz",
    "duckdb": "duckdb",
    "pandera.polars": "pandera[polars]",
}

for mod, pip_name in deps.items():
    try:
        __import__(mod)
        print(f"  ✓ {mod}")
    except ImportError:
        print(f"  ✗ {mod}  — install via: pip install {pip_name}")
        import_errors.append(pip_name)

if import_errors:
    unique_pkgs = list(dict.fromkeys(import_errors))
    print(f"\n  ⚠  Missing packages: {', '.join(unique_pkgs)}")
    print("  Run: pip install " + " ".join(f'"{p}"' for p in unique_pkgs))
    sys.exit(1)

# Now import our modules
try:
    from src.datasets.schemas import SilverRecordSchema
    print("  ✓ src.datasets.schemas")
except Exception as e:
    print(f"  ✗ src.datasets.schemas — {e}")
    sys.exit(1)

try:
    from src.datasets.silver import SilverLayer, CircuitBreakerError
    print("  ✓ src.datasets.silver")
except Exception as e:
    print(f"  ✗ src.datasets.silver — {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Phase 3: Unit tests
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("PHASE 3: Unit Tests")
print("=" * 60)

import polars as pl
from great_expectations.core import ExpectationSuite
from unittest.mock import MagicMock, patch
import duckdb as duckdb_mod

passed = 0
failed = 0


def run_test(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  ✓ {name}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        failed += 1


def make_layer(**kwargs):
    """Create a SilverLayer with mocked external dependencies."""
    mock_ol = MagicMock()
    suite = ExpectationSuite(expectation_suite_name="test_suite")
    tmp_db = os.path.join(tempfile.mkdtemp(), "test_drift.duckdb")
    return SilverLayer(
        expectation_suite=suite,
        ol_client=mock_ol,
        drift_db_path=tmp_db,
    )


def make_sample_df():
    """Create a minimal valid Bronze dataframe."""
    return pl.DataFrame({
        "customer_id": ["C001", "C002"],
        "customer_name": [" Alice Smith ", "bob jones "],
        "email": ["ALICE@EXAMPLE.COM ", " bob@test.com"],
        "tax_id": ["TAX001", "TAX002"],
        "currency": ["usd", None],
        "amount": [100.50, 200.00],
        "timestamp": ["2024-01-15T10:30:00", "2024-02-20T14:00:00"],
    })


# ---------- Test: _clean_and_standardize ----------
def test_clean_and_standardize():
    layer = make_layer()
    df = make_sample_df()
    result = layer._clean_and_standardize(df)
    assert result["email"][0] == "alice@example.com", f"Email not lowercased: {result['email'][0]}"
    assert result["customer_name"][0] == "ALICE SMITH", f"Name not uppercased: {result['customer_name'][0]}"
    assert result["currency"][1] == "USD", f"Null currency not filled: {result['currency'][1]}"

run_test("_clean_and_standardize: normalizes email/name/currency", test_clean_and_standardize)


# ---------- Test: _split_fact_and_dimension (dedup) ----------
def test_split_fact_and_dimension():
    layer = make_layer()
    df = pl.DataFrame({
        "customer_id": ["C001", "C001", "C002"],
        "customer_name": ["ALICE SMITH", "ALICE SMITH", "BOB JONES"],
        "email": ["alice@example.com", "alice@example.com", "bob@test.com"],
        "tax_id": ["TAX001", "TAX001", "TAX002"],
        "currency": ["USD", "USD", "USD"],
        "amount": [100.0, 200.0, 300.0],
        "timestamp": ["2024-01-15T10:30:00", "2024-02-20T14:00:00", "2024-03-01T09:00:00"],
    })
    df = df.with_columns(pl.col("timestamp").str.to_datetime())
    transaction_fact_df, customer_dimension_df = layer._split_fact_and_dimension(df)
    assert transaction_fact_df.height == 3 and customer_dimension_df.height == 2, f"Expected 2 rows after dedup, got {result.height}"
    assert "golden_record_id" in customer_dimension_df.columns, "Missing golden_record_id column"

run_test("_split_fact_and_dimension: collapses duplicates and adds golden_record_id", test_split_fact_and_dimension)


# ---------- Test: _generate_data_drift_report (DuckDB) ----------
def test_drift_duckdb():
    layer = make_layer()
    bronze = pl.DataFrame({
        "customer_id": ["C001", "C002", "C003"],
        "customer_name": ["A", "B", "C"],
        "email": ["a@x.com", "b@x.com", "c@x.com"],
        "tax_id": ["T1", None, "T3"],
        "currency": ["USD", "USD", "USD"],
        "amount": [1.0, 2.0, 3.0],
        "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
    })
    silver = bronze.head(2)
    layer._generate_data_drift_report(bronze, silver, silver)

    # Verify DuckDB wrote a row
    con = duckdb_mod.connect(layer._drift_db_path)
    rows = con.execute("SELECT * FROM drift_reports").fetchall()
    con.close()
    assert len(rows) == 1, f"Expected 1 drift report row, got {len(rows)}"
    assert rows[0][3] == 3, f"Expected bronze_row_count=3, got {rows[0][3]}"

run_test("_generate_data_drift_report: persists to DuckDB", test_drift_duckdb)


# ---------- Test: CircuitBreakerError exists and is raisable ----------
def test_circuit_breaker_error():
    try:
        raise CircuitBreakerError("test failure")
    except CircuitBreakerError as e:
        assert "test failure" in str(e)

run_test("CircuitBreakerError: is raisable", test_circuit_breaker_error)


# ---------- Test: _build_strict_compliance_suite ----------
def test_default_suite():
    mock_ol = MagicMock()
    layer = SilverLayer(ol_client=mock_ol)
    assert layer.expectation_suite is not None
    assert layer.expectation_suite.expectation_suite_name == "strict_financial_compliance_suite"
    # Verify it has expectations configured
    expectations = layer.expectation_suite.expectations
    assert len(expectations) >= 3, f"Expected ≥3 expectations, got {len(expectations)}"

run_test("_build_strict_compliance_suite: generates default suite with expectations", test_default_suite)


# ---------- Test: ER_LOGIC_VERSION is set ----------
def test_er_version():
    assert SilverLayer.ER_LOGIC_VERSION == "v1.0.0"
    assert SilverLayer.FUZZY_MATCH_THRESHOLD == 0.85

run_test("Class constants: ER_LOGIC_VERSION and FUZZY_MATCH_THRESHOLD", test_er_version)


# ---------- Test: OpenLineage emit called during transform ----------
def test_lineage_emission():
    layer = make_layer()
    df = make_sample_df()
    with patch.object(layer, "_validate_quality", return_value=(df, pl.DataFrame(), {})):
        with patch.object(layer, "_split_fact_and_dimension", return_value=(df, df)):
            with patch.object(layer, "_generate_data_drift_report"):
                layer.transform_to_silver(df)

    # OL client should have been called: START + COMPLETE = 2 emit calls
    assert layer.ol_client.emit.call_count == 2, (
        f"Expected 2 OL emit calls, got {layer.ol_client.emit.call_count}"
    )

run_test("OpenLineage: emits START and COMPLETE events", test_lineage_emission)

# ---------- Test: Lineage failure on transform error ----------
def test_lineage_on_failure():
    layer = make_layer()
    df = make_sample_df()
    with patch.object(layer, "_clean_and_standardize", side_effect=ValueError("boom")):
        try:
            layer.transform_to_silver(df)
        except ValueError:
            pass
    # Should have emitted START + FAIL = 2 calls
    assert layer.ol_client.emit.call_count == 2, (
        f"Expected 2 OL emit calls on failure, got {layer.ol_client.emit.call_count}"
    )

run_test("OpenLineage: emits START and FAIL on error", test_lineage_on_failure)

# ---------- Test: partition_key flows through ----------
def test_partition_key():
    layer = make_layer()
    df = make_sample_df()
    with patch.object(layer, "_validate_quality", return_value=(df, pl.DataFrame(), {})):
        with patch.object(layer, "_split_fact_and_dimension", return_value=(df, df)):
            with patch.object(layer, "_generate_data_drift_report") as mock_drift:
                layer.transform_to_silver(df, partition_key="2024-01-15")
    mock_drift.assert_called_once()
    _, kwargs = mock_drift.call_args
    assert kwargs.get("partition_key") == "2024-01-15"

run_test("transform_to_silver: passes partition_key to drift report", test_partition_key)



# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
total = passed + failed
print(f"RESULTS: {passed}/{total} tests passed, {failed} failed")
print("=" * 60)
if failed > 0:
    sys.exit(1)
else:
    print("✓ All validations passed!")

# ---------- Test: Fact-Dimension Separation Preserves Transaction History ----------
def test_fact_dimension_separation_preserves_history():
    """Test that duplicate customer records with distinct timestamps are preserved as unique history rows."""
    layer = make_layer()
    
    # Create test data: same customer with 3 distinct transactions
    df = pl.DataFrame({
        "customer_id": ["C001", "C001", "C001"],
        "customer_name": ["ALICE SMITH", "ALICE SMITH", "ALICE SMITH"],
        "email": ["alice@example.com", "alice@example.com", "alice@example.com"],
        "tax_id": ["TAX001", "TAX001", "TAX001"],
        "currency": ["USD", "USD", "USD"],
        "amount": [100.0, 200.0, 300.0],
        "timestamp": ["2024-01-15T10:30:00", "2024-02-20T14:00:00", "2024-03-01T09:00:00"],
    })
    df = df.with_columns(pl.col("timestamp").str.to_datetime())
    
    # Split into fact and dimension
    transaction_fact_df, customer_dimension_df = layer._split_fact_and_dimension(df)
    
    # Assert transaction fact stream preserves all 3 events
    assert transaction_fact_df.height == 3, f"Expected 3 transaction events, got {transaction_fact_df.height}"
    
    # Assert customer dimension registry collapses to 1 unique customer
    assert customer_dimension_df.height == 1, f"Expected 1 unique customer, got {customer_dimension_df.height}"
    
    # Assert golden_record_id is present in dimension
    assert "golden_record_id" in customer_dimension_df.columns, "Missing golden_record_id in customer dimension"
    
    # Assert fact stream has golden_record_id for join capability
    assert "golden_record_id" in transaction_fact_df.columns, "Missing golden_record_id in transaction fact stream"

run_test("Fact-Dimension Separation: preserves transaction history for duplicate customers", test_fact_dimension_separation_preserves_history)
