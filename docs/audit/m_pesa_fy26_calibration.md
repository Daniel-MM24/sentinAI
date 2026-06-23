# Architectural & Compliance Justification Report — `m_pesa_fy26_calibrated`

**Change set:** `config/simulation_profiles.yaml` (profile `m_pesa_fy26_calibrated`),
`src/data/validators.py` (Pydantic V2 validation layer).
**Source telemetry:** Audited Safaricom PLC FY2026 Annual Results (M-PESA ecosystem).
**Compliance frame:** Model Risk Management (MRM) — audit-first, type-safe, fail-closed.

## 1. Provenance (Audit-First)

| Parameter | Value | Derivation from audited filing / statute |
|---|---|---|
| `dataset_size` | 40,990,000 | Unique 30-day active customer base (active nodes). |
| Macro velocity | 46.41 B txns/yr | Total annual transaction volume. |
| Global value | KSh 41.68 T | Total annual transaction value. |
| Avg txn value | KSh 898.08 | Global value ÷ annual volume. |
| `clipping_bound` | 250,000.0 | Statutory single-transaction limit (KSh). |

## 2. Categorical Transaction Mix (Closure Guarantee)

```
C2B = 0.46   P2P = 0.38   B2C = 0.11   B2B = 0.05   →   Σ = 1.00
```

C2B-dominant mix reflects the merchant-payment ("Lipa na M-PESA") expansion of the
FY2026 ecosystem. The validator asserts the mass sums to exactly 1.0 (tolerance
1e-9), preventing silent probability-mass leakage in the categorical sampler.

## 3. Log-Normal Amount Distribution (Skew & Tail Fidelity)

Underlying-normal parameters `amount_mean = 6.02` (μ), `amount_std = 1.25` (σ):

```
median = exp(μ)              = exp(6.02)            ≈ KSh 411.6   (kadogo core)
mean   = exp(μ + σ²/2)       = exp(6.02 + 0.78125) ≈ KSh 898.6   (≈ audited 898.08)
```

The median ≪ mean separation reproduces the heavy positive skew of the low-value
"kadogo" economy while the closed-form mean reconciles with the audited macro
average. The σ = 1.25 tail is clipped at the statutory KSh 250,000 ceiling before
DP noise, bounding sensitivity without distorting the bulk of the distribution.

## 4. Temporal Velocity (Exponential Inter-Arrival)

`velocity_lambda = 320.0` minutes models per-entity inter-arrival as an
exponential process over a 16-hour (960-minute) daily waking window
(≈ 3 transactions/active-entity/day), matching individual human-agent behavior
rather than aggregate machine cadence.

## 5. Differential Privacy Meta-Parameters

| Parameter | Value | Role |
|---|---|---|
| `dataset_size` | 40,990,000 | δ denominator (δ = 1 / (N·10)). |
| `total_queries_per_year` | 50,000 | ε budget allocation across the annual query set. |
| `clipping_bound` | 250,000.0 | Global query sensitivity ceiling = statutory limit. |

The clipping bound is bound to the legal maximum so DP sensitivity can never be
configured above the statutory transaction limit.

## 6. Type-Safety & Enforcement (`src/data/validators.py`)

`SimulationProfile` (Pydantic V2, `extra="forbid"`, `frozen=True`) enforces:

- `transaction_type_probs`: each component in (0, 1]; `@model_validator(mode="after")`
  asserts Σ = 1.0 exactly.
- `clipping_bound`: field bound `le=250000.0` **and** an after-validator re-asserting
  it does not exceed `LEGAL_TRANSACTION_LIMIT` (defense in depth).
- All remaining numerics strictly typed and positively bounded for low-overhead,
  deterministic serialization in the production K8s path.

`load_simulation_profiles()` validates the entire YAML registry at load time,
guaranteeing no unvalidated profile reaches `DistributionParams`.

## 7. Modular Integrity

Changes are confined to the permitted surfaces — `config/` and `src/data/` (plus
this audit doc). No modification to `src/agents/` or `src/datasets/`.
