# Statistical Distributions Used

## Core M-Pesa Sampling
*(From `distribution_sampler.py`)*

| Feature | Distribution | Parameters |
| :--- | :--- | :--- |
| **Transaction Amount** | Log-Normal | μ = 5.95, σ = 1.25<br>Clipped to [1.0, 250000.0] |
| **Inter-Arrival Times** | Exponential | scale = 320.0 |
| **Anomaly Flags** | Fixed contamination rate | 0.15%<br>Anomaly types sampled categorically from predefined FMS vectors |

---

## Synthetic Distribution Engine
*(From `synthetic_distributions.py`)*

### Transaction Amounts (Mixture Model)

| Segment | Proportion | Distribution | Scaling |
| :--- | :--- | :--- | :--- |
| **Kadogo** | 45% | Beta(2, 5) | Scaled to [1.0, 100.0] |
| **Standard** | 50% | Truncated Log-Normal (s = 1.0) | Scaled to [101.0, 150000.0] |
| **Premium** | 5% | Pareto Type I (shape b = 2.0) | Mapped to [150001.0, 250000.0] |

### Timestamps
- **Model**: Non-homogeneous Poisson process approximation
- **Method**: Uniform proposals + acceptance based on time-of-day intensity function

### Latent Balances
| Feature | Distribution | Parameters |
| :--- | :--- | :--- |
| **Balances** | Log-Normal | s = 1.0, scale = 1000.0 |

### Credit Triggers
| Feature | Distribution | Type |
| :--- | :--- | :--- |
| `is_fuliza` | Conditional Bernoulli | Binomial trials |
| `is_mshwari` | Conditional Bernoulli | Binomial trials |

### Channels
| Channel | Probability |
| :--- | :--- |
| USSD | 0.5 |
| App | 0.4 |
| STK_Push | 0.1 |

---

## Clean/Legacy Synthetic Generator
*(From `synthetic_generator.py`)*

### Feature Transformations

| Feature | Transformation Method |
| :--- | :--- |
| **Baseline Features** | Correlated from multivariate normal |
| **Transaction Amounts** | Log-normal via exp(normal) |
| **Volume** | Exponential-like via exp(normal / 2) |
| **Account Balances** | Log-normal via exp(normal + 8) |
| **Transaction Count** | Poisson-like via round(exp(normal)) |

### Temporal Patterns
| Feature | Distribution | Parameters |
| :--- | :--- | :--- |
| **Inter-Arrival Times** | Exponential | velocity_lambda = 15.0 (scale = 1/15) |

---

## DP/Noise Layer
*(From `synthetic_engine.py`)*

| Feature | Mechanism | Parameters |
| :--- | :--- | :--- |
| **Differential Privacy Noise** | Laplace | scale (variable per configuration) |

---

## Exact Parameter Values in Current Profile
*(From `simulation_profiles.yaml` under `profiles.m_pesa_fy26_calibrated`)*

### Transaction Type Probabilities

| Transaction Type | Probability |
| :--- | :--- |
| C2B | 0.46 |
| P2P | 0.38 |
| B2C | 0.11 |
| B2B | 0.05 |

### Amount Distribution Parameters

| Parameter | Value |
| :--- | :--- |
| `amount_mean` | 6.02 |
| `amount_std` | 1.25 |

### Temporal Parameters

| Parameter | Value |
| :--- | :--- |
| `velocity_lambda` | 320.0 |

### Differential Privacy / Audit Parameters

| Parameter | Value |
| :--- | :--- |
| `dataset_size` | 40,990,000 |
| `total_queries_per_year` | 50,000 |
| `query_type` | "standard" |
| `clipping_bound` | 250,000.0 |
| `seed` | 42 |

### Model Metadata

| Parameter | Value |
| :--- | :--- |
| `model_version` | "fy26.1.0" |

---

## Derived Distribution Behavior
*(Using profile values in `synthetic_engine.py`)*

### Transaction Amounts
- Sampled from Log-Normal with:
  - `mean = 6.02`
  - `sigma = 1.25`

### Inter-Arrival Times
- Exponential with `lambda = 320.0`
  - **Note**: In some code paths, this translates to `scale = 1/320.0`; in `MpesaDistributionSampler`, it uses `scale = 320.0`

### Differential Privacy Noise
- Laplace noise with calibrated scale:
  - `epsilon = 1.0 / 50000 = 0.00002`
  - `delta = 1 / (40990000 * 10) ≈ 2.439e-09`

---

## Default Generator Parameter Fallbacks
*(From `synthetic_engine.py` default `DistributionParams`)*

| Parameter | Default Value |
| :--- | :--- |
| `amount_mean` | 5.0 |
| `amount_std` | 1.0 |
| `velocity_lambda` | 10.0 |
| `dataset_size` | 500,000 |
| `total_queries_per_year` | 12 |
| `query_type` | "standard" |
| `clipping_bound` | 10,000.0 |
| `seed` | 42 |
| `model_version` | "v1.0" |

> **Note**: The active current profile overrides the defaults with the FY26-calibrated values listed above.