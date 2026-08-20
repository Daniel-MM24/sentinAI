# Data Collection vs. Data Provisioning

**Yes — but only if you label it clearly as a substitute rather than pretending it was real collection.**

---

## Rationale

In a project like this, the "data collection" step can reasonably be replaced by one of the following:

- Synthetic data generation
- Simulated or profile-based data
- Publicly available / open datasets
- Manually curated sample data

---

## Requirements for Substitution

This approach is acceptable **if** you explicitly state:

| Requirement | Description |
| :--- | :--- |
| **Why real data was not available** | Justify the absence of genuine production data (e.g., privacy constraints, regulatory barriers, proprietary restrictions) |
| **What substitute was used** | Clearly identify the type of synthetic/simulated/public data employed |
| **How the substitute was validated** | Describe the validation methodology (e.g., statistical comparisons, distribution matching, expert review) |
| **What limitations it introduces** | Acknowledge the constraints and potential biases introduced by using substitutes |

---

## Recommended Phrasing

| ❌ Weak Phrasing | ✅ Stronger Phrasing |
| :--- | :--- |
| "Data collection" | **"Data provisioning"** |
| "We collected transaction data" | **"Synthetic data generation for experimentation and validation"** |
| "The dataset was obtained from..." | **"We provisioned a synthetic dataset calibrated to M-Pesa-like distributions..."** |

---

## Important Distinction

> **If the project is meant to be production-grade or evidence-based, then this should be described as a proxy or fallback, not as full real-world data collection.**

---

## Summary

| Aspect | Recommendation |
| :--- | :--- |
| **Terminology** | Use "Data Provisioning" or "Synthetic Data Generation" |
| **Transparency** | Clearly label the data as a substitute |
| **Documentation** | Include justification, validation method, and limitations |
| **Honesty** | Never imply the data came from actual production systems |
