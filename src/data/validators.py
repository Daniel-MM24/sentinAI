"""
Type-Safe Validation Layer for Synthetic Simulation Profiles (Pydantic V2)

This module enforces Model Risk Management (MRM) constraints on the distribution
profiles defined in ``config/simulation_profiles.yaml`` before they are handed to
``src.data.synthetic_engine.DistributionParams``.

Guarantees enforced here (audit-first, fail-closed):
    1. Categorical transaction probabilities are well-formed and sum to exactly
       1.0 (no probability mass leakage).
    2. The Differential Privacy clipping bound never exceeds the Kenyan statutory
       single-transaction limit of KSh 250,000.00.
    3. All numeric parameters are strictly typed and bounded, keeping
       serialization low-overhead and deterministic for K8s throughput.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Kenyan statutory ceiling for a single M-PESA operation (KSh).
LEGAL_TRANSACTION_LIMIT: float = 250_000.0

# Tolerance for floating-point comparison of the categorical probability mass.
_PROBABILITY_SUM_TOLERANCE: float = 1e-9


class SimulationProfile(BaseModel):
    """Validated, type-safe representation of a single simulation profile.

    Field names mirror ``DistributionParams`` so a validated profile can be
    splatted directly into the generator: ``DistributionParams(**profile.dict())``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- Provenance / audit metadata ---
    model_version: str = Field(
        default="v1.0", description="Version tag of the generator calibration."
    )
    description: str = Field(
        default="", description="Human-readable provenance of the profile."
    )
    source: str = Field(
        default="", description="Audited filing the parameters are derived from."
    )

    # --- Categorical transaction mix ---
    transaction_type_probs: dict[str, float] = Field(
        ..., description="Transaction-type categorical probabilities (sum == 1.0)."
    )

    # --- Log-normal amount distribution ---
    amount_mean: float = Field(
        ..., gt=0.0, description="mu of the underlying normal for amounts."
    )
    amount_std: float = Field(
        ..., gt=0.0, description="sigma of the underlying normal for amounts."
    )

    # --- Temporal velocity ---
    velocity_lambda: float = Field(
        ...,
        gt=0.0,
        description="Mean minutes between transactions per unique entity.",
    )

    # --- Differential Privacy meta-parameters ---
    dataset_size: int = Field(
        ..., gt=0, description="Distinct active entities (Delta denominator)."
    )
    total_queries_per_year: int = Field(
        ..., gt=0, description="Annual query budget for epsilon allocation."
    )
    query_type: str = Field(default="standard", description="Query sensitivity class.")
    clipping_bound: float = Field(
        ...,
        gt=0.0,
        le=LEGAL_TRANSACTION_LIMIT,
        description="DP sensitivity ceiling; bounded by the statutory limit.",
    )

    # --- Reproducibility ---
    seed: int = Field(default=42, ge=0, description="Deterministic RNG seed.")

    @field_validator("transaction_type_probs")
    @classmethod
    def _validate_probability_components(
        cls, value: dict[str, float]
    ) -> dict[str, float]:
        """Each categorical probability must be a real number in (0, 1]."""
        if not value:
            raise ValueError("transaction_type_probs must not be empty.")
        for label, prob in value.items():
            if not 0.0 < prob <= 1.0:
                raise ValueError(
                    f"Probability for '{label}' must be in (0, 1]; got {prob}."
                )
        return value

    @model_validator(mode="after")
    def _validate_probability_mass(self) -> Self:
        """Categorical probabilities must sum to exactly 1.0 (MRM closure)."""
        total = sum(self.transaction_type_probs.values())
        if abs(total - 1.0) > _PROBABILITY_SUM_TOLERANCE:
            raise ValueError(
                "transaction_type_probs must sum to exactly 1.0; "
                f"got {total} from {self.transaction_type_probs}."
            )
        return self

    @model_validator(mode="after")
    def _validate_clipping_within_legal_limit(self) -> Self:
        """DP clipping bound must not exceed the statutory transaction limit."""
        if self.clipping_bound > LEGAL_TRANSACTION_LIMIT:
            raise ValueError(
                f"clipping_bound {self.clipping_bound} exceeds the legal limit "
                f"of {LEGAL_TRANSACTION_LIMIT}."
            )
        return self


class SimulationProfileRegistry(BaseModel):
    """Top-level schema for ``config/simulation_profiles.yaml``."""

    model_config = ConfigDict(extra="forbid")

    profiles: dict[str, SimulationProfile] = Field(
        ..., description="Named simulation profiles keyed by profile id."
    )

    def get(self, name: str) -> SimulationProfile:
        """Return a validated profile by name, raising ``KeyError`` if absent."""
        return self.profiles[name]


def load_simulation_profiles(
    path: str | Path = "config/simulation_profiles.yaml",
) -> SimulationProfileRegistry:
    """Load and fully validate the simulation profile registry from YAML."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return SimulationProfileRegistry.model_validate(raw)
