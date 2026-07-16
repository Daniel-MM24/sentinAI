"""Model configuration dataclasses for the sentinAI AML model suite."""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class ModelConfig:
    """Base configuration shared across all models."""
    random_state: int = 42
    test_size: float = 0.2
    n_jobs: int = -1


@dataclass
class LightGBMConfig(ModelConfig):
    """LightGBM hyperparameters."""
    model_type: Literal["lightgbm"] = "lightgbm"
    n_estimators: int = 500
    max_depth: int = 7
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: Optional[float] = None  # "auto" → computed from class distribution
    early_stopping_rounds: int = 50
    objective: str = "binary"
    metric: str = "auc"
    verbosity: int = -1
    reg_alpha: float = 1.0
    reg_lambda: float = 2.0
    min_gain_to_split: float = 0.0


@dataclass
class XGBoostConfig(ModelConfig):
    """XGBoost hyperparameters (aligns with existing ValidationConfig + Phase 3 doc)."""
    model_type: Literal["xgboost"] = "xgboost"
    n_estimators: int = 500
    max_depth: int = 7
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    scale_pos_weight: Optional[float] = None  # computed from class distribution
    early_stopping_rounds: int = 50
    objective: str = "binary:logistic"
    eval_metric: str = "auc"
    verbosity: int = 0


@dataclass
class RandomForestConfig(ModelConfig):
    """Random Forest baseline hyperparameters."""
    model_type: Literal["random_forest"] = "random_forest"
    n_estimators: int = 300
    max_depth: int = 10
    min_samples_split: int = 10
    min_samples_leaf: int = 4
    class_weight: str = "balanced"
    max_features: str = "sqrt"


# Registry for easy lookup
CONFIG_REGISTRY = {
    "lightgbm": LightGBMConfig,
    "xgboost": XGBoostConfig,
    "random_forest": RandomForestConfig,
}
