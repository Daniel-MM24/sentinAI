"""
XGBoost Verification Engine for Anomaly Detection.

This module implements an ML verification pipeline using XGBoost to validate
the detectability of injected anomalies. It trains a gradient boosted tree
model on the post-injection dataset and evaluates classification performance.

CRITICAL DATA PREPARATION:
- Drop the anomaly_type column from features (X)
- Isolate anomaly_flag as the target (y)
- Ensure features like previous_fraud_flag_count are completely absent
- Handle 15:85 class imbalance using scale_pos_weight

The verification pipeline:
1. Data Preparation: Split data, drop target columns, isolate features
2. Model Training: Train XGBClassifier with imbalance handling
3. Evaluation: Generate classification report (Precision, Recall, F1)
4. Interpretation: SHAP feature importance analysis
"""

import logging
import numpy as np
import polars as pl
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import shap

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for the XGBoost validation pipeline."""
    test_size: float = 0.2
    random_state: int = 42
    scale_pos_weight: Optional[float] = None  # Auto-calculated if None
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 1
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 10
    shap_sample_size: int = 100  # Number of samples for SHAP analysis


class XGBoostValidator:
    """
    XGBoost verification engine for anomaly detection validation.
    
    This class implements a complete ML pipeline to validate the detectability
    of injected anomalies using gradient boosted trees. It handles:
    - Data preparation with proper feature/target separation
    - Class imbalance handling via scale_pos_weight
    - Model training with early stopping
    - Comprehensive evaluation metrics
    - SHAP-based feature importance analysis
    
    The validator ensures that:
    - anomaly_type is dropped from features (used only for analysis)
    - anomaly_flag is isolated as the target variable
    - No historical fraud counters leak into features
    - All categorical features are properly encoded
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the XGBoostValidator.
        
        Args:
            config: ValidationConfig instance with validation parameters.
                   If None, uses default configuration.
        """
        self.config = config or ValidationConfig()
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        self.shap_values = None
        
        logger.info(
            f"Initialized XGBoostValidator with test_size={self.config.test_size}, "
            f"n_estimators={self.config.n_estimators}"
        )
    
    def _prepare_features(
        self, 
        df: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.Series]:
        """
        Prepare features and target from the input DataFrame.
        
        This method:
        1. Drops anomaly_type from features (not used for training)
        2. Isolates anomaly_flag as target
        3. Drops any historical fraud counters if present
        4. Encodes categorical features
        5. Handles missing values
        
        Args:
            df: Input DataFrame with anomalies injected
            
        Returns:
            Tuple of (features DataFrame, target Series)
            
        Raises:
            ValueError: If required columns are missing
        """
        logger.info("Preparing features and target")
        
        # Validate required columns
        if "anomaly_flag" not in df.columns:
            raise ValueError("anomaly_flag column is required")
        
        # Separate target
        y = df["anomaly_flag"]
        
        # Drop columns that should not be features
        columns_to_drop = ["anomaly_flag", "anomaly_type"]
        
        # Drop any historical fraud counters if present
        fraud_counter_cols = [
            col for col in df.columns 
            if "fraud" in col.lower() or "prev_" in col.lower()
        ]
        columns_to_drop.extend(fraud_counter_cols)
        
        # Also drop non-feature columns
        non_feature_cols = ["entity_id", "timestamp"]
        columns_to_drop.extend(non_feature_cols)
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in columns_to_drop]
        
        logger.info(f"Feature columns: {feature_cols}")
        logger.info(f"Dropped columns: {columns_to_drop}")
        
        X = df.select(feature_cols)
        
        # Encode categorical features
        categorical_cols = X.select(pl.col(pl.String)).columns
        logger.info(f"Categorical columns to encode: {categorical_cols}")
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            # Get unique values and fit encoder
            unique_values = X[col].unique().to_numpy()
            self.label_encoders[col].fit(unique_values)
            
            # Transform
            encoded = self.label_encoders[col].transform(X[col].to_numpy())
            X = X.with_columns([pl.Series(col, encoded)])
        
        # Convert to float for XGBoost
        for col in X.columns:
            if X[col].dtype != pl.Float64:
                X = X.with_columns([pl.col(col).cast(pl.Float64)])
        
        # Handle missing values (fill with 0 for simplicity)
        X = X.fill_null(0)
        
        self.feature_names = feature_cols
        
        logger.info(f"Prepared {len(X.columns)} features for {len(X)} samples")
        
        return X, y
    
    def _calculate_scale_pos_weight(self, y: pl.Series) -> float:
        """
        Calculate scale_pos_weight to handle class imbalance.
        
        Args:
            y: Target series with anomaly_flag values
            
        Returns:
            Scale position weight for XGBoost
        """
        n_negative = (y == 0).sum()
        n_positive = (y == 1).sum()
        
        if n_positive == 0:
            logger.warning("No positive samples in target, setting scale_pos_weight to 1")
            return 1.0
        
        scale_pos_weight = n_negative / n_positive
        logger.info(
            f"Class imbalance: {n_negative} negative, {n_positive} positive. "
            f"scale_pos_weight = {scale_pos_weight:.2f}"
        )
        
        return scale_pos_weight
    
    def train(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Train XGBoost model on the input data.
        
        This method:
        1. Prepares features and target
        2. Splits data into train/test sets
        3. Calculates scale_pos_weight for imbalance handling
        4. Trains XGBClassifier with early stopping
        5. Evaluates model performance
        
        Args:
            df: Input DataFrame with anomalies injected
            
        Returns:
            Dictionary containing training results and metrics
            
        Raises:
            ValueError: If data preparation fails
        """
        logger.info("Starting XGBoost training pipeline")
        
        # Prepare features and target
        X, y = self._prepare_features(df)
        
        # Convert to numpy for sklearn
        X_np = X.to_numpy()
        y_np = y.to_numpy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_np, y_np,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y_np
        )
        
        logger.info(
            f"Train set: {len(X_train)} samples, "
            f"Test set: {len(X_test)} samples"
        )
        
        # Calculate scale_pos_weight if not provided
        if self.config.scale_pos_weight is None:
            scale_pos_weight = self._calculate_scale_pos_weight(
                pl.Series("y_train", y_train)
            )
        else:
            scale_pos_weight = self.config.scale_pos_weight
        
        # Initialize XGBoost classifier
        self.model = xgb.XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            min_child_weight=self.config.min_child_weight,
            gamma=self.config.gamma,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=self.config.random_state,
            eval_metric="logloss",
            use_label_encoder=False
        )
        
        # Train model with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "roc_auc_score": float(auc_score),
            "feature_importance": dict(zip(
                self.feature_names,
                self.model.feature_importances_.tolist()
            )),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "scale_pos_weight": scale_pos_weight
        }
        
        logger.info(
            f"Training complete. ROC AUC: {auc_score:.4f}, "
            f"Precision (anomaly): {report['1']['precision']:.4f}, "
            f"Recall (anomaly): {report['1']['recall']:.4f}, "
            f"F1 (anomaly): {report['1']['f1-score']:.4f}"
        )
        
        return results
    
    def generate_shap_analysis(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Generate SHAP feature importance analysis.
        
        This method:
        1. Prepares features from the input data
        2. Creates a SHAP explainer
        3. Calculates SHAP values for a sample of data
        4. Returns feature importance and SHAP values
        
        Args:
            df: Input DataFrame with anomalies injected
            
        Returns:
            Dictionary containing SHAP analysis results
            
        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model must be trained before SHAP analysis")
        
        logger.info("Generating SHAP feature importance analysis")
        
        # Prepare features
        X, _ = self._prepare_features(df)
        X_np = X.to_numpy()
        
        # Sample for SHAP analysis (for performance)
        if len(X_np) > self.config.shap_sample_size:
            sample_indices = np.random.choice(
                len(X_np), 
                self.config.shap_sample_size, 
                replace=False
            )
            X_sample = X_np[sample_indices]
        else:
            X_sample = X_np
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Calculate feature importance (mean absolute SHAP value)
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Store SHAP values
        self.shap_values = shap_values
        
        results = {
            "feature_importance": dict(zip(self.feature_names, feature_importance.tolist())),
            "shap_values": shap_values.tolist(),
            "sample_size": len(X_sample)
        }
        
        logger.info("SHAP analysis complete")
        
        return results
    
    def plot_shap_summary(self, save_path: Optional[str] = None) -> None:
        """
        Plot SHAP summary plot.
        
        Args:
            save_path: Optional path to save the plot. If None, displays the plot.
            
        Raises:
            ValueError: If SHAP values have not been calculated
        """
        if self.shap_values is None:
            raise ValueError("SHAP values must be calculated before plotting")
        
        logger.info("Generating SHAP summary plot")
        
        # Create summary plot
        shap.summary_plot(
            self.shap_values,
            feature_names=self.feature_names,
            show=False
        )
        
        if save_path:
            import matplotlib.pyplot as plt
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
            plt.close()
        else:
            import matplotlib.pyplot as plt
            plt.show()
    
    def evaluate(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Complete evaluation pipeline including SHAP analysis.
        
        This method:
        1. Trains the XGBoost model
        2. Generates classification metrics
        3. Performs SHAP feature importance analysis
        4. Returns comprehensive evaluation results
        
        Args:
            df: Input DataFrame with anomalies injected
            
        Returns:
            Dictionary containing complete evaluation results
        """
        logger.info("Starting complete evaluation pipeline")
        
        # Train model and get metrics
        training_results = self.train(df)
        
        # Generate SHAP analysis
        shap_results = self.generate_shap_analysis(df)
        
        # Combine results
        evaluation_results = {
            **training_results,
            "shap_analysis": shap_results
        }
        
        return evaluation_results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
            
        Raises:
            ValueError: If model has not been trained
        """
        if self.model is None:
            raise ValueError("Model must be trained before getting feature importance")
        
        importance_dict = dict(zip(
            self.feature_names,
            self.model.feature_importances_.tolist()
        ))
        
        # Sort by importance
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return sorted_importance


def main():
    """Example usage of the XGBoostValidator."""
    logging.basicConfig(level=logging.INFO)
    
    # Import data generation modules
    from src.data.synthetic_generator import CleanDataGenerator, GeneratorConfig
    from src.data.anomaly_injector import FinancialAnomalyInjector, InjectorConfig
    
    # Generate clean data
    config = GeneratorConfig(num_records=10000, num_entities=500, seed=42)
    generator = CleanDataGenerator(config)
    clean_df = generator.generate()
    
    # Inject anomalies
    injector_config = InjectorConfig(anomaly_ratio=0.015, seed=42)
    injector = FinancialAnomalyInjector(injector_config)
    anomalous_df = injector.inject(clean_df)
    
    print("\n=== Data Prepared ===")
    print(f"Shape: {anomalous_df.shape}")
    print(f"Anomaly ratio: {anomalous_df['anomaly_flag'].mean():.2%}")
    
    # Train and evaluate
    validator_config = ValidationConfig(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    validator = XGBoostValidator(validator_config)
    
    results = validator.evaluate(anomalous_df)
    
    print("\n=== Evaluation Results ===")
    print(f"ROC AUC: {results['roc_auc_score']:.4f}")
    print(f"Train samples: {results['train_samples']}")
    print(f"Test samples: {results['test_samples']}")
    print(f"Scale position weight: {results['scale_pos_weight']:.2f}")
    
    print("\n=== Classification Report ===")
    for class_label, metrics in results['classification_report'].items():
        if class_label in ['0', '1']:
            print(f"\nClass {class_label}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1-score']:.4f}")
    
    print("\n=== Feature Importance (XGBoost) ===")
    feature_imp = validator.get_feature_importance()
    for feature, importance in list(feature_imp.items())[:10]:
        print(f"  {feature}: {importance:.4f}")
    
    print("\n=== Feature Importance (SHAP) ===")
    shap_imp = results['shap_analysis']['feature_importance']
    sorted_shap = dict(sorted(shap_imp.items(), key=lambda x: x[1], reverse=True))
    for feature, importance in list(sorted_shap.items())[:10]:
        print(f"  {feature}: {importance:.4f}")


if __name__ == "__main__":
    main()
