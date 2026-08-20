"""
TVAE-based Synthetic Data Generator
Responsible for training a Tabular VAE on isolated M-PESA events and sampling from it.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from scipy import stats

try:
    from ctgan import TVAE
except ImportError:
    # Fallback if ctgan is not installed yet (incompatible with Python 3.12)
    TVAE = None

class TVAEGenerator:
    """
    Hybrid TVAE synthetic data generator. 
    Focuses only on core event features.
    
    Falls back to statistical sampling when ctgan is unavailable (Python 3.12+).
    """
    def __init__(self, epochs: int = 100, batch_size: int = 500):
        self.epochs = epochs
        self.batch_size = batch_size
        
        # We define our discrete categorical columns for the TVAE
        self.discrete_columns = [
            'customer_id',
            'tier',
            'customer_tier',  # Alternative tier column name
            'archetype',
            'transaction_type',
            'direction',
            'is_international'
        ]
        
        # Columns to exclude from distribution fitting (IDs, etc.)
        self.exclude_columns = [
            'transaction_id',
            'sender_account_id',
            'receiver_account_id',
            'customer_id',  # customer_id will be sampled from existing values
            'counterparty_id',
            'receiver_id'
        ]
        
        self.use_fallback = False
        self.baseline_data = None
        self.column_distributions = None
        
        if TVAE is not None:
            self.model = TVAE(
                epochs=self.epochs, 
                batch_size=self.batch_size
            )
        else:
            self.model = None
            self.use_fallback = True
            print("Warning: CTGAN/TVAE not installed (incompatible with Python 3.12). Using statistical fallback.")

    def fit(self, df: pd.DataFrame):
        """
        Train the TVAE on clean, historical/baseline synthetic data.
        Uses statistical fallback when ctgan is unavailable.
        """
        if self.use_fallback:
            print(f"Using statistical fallback on {len(df)} records...")
            self._fit_statistical_fallback(df)
            print("Statistical fallback training complete.")
            return
            
        if self.model is None:
            raise RuntimeError("TVAE model is not initialized.")
            
        print(f"Training TVAE on {len(df)} records...")
        
        # Ensure correct types before fitting
        train_df = df.copy()
        
        # For dates, TVAE handles them if they are datetime objects or we can extract numerical features
        # Assuming timestamp is already handled or we can extract unix epoch
        if 'timestamp' in train_df.columns:
            # Simple approach: convert to numerical seconds for the VAE
            # We will reconstruct proper datetimes post-generation
            if pd.api.types.is_datetime64_any_dtype(train_df['timestamp']):
                train_df['timestamp_numeric'] = train_df['timestamp'].astype(np.int64) // 10**9
            else:
                train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
                train_df['timestamp_numeric'] = train_df['timestamp'].astype(np.int64) // 10**9
                
            train_df = train_df.drop('timestamp', axis=1)
            
        self.model.fit(train_df, self.discrete_columns)
        print("Training complete.")

    def _fit_statistical_fallback(self, df: pd.DataFrame):
        """
        Fit statistical distributions for fallback sampling.
        """
        self.baseline_data = df.copy()
        self.column_distributions = {}
        
        for col in df.columns:
            if col in self.exclude_columns:
                # Skip ID columns - will regenerate during sampling
                continue
            elif col == 'timestamp':
                # Store timestamp range for temporal sampling
                self.column_distributions[col] = {
                    'type': 'datetime',
                    'min': df[col].min(),
                    'max': df[col].max()
                }
            elif col in self.discrete_columns or df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # Categorical: store value probabilities
                value_counts = df[col].value_counts(normalize=True)
                if len(value_counts) > 0:
                    self.column_distributions[col] = {
                        'type': 'categorical',
                        'values': value_counts.index.tolist(),
                        'probabilities': value_counts.values.tolist()
                    }
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Continuous: fit distribution parameters
                self.column_distributions[col] = {
                    'type': 'continuous',
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
            else:
                # Skip unknown types
                continue

    def sample(self, n_samples: int) -> pd.DataFrame:
        """
        Sample new synthetic events from the trained TVAE.
        Uses statistical fallback when ctgan is unavailable.
        """
        if self.use_fallback:
            print(f"Sampling {n_samples} records using statistical fallback...")
            return self._sample_statistical_fallback(n_samples)
            
        if self.model is None:
            raise RuntimeError("TVAE model is not initialized.")
            
        print(f"Sampling {n_samples} records from TVAE...")
        synthetic_data = self.model.sample(n_samples)
        
        # Reconstruct timestamp
        if 'timestamp_numeric' in synthetic_data.columns:
            synthetic_data['timestamp'] = pd.to_datetime(synthetic_data['timestamp_numeric'], unit='s')
            synthetic_data = synthetic_data.drop('timestamp_numeric', axis=1)
            
        # Ensure amounts are non-negative (VAE might output negative continuous values)
        if 'amount' in synthetic_data.columns:
            synthetic_data['amount'] = synthetic_data['amount'].clip(lower=1.0).round(2)
            
        return synthetic_data

    def _sample_statistical_fallback(self, n_samples: int) -> pd.DataFrame:
        """
        Sample using statistical distributions from baseline data.
        """
        synthetic_data = pd.DataFrame(index=range(n_samples))
        
        # Generate transaction IDs
        synthetic_data['transaction_id'] = [f"TXN_{i:010d}" for i in range(n_samples)]
        
        # Generate account IDs if they were in original data
        if 'sender_account_id' in self.baseline_data.columns:
            synthetic_data['sender_account_id'] = [f"ACC_{i:010d}" for i in range(n_samples)]
        if 'receiver_account_id' in self.baseline_data.columns:
            synthetic_data['receiver_account_id'] = [f"ACC_{i+n_samples:010d}" for i in range(n_samples)]
        
        # Sample customer_id from existing values
        if 'customer_id' in self.baseline_data.columns:
            customer_ids = self.baseline_data['customer_id'].dropna().unique()
            if len(customer_ids) > 0:
                synthetic_data['customer_id'] = np.random.choice(customer_ids, size=n_samples)
            else:
                # Fallback: generate synthetic customer IDs
                synthetic_data['customer_id'] = [f"CUST_{i:010d}" for i in np.random.randint(0, 10000, n_samples)]
        
        for col, dist_info in self.column_distributions.items():
            if dist_info['type'] == 'datetime':
                # Sample timestamps uniformly within the baseline range
                time_range = (dist_info['max'] - dist_info['min']).total_seconds()
                random_seconds = np.random.uniform(0, time_range, n_samples)
                synthetic_data[col] = dist_info['min'] + pd.to_timedelta(random_seconds, unit='s')
            elif dist_info['type'] == 'categorical':
                # Sample from categorical distribution
                if len(dist_info['values']) > 0 and len(dist_info['probabilities']) > 0:
                    synthetic_data[col] = np.random.choice(
                        dist_info['values'],
                        size=n_samples,
                        p=dist_info['probabilities']
                    )
                else:
                    # Fallback: sample from baseline if distribution is empty
                    if col in self.baseline_data.columns:
                        baseline_values = self.baseline_data[col].dropna().values
                        if len(baseline_values) > 0:
                            synthetic_data[col] = np.random.choice(baseline_values, size=n_samples)
                        else:
                            # Final fallback: use most common value or placeholder
                            print(f"Warning: No valid values for column {col}, using placeholder")
                            synthetic_data[col] = "UNKNOWN"
            elif dist_info['type'] == 'continuous':
                # Sample from truncated normal distribution
                if dist_info['std'] > 0:  # Avoid division by zero
                    samples = stats.truncnorm.rvs(
                        (dist_info['min'] - dist_info['mean']) / dist_info['std'],
                        (dist_info['max'] - dist_info['mean']) / dist_info['std'],
                        loc=dist_info['mean'],
                        scale=dist_info['std'],
                        size=n_samples
                    )
                else:
                    # If std is 0, just use the mean
                    samples = np.full(n_samples, dist_info['mean'])
                synthetic_data[col] = samples
        
        # Ensure amounts are non-negative and properly rounded
        if 'amount' in synthetic_data.columns:
            synthetic_data['amount'] = synthetic_data['amount'].clip(lower=1.0).round(2)
        
        return synthetic_data
