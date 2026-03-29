from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

class DataProcessor(ABC):
    """
    Abstract base class for data transformations.
    Handles: Raw DF -> Tensors -> Inverse Transform to Prices.
    """
    def __init__(self):
        self.scaler = None

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms raw price DataFrame into processed features (e.g., scaled log-returns).
        Returns: (N_samples, Features) array (flattened time, or windows handled later)
        """
        pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray, initial_prices: np.ndarray) -> pd.DataFrame:
        """
        Reconstructs prices from processed data.
        Args:
            data: (Seq_Len, Features) or (Batch, Seq_Len, Features)
            initial_prices: (Features,) price vector to start reconstruction from.
        """
        pass
    
    def save(self, path):
        joblib.dump(self, path)
        
    @staticmethod
    def load(path):
        return joblib.load(path)

class LogReturnProcessor(DataProcessor):
    """
    Standard processor: Prices -> Log Returns -> StandardScaler.

    Supports masked data where some values are missing. When a mask is provided:
    - fit() fits the scaler only on valid (non-masked) values
    - transform() returns (data, mask) tuple where mask propagates NaN positions
    """
    def __init__(self, scaling='global', min_periods=60):
        super().__init__()
        self.scaling = scaling
        self.min_periods = min_periods
        self.scaler = StandardScaler()
        self._fitted_expanding = False
        self.expanding_means_ = None
        self.expanding_stds_ = None

    def _compute_masked_log_returns(self, df, mask):
        """
        Compute log returns with mask propagation.

        Args:
            df: Price DataFrame
            mask: Mask DataFrame (1=valid, 0=missing)

        Returns:
            tuple: (log_returns_array, mask_array) where mask accounts for shift
        """
        mask_shifted = mask.shift(1)
        return_mask = (mask * mask_shifted).iloc[1:]
        mask_array = return_mask.values

        df_prev = df.shift(1).iloc[1:]
        df_curr = df.iloc[1:]

        df_prev_safe = df_prev.where(df_prev != 0, 1.0)
        ratio = df_curr / df_prev_safe
        ratio = ratio.where(return_mask.astype(bool), 1.0)

        log_returns_array = np.log(ratio.values)
        return log_returns_array, mask_array

    def fit(self, df: pd.DataFrame, mask: pd.DataFrame = None):
        """
        Fit scaler on log returns.

        Args:
            df: Price DataFrame
            mask: Optional mask DataFrame (1=valid, 0=missing)
                  When provided, fits only on valid values
        """
        if mask is not None:
            log_returns_array, mask_array = self._compute_masked_log_returns(df, mask)
            mask_bool = mask_array.astype(bool)

            if self.scaling == 'expanding':
                self._fit_expanding_masked(log_returns_array, mask_bool)
            else:
                # Compute mean and std per feature using only valid values
                means = []
                stds = []
                for i in range(log_returns_array.shape[1]):
                    valid_values = log_returns_array[mask_bool[:, i], i]
                    if len(valid_values) > 0:
                        means.append(np.mean(valid_values))
                        stds.append(np.std(valid_values) if len(valid_values) > 1 else 1.0)
                    else:
                        means.append(0.0)
                        stds.append(1.0)

                # Set scaler parameters manually
                self.scaler.mean_ = np.array(means)
                self.scaler.scale_ = np.array(stds)
                self.scaler.var_ = self.scaler.scale_ ** 2
                self.scaler.n_features_in_ = log_returns_array.shape[1]
                self.scaler.n_samples_seen_ = mask_bool.sum(axis=0)
        else:
            # Compute log returns: ln(P_t / P_{t-1})
            log_returns = np.log(df / df.shift(1))
            log_returns = log_returns.dropna()
            log_returns_array = log_returns.values

            if self.scaling == 'expanding':
                self._fit_expanding(log_returns_array)
            else:
                self.scaler.fit(log_returns_array)

    def _fit_expanding(self, log_returns):
        """Compute expanding-window mean and std for each timestep."""
        T, F = log_returns.shape

        if self.min_periods >= T:
            raise ValueError(
                f"min_periods ({self.min_periods}) must be less than "
                f"the number of returns ({T})"
            )

        # Expanding mean and std using cumulative sums (ddof=0 to match StandardScaler)
        cumsum = np.cumsum(log_returns, axis=0)
        cumsum_sq = np.cumsum(log_returns ** 2, axis=0)
        counts = np.arange(1, T + 1, dtype=float)[:, None]  # (T, 1)

        expanding_means = cumsum / counts
        # Population variance (ddof=0): Var = E[X^2] - E[X]^2
        expanding_var = cumsum_sq / counts - expanding_means ** 2
        # Row 0 has count=1: variance is 0, set to 1.0 as fallback
        expanding_var[0] = 1.0
        expanding_var = np.maximum(expanding_var, 0.0)  # Numerical guard
        expanding_stds = np.sqrt(expanding_var)
        expanding_stds = np.maximum(expanding_stds, 1e-8)  # Prevent div-by-zero

        self.expanding_means_ = expanding_means
        self.expanding_stds_ = expanding_stds

        # Set scaler to final (converged) statistics for inverse_transform
        self.scaler.mean_ = expanding_means[-1]
        self.scaler.scale_ = expanding_stds[-1]
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = F
        self.scaler.n_samples_seen_ = T

        self._fitted_expanding = True

    def _fit_expanding_masked(self, log_returns, mask_bool):
        """Compute expanding-window mean and std with masked data."""
        T, F = log_returns.shape

        if self.min_periods >= T:
            raise ValueError(
                f"min_periods ({self.min_periods}) must be less than "
                f"the number of returns ({T})"
            )

        # Use masked values (set invalid to 0 for cumsum)
        masked_returns = log_returns * mask_bool
        cumsum = np.cumsum(masked_returns, axis=0)
        cumsum_sq = np.cumsum((masked_returns ** 2), axis=0)
        counts = np.cumsum(mask_bool.astype(float), axis=0)
        counts_safe = np.maximum(counts, 1.0)

        expanding_means = cumsum / counts_safe
        # Population variance (ddof=0) to match StandardScaler
        expanding_var = cumsum_sq / counts_safe - expanding_means ** 2
        expanding_var = np.maximum(expanding_var, 0.0)
        expanding_stds = np.sqrt(expanding_var)

        # Fallback: where count < 2, use defaults
        no_data = counts < 2
        expanding_means[no_data] = 0.0
        expanding_stds[no_data] = 1.0
        expanding_stds = np.maximum(expanding_stds, 1e-8)

        self.expanding_means_ = expanding_means
        self.expanding_stds_ = expanding_stds

        self.scaler.mean_ = expanding_means[-1]
        self.scaler.scale_ = expanding_stds[-1]
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = F
        self.scaler.n_samples_seen_ = counts[-1]

        self._fitted_expanding = True

    def transform(self, df: pd.DataFrame, mask: pd.DataFrame = None):
        """
        Transform prices to scaled log returns.

        Args:
            df: Price DataFrame
            mask: Optional mask DataFrame (1=valid, 0=missing)

        Returns:
            np.ndarray: Scaled log returns (time_steps-1, features)
            tuple[np.ndarray, np.ndarray]: (data, mask) if mask provided

        Notes:
            When scaling='expanding' and _fitted_expanding=True (training data),
            uses per-timestep causal statistics and drops the first min_periods rows,
            then resets _fitted_expanding to False. Subsequent calls (test/eval data)
            fall back to converged global stats via scaler.transform().
        """
        if self.scaler is None or not hasattr(self.scaler, 'mean_'):
            raise ValueError("Processor not fitted. Call fit() first.")

        if mask is not None:
            log_returns_array, mask_array = self._compute_masked_log_returns(df, mask)

            if self.scaling == 'expanding' and self._fitted_expanding:
                scaled_returns = (log_returns_array - self.expanding_means_) / self.expanding_stds_
                scaled_returns = scaled_returns * mask_array
                self._fitted_expanding = False
                return scaled_returns[self.min_periods:], mask_array[self.min_periods:]
            else:
                scaled_returns = self.scaler.transform(log_returns_array)
                scaled_returns = scaled_returns * mask_array
                return scaled_returns, mask_array
        else:
            log_returns = np.log(df / df.shift(1))
            log_returns = log_returns.dropna()
            log_returns_array = log_returns.values

            if self.scaling == 'expanding' and self._fitted_expanding:
                scaled_returns = (log_returns_array - self.expanding_means_) / self.expanding_stds_
                self._fitted_expanding = False
                return scaled_returns[self.min_periods:]
            else:
                return self.scaler.transform(log_returns_array)

    def validate_variance(self, data: np.ndarray, target_std: float = 1.0, tolerance: float = 0.2):
        """
        Validate that transformed data has expected variance.

        StandardScaler should produce z-scores with std ≈ 1.0. This method checks
        that generated samples have the correct variance before inverse transform.

        Args:
            data: Transformed data (z-scores), shape (Samples, Features) or (Batch, Seq, Features)
            target_std: Expected standard deviation (1.0 for StandardScaler)
            tolerance: Acceptable deviation from target (0.2 = ±20%)

        Returns:
            dict: Validation statistics (mean_std, min_std, max_std, is_valid)

        Raises:
            Warning if variance is outside tolerance
        """
        # Flatten data to compute global statistics
        flat_data = data.reshape(-1, data.shape[-1]) if data.ndim > 2 else data

        # Compute std per feature
        stds = np.std(flat_data, axis=0)
        mean_std = np.mean(stds)
        min_std = np.min(stds)
        max_std = np.max(stds)

        # Check if within tolerance
        is_valid = abs(mean_std - target_std) <= tolerance

        stats = {
            'mean_std': mean_std,
            'min_std': min_std,
            'max_std': max_std,
            'target_std': target_std,
            'tolerance': tolerance,
            'is_valid': is_valid
        }

        if not is_valid:
            print(f"WARNING: Variance validation failed!")
            print(f"  Expected std: {target_std:.3f} ± {tolerance:.3f}")
            print(f"  Actual mean std: {mean_std:.3f} (min={min_std:.3f}, max={max_std:.3f})")
            print(f"  This may indicate issues with diffusion sampling or data preprocessing.")
        else:
            print(f"Variance validation passed: mean_std={mean_std:.3f} (target={target_std:.3f})")

        return stats

    def inverse_transform(self, data: np.ndarray, initial_prices: np.ndarray) -> np.ndarray:
        """
        Args:
            data: (Batch, Seq, Feat) or (Seq, Feat) scaled log-returns.
            initial_prices: (Feat,) or (Batch, Feat)
        Returns:
            prices: (Batch, Seq+1, Feat)
        """
        # Handle 2D input (single sequence)
        if data.ndim == 2:
            data = data[np.newaxis, :, :] # (1, Seq, Feat)
            
        # Handle initial_prices shape
        if initial_prices.ndim == 1:
            initial_prices = initial_prices[np.newaxis, :] # (1, Feat)
            
        batch_size, seq_len, features = data.shape
        
        # 1. Inverse Scale
        # Flatten to (Batch*Seq, Feat) for scaler
        flat_data = data.reshape(-1, features)
        inv_log_returns = self.scaler.inverse_transform(flat_data)
        inv_log_returns = inv_log_returns.reshape(batch_size, seq_len, features)
        
        # 2. Cumulative Sum (Log Prices)
        # cumsum starts from index 0. 
        # log_P_t = log_P_0 + sum(r_1...r_t)
        cum_returns = np.cumsum(inv_log_returns, axis=1)
        
        # Prepend 0s for P_0
        zeros = np.zeros((batch_size, 1, features))
        cum_returns = np.concatenate([zeros, cum_returns], axis=1)
        
        # 3. Exponentiate
        # P_t = P_0 * exp(cum_returns)
        # Broadcast initial_prices (Batch, 1, Feat)
        prices = initial_prices[:, np.newaxis, :] * np.exp(cum_returns)
        
        return prices
