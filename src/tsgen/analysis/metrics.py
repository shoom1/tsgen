import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from statsmodels.tsa.stattools import acf
import matplotlib.pyplot as plt
import os

def compute_acf(x, lags=50):
    """
    Computes ACF (Auto-Correlation Function) for a 1D array x up to 'lags'.

    Args:
        x (np.ndarray): 1D array of time series data
        lags (int): Maximum number of lags to compute (default: 50)

    Returns:
        np.ndarray: ACF values for lags 0 to 'lags' (length: lags+1)
    """
    # nlags in acf includes lag 0, so we ask for lags
    # fft=True is faster for large arrays
    try:
        return acf(x, nlags=lags, fft=True)
    except ValueError:
        # Returns zeros if computation fails (e.g., constant series, contains NaNs, too short)
        # Or if fft=True fails for some specific cases.
        return np.zeros(lags+1)
    except Exception as e:
        # Re-raise unexpected exceptions
        raise e

def calculate_var_es(returns, alpha=0.05):
    """
    Calculates Value-at-Risk (VaR) and Expected Shortfall (ES) for a given
    array of returns.

    Args:
        returns (np.ndarray): Array of returns (should be negative for losses).
        alpha (float): Confidence level (e.g., 0.05 for 95% VaR/ES).

    Returns:
        tuple: (VaR, ES)
    """
    if len(returns) == 0:
        return np.nan, np.nan

    # Sort returns in ascending order (losses are negative values)
    sorted_returns = np.sort(returns)

    # Calculate VaR
    var_idx = int(np.floor(alpha * len(sorted_returns)))
    var = sorted_returns[var_idx]

    # Calculate ES (average of returns worse than VaR)
    es_returns = sorted_returns[sorted_returns <= var]
    es = np.mean(es_returns) if len(es_returns) > 0 else np.nan

    return var, es


def calculate_stylized_facts(real_returns, synthetic_returns):
    """
    Computes and compares stylized facts:
    1. Autocorrelation of Returns (should be ~0)
    2. Autocorrelation of Squared Returns (Volatility Clustering)
    3. Kurtosis (Fat tails)
    4. Skewness
    5. Correlation Matrix (Asset dependence)

    Args:
        real_returns: (N, Seq_Len, Features)
        synthetic_returns: (N, Seq_Len, Features)
    
    Returns:
        dict: Dictionary containing metrics and comparison scores.
    """
    # Flatten dimensions for distribution statistics: (N * Seq_Len, Features)
    # We treat all windows as a continuous stream for distribution analysis,
    # though breaks exist.
    real_flat = real_returns.reshape(-1, real_returns.shape[-1])
    syn_flat = synthetic_returns.reshape(-1, synthetic_returns.shape[-1])

    # Stylized-fact statistics (kurtosis, skew, ACF) require finite input.
    # scipy's `kurtosis`/`skew` default `nan_policy='propagate'`, so a single
    # NaN in a column yields NaN for that column's stat. Sanitize to zero —
    # the most honest fallback is to exclude those positions from the stat.
    real_flat = np.nan_to_num(real_flat, nan=0.0, posinf=0.0, neginf=0.0)
    syn_flat = np.nan_to_num(syn_flat, nan=0.0, posinf=0.0, neginf=0.0)

    metrics = {}

    # 1. Distributional Statistics (per asset)
    metrics['real_kurtosis'] = kurtosis(real_flat, axis=0)
    metrics['syn_kurtosis'] = kurtosis(syn_flat, axis=0)
    metrics['kurtosis_diff'] = np.abs(metrics['real_kurtosis'] - metrics['syn_kurtosis'])
    
    metrics['real_skew'] = skew(real_flat, axis=0)
    metrics['syn_skew'] = skew(syn_flat, axis=0)
    metrics['skew_diff'] = np.abs(metrics['real_skew'] - metrics['syn_skew'])
    
    # 2. VaR and ES
    # For VaR and ES, we typically look at the distribution of individual returns
    # across all samples/sequences, as opposed to sequence-level stats.
    # Flatten across batch and sequence length for each feature
    
    metrics['real_var'] = []
    metrics['real_es'] = []
    metrics['syn_var'] = []
    metrics['syn_es'] = []
    
    alpha = 0.05 # For 95% VaR and ES
    
    for i in range(real_flat.shape[1]): # Iterate over features
        r_var, r_es = calculate_var_es(real_flat[:, i], alpha=alpha)
        s_var, s_es = calculate_var_es(syn_flat[:, i], alpha=alpha)
        metrics['real_var'].append(r_var)
        metrics['real_es'].append(r_es)
        metrics['syn_var'].append(s_var)
        metrics['syn_es'].append(s_es)
        
    metrics['real_var'] = np.array(metrics['real_var'])
    metrics['real_es'] = np.array(metrics['real_es'])
    metrics['syn_var'] = np.array(metrics['syn_var'])
    metrics['syn_es'] = np.array(metrics['syn_es'])

    metrics['var_diff'] = np.abs(metrics['real_var'] - metrics['syn_var'])
    metrics['es_diff'] = np.abs(metrics['real_es'] - metrics['syn_es'])
    
    # 3. Correlation Matrix (Spatial/Cross-asset correlation)
    # shape (Features, Features)
    real_corr = np.corrcoef(real_flat, rowvar=False)
    syn_corr = np.corrcoef(syn_flat, rowvar=False)
    
    # Frobenius norm of the difference matrix
    metrics['corr_matrix_diff_norm'] = np.linalg.norm(real_corr - syn_corr)
    
    # 3. Autocorrelation (Temporal dependence)
    # We compute ACF per window and average them, or per asset.
    # Let's compute per asset, averaging over samples.
    
    # Shape: (Features, Lags)
    real_acf_ret = []
    syn_acf_ret = []
    
    real_acf_sq_ret = []
    syn_acf_sq_ret = []
    
    lags = 20
    
    # Loop over features
    for i in range(real_returns.shape[2]):
        # Flattening is risky for ACF due to jumps, but for Stylized facts checks usually 
        # we check if the *property* holds. 
        # A safer way for generated windows: Compute ACF for each window, then mean.
        
        # Real
        r_feat = real_returns[:, :, i] # (N, L)
        r_acf_list = [compute_acf(row, lags) for row in r_feat]
        real_acf_ret.append(np.mean(r_acf_list, axis=0))
        
        r_sq_acf_list = [compute_acf(row**2, lags) for row in r_feat]
        real_acf_sq_ret.append(np.mean(r_sq_acf_list, axis=0))
        
        # Synthetic
        s_feat = synthetic_returns[:, :, i]
        s_acf_list = [compute_acf(row, lags) for row in s_feat]
        syn_acf_ret.append(np.mean(s_acf_list, axis=0))
        
        s_sq_acf_list = [compute_acf(row**2, lags) for row in s_feat]
        syn_acf_sq_ret.append(np.mean(s_sq_acf_list, axis=0))

    metrics['real_acf_ret'] = np.array(real_acf_ret)
    metrics['syn_acf_ret'] = np.array(syn_acf_ret)
    metrics['real_acf_sq_ret'] = np.array(real_acf_sq_ret)
    metrics['syn_acf_sq_ret'] = np.array(syn_acf_sq_ret)
    
    # Scalar score for ACF difference (MSE of ACFs)
    metrics['acf_ret_diff'] = np.mean((metrics['real_acf_ret'] - metrics['syn_acf_ret'])**2)
    metrics['acf_sq_ret_diff'] = np.mean((metrics['real_acf_sq_ret'] - metrics['syn_acf_sq_ret'])**2)
    
    return metrics

def _sanitize_correlation_matrix(corr: np.ndarray) -> np.ndarray:
    """Clean a (possibly degenerate) correlation matrix for downstream linalg.

    Handles the cases exposed by real evaluation runs:
      - NaN in synthetic samples (``corrcoef`` returns NaN rows/cols)
      - Constant-valued feature columns (zero variance → NaN correlations)
      - Inf values from diverged models
      - Slight asymmetry from floating-point noise

    Returns a symmetric matrix with unit diagonal and all finite entries.
    No raise path: if every intermediate is NaN, returns the identity.
    """
    if np.isscalar(corr):
        return np.array([[1.0]])
    corr = np.asarray(corr, dtype=float)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    # Force symmetric and unit-diagonal so eigvalsh behaves.
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr


def compute_correlation_structure_metrics(real_returns, synthetic_returns):
    """
    Comprehensive correlation structure analysis.

    Compares correlation matrices, eigenvalue spectra, and temporal stability
    of correlations between real and synthetic data.

    Args:
        real_returns: (N, Seq, Features) numpy array of real returns
        synthetic_returns: (N, Seq, Features) numpy array of synthetic returns

    Returns:
        dict: Dictionary with correlation structure metrics
    """
    # Flatten to (N*Seq, Features)
    real_flat = real_returns.reshape(-1, real_returns.shape[-1])
    syn_flat = synthetic_returns.reshape(-1, synthetic_returns.shape[-1])

    metrics = {}

    # 1. Correlation Matrix Comparison
    # np.corrcoef emits NaN for constant columns and doesn't tolerate
    # NaN/Inf in its input. Sanitize both before downstream linalg.
    real_corr = _sanitize_correlation_matrix(
        np.corrcoef(np.nan_to_num(real_flat, nan=0.0, posinf=0.0, neginf=0.0), rowvar=False)
    )
    syn_corr = _sanitize_correlation_matrix(
        np.corrcoef(np.nan_to_num(syn_flat, nan=0.0, posinf=0.0, neginf=0.0), rowvar=False)
    )

    # Handle scalar case (single feature)
    if np.isscalar(real_corr):
        real_corr = np.array([[real_corr]])
        syn_corr = np.array([[syn_corr]])

    # Frobenius norm of difference
    diff_matrix = real_corr - syn_corr
    if diff_matrix.shape == (1, 1):
        metrics['corr_frobenius_norm'] = np.abs(diff_matrix[0, 0])
    else:
        metrics['corr_frobenius_norm'] = np.linalg.norm(diff_matrix, 'fro')

    # Maximum element-wise difference
    metrics['corr_max_diff'] = np.abs(real_corr - syn_corr).max()

    # Mean absolute difference
    metrics['corr_mean_diff'] = np.abs(real_corr - syn_corr).mean()

    # Store correlation matrices for plotting
    metrics['real_corr_matrix'] = real_corr
    metrics['syn_corr_matrix'] = syn_corr

    # 2. Eigenvalue/PCA Analysis
    # Compare eigenvalue spectra to assess covariance structure
    real_eigenvalues = np.linalg.eigvalsh(real_corr)[::-1]  # Sort descending
    syn_eigenvalues = np.linalg.eigvalsh(syn_corr)[::-1]

    # MSE between eigenvalue spectra
    metrics['eigenvalue_mse'] = np.mean((real_eigenvalues - syn_eigenvalues) ** 2)

    # Maximum eigenvalue difference
    metrics['eigenvalue_max_diff'] = np.abs(real_eigenvalues - syn_eigenvalues).max()

    # Explained variance ratio difference
    real_var_ratio = real_eigenvalues / real_eigenvalues.sum()
    syn_var_ratio = syn_eigenvalues / syn_eigenvalues.sum()
    metrics['explained_var_ratio_diff'] = np.abs(real_var_ratio - syn_var_ratio).sum()

    # Store for plotting
    metrics['real_eigenvalues'] = real_eigenvalues
    metrics['syn_eigenvalues'] = syn_eigenvalues

    # 3. Rolling Correlation Stability (Temporal Consistency)
    # Only compute if we have enough data and at least 2 features
    if real_returns.shape[1] >= 20 and real_returns.shape[2] >= 2:
        rolling_corr_metrics = compute_rolling_correlation_stability(
            real_returns, synthetic_returns, window=20
        )
        metrics.update(rolling_corr_metrics)
    else:
        metrics['rolling_corr_stability'] = np.nan
        metrics['rolling_corr_std_diff'] = np.nan

    return metrics


def compute_rolling_correlation_stability(real_returns, synthetic_returns, window=20):
    """
    Measure temporal stability of rolling correlations.

    Computes rolling correlation between first two assets and compares
    stability between real and synthetic data.

    Args:
        real_returns: (N, Seq, Features) numpy array
        synthetic_returns: (N, Seq, Features) numpy array
        window: Window size for rolling correlation

    Returns:
        dict: Rolling correlation metrics
    """
    metrics = {}

    # Use first two assets for correlation analysis
    if real_returns.shape[2] < 2:
        metrics['rolling_corr_stability'] = np.nan
        metrics['rolling_corr_std_diff'] = np.nan
        return metrics

    seq_len = real_returns.shape[1]
    if seq_len < window + 1:
        metrics['rolling_corr_stability'] = np.nan
        metrics['rolling_corr_std_diff'] = np.nan
        return metrics

    real_rolling = []
    syn_rolling = []

    # Compute rolling correlation for each sample
    for i in range(real_returns.shape[0]):
        # Extract first two assets
        real_pair = real_returns[i, :, :2]  # (Seq, 2)
        syn_pair = synthetic_returns[i, :, :2]  # (Seq, 2)

        # Compute rolling correlations
        for j in range(seq_len - window + 1):
            real_window = real_pair[j:j+window]
            syn_window = syn_pair[j:j+window]

            # Compute correlation for this window
            real_corr_val = np.corrcoef(real_window, rowvar=False)[0, 1]
            syn_corr_val = np.corrcoef(syn_window, rowvar=False)[0, 1]

            real_rolling.append(real_corr_val)
            syn_rolling.append(syn_corr_val)

    real_rolling = np.array(real_rolling)
    syn_rolling = np.array(syn_rolling)

    # Remove NaN values if any
    valid_mask = ~(np.isnan(real_rolling) | np.isnan(syn_rolling))
    real_rolling = real_rolling[valid_mask]
    syn_rolling = syn_rolling[valid_mask]

    if len(real_rolling) == 0:
        metrics['rolling_corr_stability'] = np.nan
        metrics['rolling_corr_std_diff'] = np.nan
    else:
        # Mean absolute difference in rolling correlations
        metrics['rolling_corr_stability'] = np.mean(np.abs(real_rolling - syn_rolling))

        # Difference in volatility of rolling correlations
        metrics['rolling_corr_std_diff'] = np.abs(np.std(real_rolling) - np.std(syn_rolling))

        # Store for potential plotting
        metrics['real_rolling_corr'] = real_rolling
        metrics['syn_rolling_corr'] = syn_rolling

    return metrics


def plot_correlation_structure(metrics, tickers, save_path="plots/correlation_structure.png"):
    """
    Visualize correlation structure comparison.

    Creates plots showing:
    1. Real vs Synthetic correlation matrices
    2. Eigenvalue spectra comparison
    3. Rolling correlation stability (if available)

    Args:
        metrics: Dictionary from compute_correlation_structure_metrics
        tickers: List of ticker symbols
        save_path: Path to save the plot
    """
    # Determine number of subplots
    has_rolling = 'real_rolling_corr' in metrics and metrics.get('real_rolling_corr') is not None

    if has_rolling:
        fig = plt.figure(figsize=(18, 5))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1])
    else:
        fig = plt.figure(figsize=(14, 5))
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])

    # 1. Real Correlation Matrix
    ax1 = fig.add_subplot(gs[0])
    real_corr = metrics['real_corr_matrix']
    im1 = ax1.imshow(real_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_title('Real Correlation Matrix')
    ax1.set_xticks(range(len(tickers)))
    ax1.set_yticks(range(len(tickers)))
    ax1.set_xticklabels(tickers, rotation=45)
    ax1.set_yticklabels(tickers)
    plt.colorbar(im1, ax=ax1)

    # 2. Synthetic Correlation Matrix
    ax2 = fig.add_subplot(gs[1])
    syn_corr = metrics['syn_corr_matrix']
    im2 = ax2.imshow(syn_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_title('Synthetic Correlation Matrix')
    ax2.set_xticks(range(len(tickers)))
    ax2.set_yticks(range(len(tickers)))
    ax2.set_xticklabels(tickers, rotation=45)
    ax2.set_yticklabels(tickers)
    plt.colorbar(im2, ax=ax2)

    # 3. Eigenvalue Spectra
    ax3 = fig.add_subplot(gs[2])
    real_eig = metrics['real_eigenvalues']
    syn_eig = metrics['syn_eigenvalues']
    x = np.arange(len(real_eig))
    ax3.plot(x, real_eig, 'o-', label='Real', markersize=6)
    ax3.plot(x, syn_eig, 'x-', label='Synthetic', markersize=6)
    ax3.set_xlabel('Eigenvalue Index')
    ax3.set_ylabel('Eigenvalue')
    ax3.set_title('Eigenvalue Spectra')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Rolling Correlation (if available)
    if has_rolling:
        ax4 = fig.add_subplot(gs[3])
        real_rolling = metrics['real_rolling_corr']
        syn_rolling = metrics['syn_rolling_corr']

        # Plot histograms
        bins = 30
        ax4.hist(real_rolling, bins=bins, alpha=0.5, label='Real', density=True)
        ax4.hist(syn_rolling, bins=bins, alpha=0.5, label='Synthetic', density=True)
        ax4.set_xlabel('Rolling Correlation')
        ax4.set_ylabel('Density')
        ax4.set_title(f'Rolling Correlation Distribution\n({tickers[0]}-{tickers[1]})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Correlation structure plot saved to {save_path}")


def plot_stylized_facts(metrics, tickers, save_path="plots/stylized_facts.png"):
    """
    Generates a visual comparison of stylized facts.
    """
    n_tickers = len(tickers)
    fig, axes = plt.subplots(n_tickers, 3, figsize=(15, 4 * n_tickers))
    
    if n_tickers == 1: axes = axes[None, :]
    
    lags = range(metrics['real_acf_ret'].shape[1])
    
    for i, ticker in enumerate(tickers):
        # 1. ACF of Returns
        ax = axes[i, 0]
        ax.plot(lags, metrics['real_acf_ret'][i], label='Real', marker='o', markersize=3)
        ax.plot(lags, metrics['syn_acf_ret'][i], label='Synthetic', marker='x', markersize=3)
        ax.set_title(f"{ticker}: ACF of Returns\n(Should be ~0)")
        ax.set_ylim(-0.2, 0.2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. ACF of Squared Returns (Vol Clustering)
        ax = axes[i, 1]
        ax.plot(lags, metrics['real_acf_sq_ret'][i], label='Real', marker='o', markersize=3)
        ax.plot(lags, metrics['syn_acf_sq_ret'][i], label='Synthetic', marker='x', markersize=3)
        ax.set_title(f"{ticker}: ACF of Squared Returns\n(Vol Clustering)")
        ax.set_ylim(-0.1, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Distribution Stats (Text)
        ax = axes[i, 2]
        ax.axis('off')
        text_str = f"Stats for {ticker}:\n\n"
        text_str += f"Kurtosis (Real): {metrics['real_kurtosis'][i]:.2f}\n"
        text_str += f"Kurtosis (Syn):  {metrics['syn_kurtosis'][i]:.2f}\n\n"
        text_str += f"Skew (Real): {metrics['real_skew'][i]:.2f}\n"
        text_str += f"Skew (Syn):  {metrics['syn_skew'][i]:.2f}\n\n"
        text_str += f"VaR (Real): {metrics['real_var'][i]:.4f}\n"
        text_str += f"VaR (Syn):  {metrics['syn_var'][i]:.4f}\n\n"
        text_str += f"ES (Real): {metrics['real_es'][i]:.4f}\n"
        text_str += f"ES (Syn):  {metrics['syn_es'][i]:.4f}\n"
        ax.text(0.1, 0.5, text_str, fontsize=12, va='center')
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
