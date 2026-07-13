import numpy as np

def fit_distributions(cal_features, bg_features):
    """
    Fits Gaussian distributions to in-domain (calibration) and background features.
    
    Args:
        cal_features (np.ndarray): In-domain features (M, D).
        bg_features (np.ndarray): Background features (K, D).
        
    Returns:
        tuple: (mu_in, Sigma_in_inv, mu_bg, Sigma_bg_inv)
    """
    # 1. In-domain
    mu_in = np.mean(cal_features, axis=0)
    Sigma_in = np.cov(cal_features.T, ddof=1)
    
    # 2. Background
    mu_bg = np.mean(bg_features, axis=0)
    Sigma_bg = np.cov(bg_features.T, ddof=1)
    
    # Ensure 2D for inversion (if features dim is 1)
    if Sigma_in.ndim == 0:
        Sigma_in = Sigma_in.reshape(1, 1)
    if Sigma_bg.ndim == 0:
        Sigma_bg = Sigma_bg.reshape(1, 1)
    
    # 3. Regularization & Inverse
    REGULARIZATION_LAMBDA = 1e-6
    I = np.eye(Sigma_in.shape[0])
    
    Sigma_in_reg = Sigma_in + REGULARIZATION_LAMBDA * I
    Sigma_bg_reg = Sigma_bg + REGULARIZATION_LAMBDA * I
    
    try:
        Sigma_in_inv = np.linalg.inv(Sigma_in_reg)
    except np.linalg.LinAlgError:
        Sigma_in_inv = np.linalg.pinv(Sigma_in_reg)
        
    try:
        Sigma_bg_inv = np.linalg.inv(Sigma_bg_reg)
    except np.linalg.LinAlgError:
        Sigma_bg_inv = np.linalg.pinv(Sigma_bg_reg)
        
    return mu_in, Sigma_in_inv, mu_bg, Sigma_bg_inv


def compute_rmd(features, mu_in, Sigma_in_inv, mu_bg, Sigma_bg_inv):
    """
    Computes Relative Mahalanobis Distance (RMD).
    
    Args:
        features (np.ndarray): Feature vector (D,).
        mu_in, Sigma_in_inv: Parameters for in-domain distribution.
        mu_bg, Sigma_bg_inv: Parameters for background distribution.
        
    Returns:
        float: RMD score.
    """
    # MD to In-domain
    diff_in = features - mu_in
    md_in = diff_in.T @ Sigma_in_inv @ diff_in
    
    # MD to Background
    diff_bg = features - mu_bg
    md_bg = diff_bg.T @ Sigma_bg_inv @ diff_bg
    
    # RMD
    return md_in - md_bg


def calculate_volatility(seq):
    """
    Calculates the standard deviation of the step-wise differences of a sequence.
    
    Args:
        seq (array-like): input sequence.
        
    Returns:
        float: Volatility metric.
    """
    # Calculate the standard deviation of changes between adjacent time steps
    # Measures the "jitter" of alpha or retention
    return np.std(np.diff(seq))
