import numpy as np

# 1. Define Pinball Loss
def pinball_loss(u, alpha):
    """
    u: beta_t - alpha_t
    alpha: target alpha (e.g., 0.1)
    """
    # Corresponds to R code: alpha*u - vecZeroMin(u)
    return alpha * u - np.minimum(u, 0)

class DtACI:
    def __init__(self, gammas, alpha, eta=None, sigma=None):
        self.gammas = np.array(gammas)
        self.K = len(gammas)
        self.alpha = alpha  # Target misalignment rate, e.g., 0.1
        
        # Initialize expert states
        self.expert_alphas = np.full(self.K, alpha)
        self.weights = np.ones(self.K)
        
        # Parameter defaults (Reference to paper/code)
        self.sigma = sigma if sigma is not None else 0.001
        self.eta = eta if eta is not None else 2.0

    def predict(self):
        """Returns the predicted alpha value for the current time step"""
        # Normalize weights
        w_sum = np.sum(self.weights)
        if w_sum <= 0:
            self.current_probs = np.ones(self.K) / self.K
        else:
            self.current_probs = self.weights / w_sum
            
        # Weighted average
        self.curr_avg_alpha = np.sum(self.current_probs * self.expert_alphas)
        # Clip boundaries to prevent numerical issues
        self.curr_avg_alpha = np.clip(self.curr_avg_alpha, 0.001, 0.999)
        return self.curr_avg_alpha

    def update(self, beta_t):
        """
        Update the model based on the current true value beta_t (p-value).
        beta_t: Quantile of the true value y in the predicted distribution.
                If beta_t < alpha_t, it means y fell outside the 1-alpha_t interval (not covered).
        """
        # 1. Calculate Pinball Loss (used for updating weights)
        u = beta_t - self.expert_alphas
        expert_losses = self.alpha * u - np.minimum(0, u)
        
        # 2. Update weights (using continuous Loss)
        self.weights = self.weights * np.exp(-self.eta * expert_losses)
        
        # 3. Restart mechanism
        W_total = np.sum(self.weights)
        if W_total < 1e-50: # Numerical stability protection
            self.weights = np.ones(self.K)
            W_total = self.K
            
        uniform_share = (self.sigma * W_total) / self.K
        self.weights = (1 - self.sigma) * self.weights + uniform_share
        
        # 4. Update expert parameters (using 0/1 gradient)
        errs = (self.expert_alphas > beta_t).astype(float)
        
        update_step = self.gammas * (self.alpha - errs)
        self.expert_alphas = self.expert_alphas + update_step
        
        # Clip boundaries
        self.expert_alphas = np.clip(self.expert_alphas, 0.001, 0.999)

    def get_weighted_gamma(self):
        """Returns the weighted average gamma for the current time step"""
        if not hasattr(self, 'current_probs'):
             # fallback if predict hasn't been called yet
             return np.mean(self.gammas)
        return np.sum(self.current_probs * self.gammas)

    def get_weighted_alpha(self):
        """Returns the weighted average alpha for the current time step"""
        if not hasattr(self, 'current_probs'):
             # fallback if predict hasn't been called yet
             return np.mean(self.expert_alphas)
        return np.sum(self.current_probs * self.expert_alphas)

# ==========================================
# Example: How to use this class
# ==========================================

if __name__ == "__main__":
    # 1. Simulate data: Sine wave + noise, distribution shift happens in the middle
    np.random.seed(42)
    T = 500
    X = np.linspace(0, 10, T)
    # First 250 steps are standard normal, last 250 steps variance suddenly increases (Jump Shift)
    noise_std = np.concatenate([np.ones(250), np.ones(250) * 3])
    Y = np.sin(X) + np.random.normal(0, noise_std, T)
    
    # 2. Define prediction interval function
    # This is a helper function to visualize the interval given an alpha
    from scipy.stats import norm
    
    def check_coverage(alpha_val, y_true):
        t_idx = int(X[np.where(Y == y_true)[0][0]]) # Simplified index retrieval
        y_pred = np.sin(X[t_idx]) if t_idx < T else 0
        
        scale = norm.ppf(1 - alpha_val / 2)
        width = scale  
        
        lower = y_pred - width
        upper = y_pred + width
        
        return lower <= y_true <= upper

    # 3. Run DtACI
    gammas = [0.001, 0.002, 0.004, 0.008, 0.0160, 0.032, 0.064, 0.128] # Different step size experts
    target_alpha = 0.1 # 90% coverage
    
    model = DtACI(gammas, target_alpha, eta=0.5, sigma=0.01)
    
    adaptive_alphas = []
    
    print("Starting online prediction...")
    for t, y in enumerate(Y):
        # 1. Get current suggested alpha
        curr_alpha = model.predict()
        adaptive_alphas.append(curr_alpha)
        
        # 2. Define current check function (closure)
        
        # Simulate current prediction mean
        pred_mean = np.sin(X[t])
        
        # Define a simple function: given alpha, return if it covers y
        def current_step_check(a, y_val):
            width = norm.ppf(1 - a / 2) 
            return (pred_mean - width) <= y_val <= (pred_mean + width)

        # 3. Update model
        model.update(y)

    print("Done. First 10 dynamic alpha values:", np.round(adaptive_alphas[:10], 3))
    print("Last 10 dynamic alpha values:", np.round(adaptive_alphas[-10:], 3))