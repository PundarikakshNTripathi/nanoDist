import numpy as np

def make_synthetic_regression_batch(batch_size, in_dim, out_dim, seed):
    """Return (x, y) where x is (batch_size, in_dim) and y is (batch_size, out_dim) float64."""
    
    # 1. Anchor the randomness. This ensures our "random" numbers are identical every run.
    np.random.seed(seed)
    
    # 2. Sample inputs (x) from a standard normal distribution (mean 0, variance 1)
    x = np.random.randn(batch_size, in_dim).astype(np.float64)
    
    # 3. Build the hidden "teacher" model (W_true)
    W_true = np.random.randn(in_dim, out_dim).astype(np.float64)
    
    # 4. Create a small amount of Gaussian noise
    # We multiply by 0.01 to keep the noise subtle but present
    noise = (np.random.randn(batch_size, out_dim) * 0.01).astype(np.float64)
    
    # 5. Generate the targets (y) via matrix multiplication (x @ W_true) and add the noise
    y = (x @ W_true) + noise
    
    return x, y