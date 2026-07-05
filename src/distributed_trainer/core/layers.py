import numpy as np

def init_mlp_params(in_dim, hidden_dim, out_dim, seed):
    """Return a dict {'W1','b1','W2','b2'} with He-initialized weights and zero biases."""
    
    # 1. Anchor the randomness to ensure exact reproducibility across workers
    np.random.seed(seed)
    
    # 2. Layer 1 (Input -> Hidden)
    # fan_in is the number of inputs coming into this layer (in_dim)
    fan_in_1 = in_dim
    std_1 = np.sqrt(2.0 / fan_in_1)
    W1 = (np.random.randn(in_dim, hidden_dim) * std_1).astype(np.float64)
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    
    # 3. Layer 2 (Hidden -> Output)
    # fan_in is the number of inputs coming into THIS layer (hidden_dim)
    fan_in_2 = hidden_dim
    std_2 = np.sqrt(2.0 / fan_in_2)
    W2 = (np.random.randn(hidden_dim, out_dim) * std_2).astype(np.float64)
    b2 = np.zeros(out_dim, dtype=np.float64)
    
    # 4. Package everything into a dictionary
    return {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }


def linear_forward(x, w, b):
    """
    Apply a single fully connected (linear/affine) layer to a batch of inputs.
    
    Args:
        x: Input array of shape (N, in_dim).
        w: Weight matrix of shape (in_dim, out_dim).
        b: Bias vector of shape (out_dim,).
        
    Returns:
        y: Output array of shape (N, out_dim) after applying y = x @ w + b.
    """
    # The @ operator handles matrix multiplication, and NumPy automatically 
    # broadcasts the bias vector across the batch dimension.
    y = (x @ w) + b
    
    return y


def relu_forward(x):
    """
    Apply the ReLU (Rectified Linear Unit) activation elementwise.
    
    Args:
        x: Input array of any shape (usually the output of a linear layer).
        
    Returns:
        A new array of the same shape where all negative values are clipped to 0.
    """
    # np.maximum compares two arrays (or a scalar and an array) element-by-element,
    # returning the maximum value. This creates a brand new array, preserving 
    # the original 'x' which we will absolutely need later for the backward pass.
    return np.maximum(0, x)


def mlp_forward(x, params):
    """
    Run a two-layer MLP forward pass.
    
    Args:
        x: Input array of shape (N, in_dim).
        params: Dictionary containing 'W1', 'b1', 'W2', 'b2'.
        
    Returns:
        y_pred: The final prediction array of shape (N, out_dim).
        cache: Dictionary containing 'x', 'z1', 'a1', 'z2' for the backward pass.
    """
    # 1. First Linear Layer: Transform inputs to the hidden dimension space
    z1 = linear_forward(x, params['W1'], params['b1'])
    
    # 2. ReLU Activation: Apply the non-linearity
    a1 = relu_forward(z1)
    
    # 3. Second Linear Layer: Transform hidden representation to final output
    z2 = linear_forward(a1, params['W2'], params['b2'])
    
    # 4. Cache intermediate values required for backpropagation
    cache = {
        'x': x,
        'z1': z1,
        'a1': a1,
        'z2': z2
    }
    
    return z2, cache