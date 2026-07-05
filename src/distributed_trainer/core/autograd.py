import numpy as np

def linear_backward(d_out, x, w):
    """
    Backpropagate through a single fully connected (linear/affine) layer.
    
    Args:
        d_out: Upstream gradient of shape (N, out_dim).
        x: Cached input array from the forward pass, shape (N, in_dim).
        w: Cached weight matrix from the forward pass, shape (in_dim, out_dim).
        
    Returns:
        dx: Gradient with respect to x, shape (N, in_dim).
        dw: Gradient with respect to w, shape (in_dim, out_dim).
        db: Gradient with respect to b, shape (out_dim,).
    """
    # 1. Gradient for the inputs (dx)
    # Project the errors backward through the transpose of the weights.
    dx = d_out @ w.T
    
    # 2. Gradient for the weights (dw)
    # Multiply the transposed inputs by the errors to calculate weight updates.
    dw = x.T @ d_out
    
    # 3. Gradient for the biases (db)
    # Since the bias was broadcast across the batch (N) in the forward pass,
    # we sum the errors across the batch dimension (axis=0) in the backward pass.
    db = np.sum(d_out, axis=0)
    
    return dx, dw, db

def relu_backward(d_out, z):
    """
    Backpropagate a gradient through a ReLU activation.
    
    Args:
        d_out: Upstream gradient of any shape.
        z: Cached pre-activation tensor from the forward pass, same shape as d_out.
        
    Returns:
        dz: Gradient with respect to z, same shape as z.
    """
    # (z > 0) creates a boolean mask of True (1) and False (0).
    # Multiplying d_out by this mask effectively zeroes out the gradients 
    # where the pre-activation was less than or equal to zero.
    dz = d_out * (z > 0)
    
    return dz

def first_linear_backward(d_z1, x, w1):
    """
    Backpropagate through the first linear layer of the MLP.
    
    Args:
        d_z1: Upstream gradient from the ReLU layer, shape (N, hidden_dim).
        x: Raw input data from the forward pass, shape (N, in_dim).
        w1: First layer weight matrix, shape (in_dim, hidden_dim).
        
    Returns:
        dx: Gradient with respect to the input x, shape (N, in_dim).
        dW1: Gradient with respect to the weights W1, shape (in_dim, hidden_dim).
        db1: Gradient with respect to the bias b1, shape (hidden_dim,).
    """
    # 1. Gradient with respect to the inputs
    dx = d_z1 @ w1.T
    
    # 2. Gradient with respect to the weights
    dW1 = x.T @ d_z1
    
    # 3. Gradient with respect to the biases (summing over the batch dimension)
    db1 = np.sum(d_z1, axis=0)
    
    return dx, dW1, db1

def mlp_backward(dy_pred, cache, params):
    """
    Run the full MLP backward pass.
    
    Args:
        dy_pred: Upstream gradient of the loss, shape (N, out_dim).
        cache: Dictionary containing 'x', 'z1', 'a1', 'z2' from the forward pass.
        params: Dictionary containing 'W1', 'b1', 'W2', 'b2'.
        
    Returns:
        A dictionary of parameter gradients with keys 'W1', 'b1', 'W2', 'b2'.
    """
    # 1. Backprop through Layer 2 (Output Layer)
    # The input to this layer during the forward pass was 'a1' (the post-ReLU activations)
    da1, dW2, db2 = linear_backward(dy_pred, cache['a1'], params['W2'])
    
    # 2. Backprop through the ReLU Activation
    # The input to ReLU was 'z1' (the raw pre-activations from Layer 1)
    dz1 = relu_backward(da1, cache['z1'])
    
    # 3. Backprop through Layer 1 (Hidden Layer)
    # The input to this layer was 'x' (the raw dataset features)
    dx, dW1, db1 = first_linear_backward(dz1, cache['x'], params['W1'])
    
    # 4. Bundle the gradients into a dictionary matching the params structure
    grads = {
        'W1': dW1,
        'b1': db1,
        'W2': dW2,
        'b2': db2
    }
    
    return grads