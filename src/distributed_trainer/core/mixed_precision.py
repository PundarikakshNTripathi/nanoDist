import numpy as np

from distributed_trainer.core.layers import mlp_forward
from distributed_trainer.core.autograd import mlp_backward
from distributed_trainer.evaluation.metrics import mse_loss_and_grad


def cast_to_half_precision(values):
    """
    Convert a dictionary of numpy arrays to float16 (half precision).
    
    Args:
        values: Dictionary containing numpy arrays (usually float32 or float64).
        
    Returns:
        A new dictionary with the exact same keys, but all arrays cast to float16.
    """
    # astype() safely creates a brand new array in memory with the requested data type,
    # ensuring we do not accidentally overwrite our original float32/float64 master copies.
    return {k: v.astype(np.float16) for k, v in values.items()}

def make_master_params(params):
    """
    Build a float32 master copy of every parameter tensor used by a mixed precision trainer.
    
    Args:
        params: Dictionary mapping names to NumPy arrays (possibly float16 or float64).
        
    Returns:
        A new dictionary mapping the same keys to independent float32 copies.
    """
    # astype(np.float32) ensures we return a fresh, independent copy in memory,
    # rather than just pointing to the original arrays.
    return {k: v.astype(np.float32) for k, v in params.items()}

def scale_loss(loss, dy_pred, scale):
    """
    Multiply the scalar loss and the upstream gradient dy_pred by a fixed loss scale.
    
    Args:
        loss: The original scalar loss float.
        dy_pred: The upstream gradient array of shape (N, out_dim).
        scale: The scalar multiplier (e.g., 1024.0 or 65536.0).
        
    Returns:
        scaled_loss: The scalar loss multiplied by scale.
        scaled_dy_pred: The gradient array multiplied by scale.
    """
    # Simply multiply both values by the scale factor
    scaled_loss = loss * scale
    scaled_dy_pred = dy_pred * scale
    
    return scaled_loss, scaled_dy_pred

def unscale_gradients(grads, scale):
    """
    Divide every gradient tensor by the scale and return a new float32 dictionary.
    
    Args:
        grads: Dictionary of scaled gradients (usually in float16).
        scale: The scalar multiplier used earlier (e.g., 1024.0).
        
    Returns:
        A new dictionary where the gradients are restored to their true 
        magnitudes and stored in safe float32.
    """
    # We cast to float32 FIRST (or during the operation) to ensure the division 
    # happens in a higher precision space, preventing any accidental underflow 
    # during the math itself.
    return {k: (v.astype(np.float32) / scale) for k, v in grads.items()}

def has_non_finite_gradients(grads):
    """
    Scan a dictionary of gradient arrays and return True if any gradient 
    contains a NaN (Not a Number) or Inf (Infinity) value.
    
    Args:
        grads: Dictionary mapping parameter names to gradient arrays.
        
    Returns:
        True if any array contains a non-finite value, otherwise False.
    """
    # np.isfinite returns a boolean array where True means the number is valid.
    # .all() checks if EVERY number in the array is valid.
    # We use Python's any() generator to check if ANY of the arrays fail this test.
    # This automatically short-circuits (stops checking) the moment it finds a bad array.
    return any(not np.isfinite(v).all() for v in grads.values())

def mixed_precision_step(x, y, master_params, scale, lr):
    """
    Run a single training step using mixed precision (fp16 math, fp32 updates).
    """
    # 1. Create fp16 shadow copies
    fp16_params = cast_to_half_precision(master_params)
    x_fp16 = x.astype(np.float16)
    y_fp16 = y.astype(np.float16)
    
    # 2. Forward pass in fp16
    y_pred, cache = mlp_forward(x_fp16, fp16_params)
    loss, dy_pred = mse_loss_and_grad(y_pred, y_fp16)
    
    # 3. Scale and backward pass in fp16
    scaled_loss, scaled_dy_pred = scale_loss(loss, dy_pred, scale)
    scaled_grads = mlp_backward(scaled_dy_pred, cache, fp16_params)
    
    # 4. Cast back to fp32 and unscale
    unscaled_grads = unscale_gradients(scaled_grads, scale)
    
    # 5. Check for overflow
    skipped = has_non_finite_gradients(unscaled_grads)
    
    # 6. Apply SGD update safely maintaining float32
    new_master = {}
    if skipped:
        new_master = {k: v.copy() for k, v in master_params.items()}
    else:
        # Convert lr to float32 beforehand so it doesn't taint the array types
        lr_fp32 = np.float32(lr)
        for k in master_params.keys():
            # Now float32 - (float32 * float32) stays float32 naturally
            new_master[k] = master_params[k] - lr_fp32 * unscaled_grads[k]
            
    return float(loss), new_master, skipped