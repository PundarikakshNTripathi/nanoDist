import numpy as np

from distributed_trainer.core.layers import mlp_forward
from distributed_trainer.core.autograd import mlp_forward_checkpointed
from distributed_trainer.core.mixed_precision import cast_to_half_precision
from distributed_trainer.optim.zero_sharded_adam import init_adam_state

def estimate_checkpointing_memory_savings(batch_size, in_dim, hidden_dim, out_dim, dtype_bytes):
    """
    Estimate activation memory in bytes for full vs checkpointed forward on the two-layer MLP.
    
    Args:
        batch_size: Number of samples in the micro-batch.
        in_dim: Input feature dimension.
        hidden_dim: Hidden layer dimension.
        out_dim: Output prediction dimension.
        dtype_bytes: Number of bytes per number (e.g., 4 for float32, 8 for float64).
        
    Returns:
        A dictionary with the calculated memory footprints in bytes.
    """
    # 1. Calculate the memory footprint of each tensor
    # Memory = number_of_rows * number_of_columns * bytes_per_element
    x_bytes = batch_size * in_dim * dtype_bytes
    z1_bytes = batch_size * hidden_dim * dtype_bytes
    a1_bytes = batch_size * hidden_dim * dtype_bytes
    
    # 2. The Full Strategy: Store the input AND the intermediates
    full_bytes = x_bytes + z1_bytes + a1_bytes
    
    # 3. The Checkpoint Strategy: Store ONLY the input
    checkpoint_bytes = x_bytes
    
    # 4. The Savings
    saved_bytes = full_bytes - checkpoint_bytes
    
    return {
        'full_bytes': full_bytes,
        'checkpoint_bytes': checkpoint_bytes,
        'saved_bytes': saved_bytes
    }

def compute_param_memory_bytes(params):
    """
    Sum the total bytes occupied by every parameter array in the dict.
    
    Args:
        params: Dictionary mapping parameter names to NumPy arrays.
        
    Returns:
        total_bytes: Integer representing the total memory consumption in bytes.
    """
    # NumPy arrays have a built-in '.nbytes' attribute which automatically 
    # calculates (number_of_elements * bytes_per_element).
    return sum(arr.nbytes for arr in params.values())

def compute_optimizer_memory_bytes(state, num_workers=1, sharded=False):
    """
    Calculate the per-worker memory footprint in bytes of the Adam optimizer state.
    
    Args:
        state: Adam state dict containing 'm', 'v' (dicts of arrays) and 't' (scalar).
        num_workers: Total number of workers (GPUs).
        sharded: Boolean flag indicating if ZeRO optimizer sharding is enabled.
        
    Returns:
        total_bytes: Integer representing the per-worker memory consumption in bytes.
    """
    # 1. Sum the bytes for all arrays in the first moment (m)
    m_bytes = sum(arr.nbytes for arr in state['m'].values())
    
    # 2. Sum the bytes for all arrays in the second moment (v)
    v_bytes = sum(arr.nbytes for arr in state['v'].values())
    
    total_bytes = m_bytes + v_bytes
    
    # 3. If ZeRO sharding is enabled, divide the memory burden across the workers
    if sharded:
        # Use integer division because byte counts are discrete numbers
        return total_bytes // num_workers
        
    return total_bytes

def compute_peak_activation_memory_bytes(x, params, checkpointed=False):
    """
    Measure the total bytes of activations retained after a forward pass.
    
    Args:
        x: Input batch array, shape (N, in_dim).
        params: Dictionary of network parameters.
        checkpointed: Boolean flag to use the memory-efficient forward pass.
        
    Returns:
        total_bytes: Integer representing the total bytes retained in the cache.
    """
    # 1. Run the appropriate forward pass based on the strategy
    if checkpointed:
        y_pred, cache = mlp_forward_checkpointed(x, params)
    else:
        y_pred, cache = mlp_forward(x, params)
        
    # 2. Calculate the memory of the "scratchpad" (the cache)
    # We ignore the final prediction (y_pred) and the parameters themselves, 
    # as we only care about the intermediate values saved specifically for backprop.
    total_bytes = sum(arr.nbytes for arr in cache.values())
    
    return int(total_bytes)

def compare_memory_with_and_without_optimizations(x, params, num_workers):
    """
    Compare per-worker memory footprint of a baseline trainer vs an optimized one.
    
    Args:
        x: Input batch array in float32.
        params: Dictionary of full-precision (float32) parameters.
        num_workers: Total number of GPUs to simulate sharding across.
        
    Returns:
        A dictionary containing the total bytes, breakdowns, and savings ratio.
    """
    # ==========================================
    # 1. BASELINE: The Naive "Memory Hog" Setup
    # ==========================================
    # Float32 parameters
    base_params_bytes = compute_param_memory_bytes(params)
    
    # Unsharded Float32 Adam state (every worker holds the full state)
    adam_state = init_adam_state(params)
    base_opt_bytes = compute_optimizer_memory_bytes(adam_state, num_workers, sharded=False)
    
    # Float32 inputs through an uncheckpointed forward pass
    base_act_bytes = compute_peak_activation_memory_bytes(x, params, checkpointed=False)
    
    base_total = base_params_bytes + base_opt_bytes + base_act_bytes
    
    # ==========================================
    # 2. OPTIMIZED: The State-of-the-Art Setup
    # ==========================================
    # Cast weights and inputs to float16 for the forward/backward passes
    params_fp16 = cast_to_half_precision(params)
    x_fp16 = x.astype(np.float16)
    
    # Float16 parameters
    opt_params_bytes = compute_param_memory_bytes(params_fp16)
    
    # ZeRO-sharded Adam state (each worker holds 1/N of the float32 moments)
    opt_opt_bytes = compute_optimizer_memory_bytes(adam_state, num_workers, sharded=True)
    
    # Float16 inputs through a checkpointed forward pass (only saving 'x')
    opt_act_bytes = compute_peak_activation_memory_bytes(x_fp16, params_fp16, checkpointed=True)
    
    opt_total = opt_params_bytes + opt_opt_bytes + opt_act_bytes
    
    # ==========================================
    # 3. REPORT
    # ==========================================
    savings_ratio = (base_total - opt_total) / base_total
    
    return {
        'baseline_bytes': base_total,
        'optimized_bytes': opt_total,
        'breakdown_baseline': {
            'params': base_params_bytes,
            'optimizer': base_opt_bytes,
            'activations': base_act_bytes
        },
        'breakdown_optimized': {
            'params': opt_params_bytes,
            'optimizer': opt_opt_bytes,
            'activations': opt_act_bytes
        },
        'savings_ratio': savings_ratio
    }