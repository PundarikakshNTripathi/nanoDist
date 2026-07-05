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