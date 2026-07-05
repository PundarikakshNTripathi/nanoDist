import numpy as np

def bucket_gradients(grads, bucket_size):
    """
    Pack flattened gradients into fixed-size buckets for efficient communication.
    
    Args:
        grads: Dictionary of gradient arrays.
        bucket_size: The maximum number of elements a bucket should ideally hold.
        
    Returns:
        buckets: A list of 1D numpy arrays (the coalesced buffers).
        meta: A list of tuples (name, shape, start, end, bucket_index) for unpacking.
    """
    buckets = []
    meta = []
    
    current_bucket_arrays = []
    current_bucket_size = 0
    bucket_index = 0
    
    # 1. Sort the keys to guarantee absolute determinism across all GPUs
    for name in sorted(grads.keys()):
        arr = grads[name]
        shape = arr.shape
        size = arr.size
        
        # Flatten the matrix into a 1D line of numbers
        flat_arr = arr.ravel()
        
        # 2. Check if we need to seal the current bucket and start a new one
        # (We only do this if the bucket isn't empty, to allow oversized tensors to fit)
        if current_bucket_size + size > bucket_size and current_bucket_size > 0:
            buckets.append(np.concatenate(current_bucket_arrays))
            
            # Reset for the new bucket
            current_bucket_arrays = []
            current_bucket_size = 0
            bucket_index += 1
            
        # 3. Calculate this tensor's exact address within the current bucket
        start = current_bucket_size
        end = start + size
        
        # 4. Record the "treasure map" so we can unpack it later
        meta.append((name, shape, start, end, bucket_index))
        
        # Add the tensor to the bucket
        current_bucket_arrays.append(flat_arr)
        current_bucket_size += size
        
    # 5. Don't forget to seal and append the very last bucket!
    if current_bucket_arrays:
        buckets.append(np.concatenate(current_bucket_arrays))
        
    return buckets, meta

def init_adam_state(params):
    """
    Build the Adam optimizer state for a parameter dict.
    
    Args:
        params: Dictionary of network parameters (e.g., 'W1', 'b1', 'W2', 'b2').
        
    Returns:
        state: A dictionary containing the first moment 'm', second moment 'v', 
               and step counter 't'.
    """
    # np.zeros_like perfectly perfectly replicates the shape and data type of 
    # the target array, creating a fresh, independent memory buffer.
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    
    return {
        'm': m,
        'v': v,
        't': 0
    }

def partition_optimizer_state(state, num_workers):
    """
    Split Adam's first and second moment tensors across num_workers.
    
    Args:
        state: The initial Adam state containing 'm', 'v', and 't'.
        num_workers: The number of simulated workers.
        
    Returns:
        worker_states: A list of length num_workers containing partitioned states.
    """
    # Initialize the structure for each worker
    worker_states = []
    for _ in range(num_workers):
        worker_states.append({
            'm': {},
            'v': {},
            't': state['t'],
            'shard_slices': {},
            'original_shapes': {}
        })
        
    # Iterate through every parameter in the network (e.g., 'W1', 'b1', ...)
    for param_name in state['m'].keys():
        m_flat = state['m'][param_name].ravel()
        v_flat = state['v'][param_name].ravel()
        
        # Calculate exactly how to divide this parameter's flat array
        N = m_flat.size
        base_chunk_size = N // num_workers
        remainder = N % num_workers
        
        current_start = 0
        
        for w in range(num_workers):
            # The first 'remainder' workers take one extra element to perfectly cover N
            chunk_size = base_chunk_size + (1 if w < remainder else 0)
            end = current_start + chunk_size
            
            # .copy() is vital here to break the memory reference.
            # We want independent shards, not views into the original array.
            worker_states[w]['m'][param_name] = m_flat[current_start:end].copy()
            worker_states[w]['v'][param_name] = v_flat[current_start:end].copy()
            
            # Save the metadata so the worker knows exactly what part of the 
            # neural network it is holding.
            worker_states[w]['shard_slices'][param_name] = (current_start, end)
            worker_states[w]['original_shapes'][param_name] = state['m'][param_name].shape
            
            current_start = end
            
    return worker_states

def local_shard_adam_update(params, grads, worker_state, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Apply an Adam update to only the local shard of each parameter using its owned moment shards.
    
    Args:
        params: Dictionary of full network parameters.
        grads: Dictionary of full network gradients.
        worker_state: This worker's partitioned Adam state (m, v, t, shard_slices, shapes).
        lr: Learning rate.
        beta1: Exponential decay rate for first moment.
        beta2: Exponential decay rate for second moment.
        eps: Small epsilon to prevent division by zero.
        
    Returns:
        updated_param_shards: Dictionary of just the updated parameter slices.
        updated_worker_state: The new Adam state with updated m, v, and t.
    """
    # 1. Increment the step counter exactly once for this whole update
    t = worker_state['t'] + 1
    
    # 2. Prepare the new state and output dictionaries
    updated_param_shards = {}
    new_worker_state = {
        'm': {},
        'v': {},
        't': t,
        'shard_slices': worker_state['shard_slices'],
        # Use .get to gracefully handle if the key is 'shapes' (prompt) or 'original_shapes' (Step 32)
        'shapes': worker_state.get('shapes', worker_state.get('original_shapes')) 
    }
    
    # 3. Process every parameter
    for k in params.keys():
        # Get this worker's assigned slice boundaries
        start, end = worker_state['shard_slices'][k]
        
        # Flatten the full arrays and extract ONLY the piece this worker owns
        p_shard = params[k].reshape(-1)[start:end].copy()
        g_shard = grads[k].reshape(-1)[start:end]
        
        m_shard = worker_state['m'][k]
        v_shard = worker_state['v'][k]
        
        # 4. Adam Math: Update the moving averages (moments)
        m_new = beta1 * m_shard + (1 - beta1) * g_shard
        v_new = beta2 * v_shard + (1 - beta2) * (g_shard ** 2)
        
        # 5. Adam Math: Bias correction using the NEW step counter (t)
        m_hat = m_new / (1 - beta1 ** t)
        v_hat = v_new / (1 - beta2 ** t)
        
        # 6. Apply the update to this specific parameter slice
        p_shard -= lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # 7. Store the results
        updated_param_shards[k] = p_shard
        new_worker_state['m'][k] = m_new
        new_worker_state['v'][k] = v_new
        
    return updated_param_shards, new_worker_state

def all_gather_param_shards(param_shards_per_worker, shapes, shard_slices_per_worker):
    """
    All-gather per-worker 1D parameter shards and restore original shapes.
    
    Args:
        param_shards_per_worker: List of dicts containing updated 1D parameter shards.
        shapes: Dictionary mapping parameter names to their original shapes (e.g., (3, 4)).
        shard_slices_per_worker: List of dicts mapping parameter names to (start, end) tuples.
        
    Returns:
        full_params: A single dictionary of fully assembled parameters matching the original shapes.
    """
    full_params = {}
    num_workers = len(param_shards_per_worker)
    
    # Process every parameter one by one (e.g., 'W1', 'b1')
    for param_name, shape in shapes.items():
        # 1. Figure out the total number of elements required
        # If shape is (3, 4), np.prod calculates 3 * 4 = 12 total elements
        total_elements = np.prod(shape)
        
        # Peek at the first worker's shard to grab the correct data type (usually float32)
        dtype = param_shards_per_worker[0][param_name].dtype
        
        # 2. Preallocate the empty flat buffer
        flat_buffer = np.zeros(total_elements, dtype=dtype)
        
        # 3. Stitch the puzzle pieces back together
        for w in range(num_workers):
            start, end = shard_slices_per_worker[w][param_name]
            shard_data = param_shards_per_worker[w][param_name]
            
            # Precisely drop this worker's chunk into the correct slot
            flat_buffer[start:end] = shard_data
            
        # 4. Fold the 1D line back into its original 2D/3D structure
        full_params[param_name] = flat_buffer.reshape(shape)
        
    return full_params

def zero_optimizer_step(params, grads, worker_states, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Run a full ZeRO step: each worker updates its shard, then all-gather rebuilds full params.
    
    Args:
        params: Full network parameters.
        grads: Full network gradients.
        worker_states: List of each worker's partitioned Adam state.
        lr, beta1, beta2, eps: Adam hyperparameters.
        
    Returns:
        new_params: The fully reassembled, updated parameters.
        new_worker_states: The list of updated Adam states for each worker.
    """
    param_shards_per_worker = []
    new_worker_states = []
    shard_slices_per_worker = []
    
    # 1. Local Compute Phase (Parallel Execution)
    # Simulate each GPU applying the Adam math ONLY to its assigned slices.
    for ws in worker_states:
        updated_param_shards, new_ws = local_shard_adam_update(
            params, grads, ws, lr=lr, beta1=beta1, beta2=beta2, eps=eps
        )
        
        param_shards_per_worker.append(updated_param_shards)
        new_worker_states.append(new_ws)
        shard_slices_per_worker.append(ws['shard_slices'])
        
    # Extract the original network shapes (identical across all workers, so we grab from worker 0)
    shapes = worker_states[0].get('shapes', worker_states[0].get('original_shapes'))
    
    # 2. Network Communication Phase (All-Gather)
    # Broadcast every worker's updated slices to every other worker, stitching 
    # the shattered neural network back together into a whole model.
    new_params = all_gather_param_shards(
        param_shards_per_worker, 
        shapes, 
        shard_slices_per_worker
    )
    
    return new_params, new_worker_states