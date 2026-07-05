import numpy as np

def all_reduce_mean(per_worker_grads):
    """
    Average a list of gradient dictionaries elementwise across workers.
    
    Args:
        per_worker_grads: A list of length `num_workers`, where each element 
                          is a gradient dictionary from one worker.
        
    Returns:
        averaged_grads: A single dictionary with the elementwise mean across workers.
    """
    
    # We can extract the parameter names from the first worker's dictionary
    keys = per_worker_grads[0].keys()
    averaged_grads = {}
    
    for k in keys:
        # Gather the exact same parameter array from every single worker
        worker_arrays = [worker_grads[k] for worker_grads in per_worker_grads]
        
        # np.mean(..., axis=0) stacks them logically and computes the average
        # straight down the list, perfectly preserving the original matrix shapes.
        averaged_grads[k] = np.mean(worker_arrays, axis=0)
        
    return averaged_grads

def ring_all_reduce_mean(per_worker_arrays):
    """
    Average arrays across workers using a bandwidth-efficient Ring All-Reduce algorithm.
    
    Args:
        per_worker_arrays: List of identical-shaped arrays, one from each worker.
        
    Returns:
        A single array representing the elementwise mean across all workers.
    """
    num_workers = len(per_worker_arrays)
    original_shape = per_worker_arrays[0].shape
    
    # 1. Flatten and Chunk
    # We flatten each worker's matrix into a 1D line and split it into N chunks.
    # np.array_split gracefully handles the math even if the array doesn't divide perfectly.
    worker_chunks = [np.array_split(arr.flatten(), num_workers) for arr in per_worker_arrays]
    
    # 2. Phase 1: Reduce-Scatter
    # After N-1 steps, every worker will hold the FULL sum for exactly ONE chunk.
    for s in range(num_workers - 1):
        # Create a fresh snapshot of the chunks to prevent in-place mutation bugs
        new_chunks = [[c.copy() for c in chunks] for chunks in worker_chunks]
        
        for w in range(num_workers):
            # Calculate which chunk to send and who to send it to
            send_chunk_idx = (w - s) % num_workers
            recv_w = (w + 1) % num_workers
            
            # The receiving worker adds the incoming chunk to its own tally
            new_chunks[recv_w][send_chunk_idx] += worker_chunks[w][send_chunk_idx]
            
        worker_chunks = new_chunks
        
    # 3. Phase 2: All-Gather
    # Now that each worker has one fully reduced chunk, we circulate them.
    for s in range(num_workers - 1):
        new_chunks = [[c.copy() for c in chunks] for chunks in worker_chunks]
        
        for w in range(num_workers):
            # The worker sends the chunk it just finished fully accumulating
            send_chunk_idx = (w + 1 - s) % num_workers
            recv_w = (w + 1) % num_workers
            
            # The receiving worker OVERWRITES its chunk (since the incoming one is the final sum)
            new_chunks[recv_w][send_chunk_idx] = worker_chunks[w][send_chunk_idx]
            
        worker_chunks = new_chunks
        
    # 4. Finalize
    # All workers now have identical chunks. We can grab worker 0's chunks, 
    # stitch them back together, divide by N, and reshape back to the original matrix.
    final_array = np.concatenate(worker_chunks[0])
    final_array = final_array / num_workers
    
    return final_array.reshape(original_shape)