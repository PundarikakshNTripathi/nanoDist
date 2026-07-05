import numpy as np

from distributed_trainer.data.distributed_sharder import shard_dataset_across_workers
from distributed_trainer.core.trainer import compute_local_gradients
from distributed_trainer.distributed.ring_allreduce import all_reduce_mean

def data_parallel_train_step(x, y, params, num_workers, lr):
    """
    Perform one synchronous data parallel SGD update.
    
    Args:
        x: Full batch input array, shape (N, in_dim).
        y: Full batch target array, shape (N, out_dim).
        params: Dictionary of network parameters.
        num_workers: The number of simulated workers (GPUs).
        lr: Learning rate for the SGD update.
        
    Returns:
        new_params: The updated parameters dict.
    """
    # 1. Distribute the data
    # Chop the massive batch into smaller, non-overlapping shards for each worker.
    shards = shard_dataset_across_workers(x, y, num_workers)
    
    # 2. Compute Phase (Math)
    # Simulate each worker calculating gradients on its own isolated piece of data.
    per_worker_grads = []
    for x_shard, y_shard in shards:
        local_grads = compute_local_gradients(x_shard, y_shard, params)
        per_worker_grads.append(local_grads)
        
    # 3. Communication Phase (Network)
    # Force the workers to get on a conference call and average their opinions together.
    global_grads = all_reduce_mean(per_worker_grads)
    
    # 4. Update Phase (SGD)
    # Every worker takes the exact same synchronized gradient and updates its clone
    # of the neural network.
    new_params = {}
    for k in params.keys():
        new_params[k] = params[k] - lr * global_grads[k]
        
    return new_params