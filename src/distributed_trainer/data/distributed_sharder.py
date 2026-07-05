import numpy as np

def shard_dataset_across_workers(x, y, num_workers):
    """
    Split the dataset (x, y) along the batch axis into contiguous shards.
    
    Args:
        x: Input array of shape (N, in_dim).
        y: Target array of shape (N, out_dim).
        num_workers: The number of workers/GPUs.
        
    Returns:
        A list of length num_workers containing tuples of (x_shard, y_shard).
    """
    # np.array_split differs from np.split. It does not raise an error if the 
    # array doesn't divide equally. Instead, it automatically distributes the 
    # remainder across the first few sub-arrays.
    x_shards = np.array_split(x, num_workers, axis=0)
    y_shards = np.array_split(y, num_workers, axis=0)
    
    # zip pairs them up element-by-element, and list() materializes it.
    return list(zip(x_shards, y_shards))