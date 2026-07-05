import numpy as np

from distributed_trainer.core.layers import mlp_forward
from distributed_trainer.core.autograd import mlp_backward
from distributed_trainer.evaluation.metrics import mse_loss_and_grad

# TODO: Implement Trainer class and training loop logic
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def train(self):
        pass

def split_into_micro_batches(x, y, micro_batch_size):
    """
    Partition a full batch (x, y) into contiguous chunks along the batch axis.
    
    Args:
        x: Input array of shape (N, in_dim).
        y: Target array of shape (N, out_dim).
        micro_batch_size: The maximum number of rows per chunk.
        
    Returns:
        A list of tuples (x_mb, y_mb), where each chunk has at most 
        micro_batch_size rows.
    """
    micro_batches = []
    num_samples = x.shape[0]
    
    # Python's range function allows stepping by micro_batch_size.
    for i in range(0, num_samples, micro_batch_size):
        # Python slicing [start:stop] handles out-of-bounds gracefully. 
        # If (i + micro_batch_size) > num_samples, it just grabs whatever is left.
        x_mb = x[i : i + micro_batch_size]
        y_mb = y[i : i + micro_batch_size]
        
        micro_batches.append((x_mb, y_mb))
        
    return micro_batches

def accumulate_gradients(accum_grads, new_grads):
    """
    Combine gradient dictionaries produced by successive micro batches.
    
    Args:
        accum_grads: The running tally of gradients (can be None on the first call).
        new_grads: The fresh gradients computed from the current micro-batch.
        
    Returns:
        A new dictionary where values are the elementwise sum of the two inputs.
    """
    # 1. Handle the very first micro-batch
    if accum_grads is None:
        # We use .copy() to ensure we don't accidentally mutate the original 
        # new_grads arrays later on.
        return {k: v.copy() for k, v in new_grads.items()}
    
    # 2. Add the new gradients to the running tally
    # The + operator automatically creates a brand new array with the summed values.
    summed_grads = {k: accum_grads[k] + new_grads[k] for k in new_grads.keys()}
    
    return summed_grads

def scale_accumulated_gradients(accum_grads, num_micro_batches):
    """
    Divide each accumulated gradient tensor by the number of micro-batches.
    
    Args:
        accum_grads: Dictionary of accumulated gradients.
        num_micro_batches: The number of micro-batches that were summed together.
        
    Returns:
        A new dictionary with the scaled gradients.
    """
    # Create a new dictionary where every tensor is divided by the scalar num_micro_batches
    return {k: v / num_micro_batches for k, v in accum_grads.items()}

def grad_accumulation_step(x, y, params, micro_batch_size):
    """
    Run one logical optimizer step under a memory budget using gradient accumulation.
    
    Args:
        x: Full batch input array, shape (N, in_dim).
        y: Full batch target array, shape (N, out_dim).
        params: Dictionary of network parameters.
        micro_batch_size: Maximum number of samples to process at once.
        
    Returns:
        A dictionary of accumulated and properly scaled gradients.
    """
    # 1. Break the massive batch into digestible chunks
    micro_batches = split_into_micro_batches(x, y, micro_batch_size)
    num_micro_batches = len(micro_batches)
    
    accum_grads = None
    
    # 2. Process each chunk sequentially
    for x_mb, y_mb in micro_batches:
        # Step A: Forward pass (guess the answer)
        y_pred, cache = mlp_forward(x_mb, params)
        
        # Step B: Calculate how wrong we were
        loss, dy_pred = mse_loss_and_grad(y_pred, y_mb)
        
        # Step C: Backward pass (calculate local gradients)
        mb_grads = mlp_backward(dy_pred, cache, params)
        
        # Step D: Throw the gradients into our running bucket
        accum_grads = accumulate_gradients(accum_grads, mb_grads)
        
    # 3. Average the bucket to simulate one massive step
    final_grads = scale_accumulated_gradients(accum_grads, num_micro_batches)
    
    return final_grads

def compute_local_gradients(x, y, params):
    """
    Compute parameter gradients for one worker's data shard.
    
    Args:
        x: The worker's assigned input shard, shape (N_shard, in_dim).
        y: The worker's assigned target shard, shape (N_shard, out_dim).
        params: Dictionary of the current network parameters.
        
    Returns:
        grads: A dictionary of parameter gradients computed ONLY on this shard.
    """
    # 1. Forward pass: The worker guesses the answers for its specific data
    y_pred, cache = mlp_forward(x, params)
    
    # 2. Loss calculation: The worker calculates how wrong its guesses were
    loss, dy_pred = mse_loss_and_grad(y_pred, y)
    
    # 3. Backward pass: The worker figures out how it would change the weights 
    # to fix its specific mistakes
    grads = mlp_backward(dy_pred, cache, params)
    
    return grads