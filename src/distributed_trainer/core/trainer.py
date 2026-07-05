import numpy as np

from distributed_trainer.core.layers import mlp_forward, init_mlp_params
from distributed_trainer.core.autograd import mlp_backward, mlp_forward_checkpointed
from distributed_trainer.evaluation.metrics import mse_loss_and_grad
from distributed_trainer.data.distributed_sharder import shard_dataset_across_workers
from distributed_trainer.core.mixed_precision import cast_to_half_precision, scale_loss, unscale_gradients, has_non_finite_gradients
from distributed_trainer.distributed.ring_allreduce import all_reduce_mean
from distributed_trainer.optim.zero_sharded_adam import init_adam_state, partition_optimizer_state, zero_optimizer_step

def split_into_micro_batches(x, y, micro_batch_size):
    micro_batches = []
    num_samples = x.shape[0]
    for i in range(0, num_samples, micro_batch_size):
        x_mb = x[i : i + micro_batch_size]
        y_mb = y[i : i + micro_batch_size]
        micro_batches.append((x_mb, y_mb))
    return micro_batches

def accumulate_gradients(accum_grads, new_grads):
    if accum_grads is None:
        return {k: v.copy() for k, v in new_grads.items()}
    summed_grads = {k: accum_grads[k] + new_grads[k] for k in new_grads.keys()}
    return summed_grads

def scale_accumulated_gradients(accum_grads, num_micro_batches):
    return {k: v / num_micro_batches for k, v in accum_grads.items()}

def grad_accumulation_step(x, y, params, micro_batch_size):
    micro_batches = split_into_micro_batches(x, y, micro_batch_size)
    num_micro_batches = len(micro_batches)
    accum_grads = None
    
    for x_mb, y_mb in micro_batches:
        y_pred, cache = mlp_forward(x_mb, params)
        loss, dy_pred = mse_loss_and_grad(y_pred, y_mb)
        mb_grads = mlp_backward(dy_pred, cache, params)
        accum_grads = accumulate_gradients(accum_grads, mb_grads)
        
    final_grads = scale_accumulated_gradients(accum_grads, num_micro_batches)
    return final_grads

def compute_local_gradients(x, y, params):
    y_pred, cache = mlp_forward(x, params)
    loss, dy_pred = mse_loss_and_grad(y_pred, y)
    grads = mlp_backward(dy_pred, cache, params)
    return grads

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_workers = cfg.training.num_workers
        self.num_steps = cfg.training.num_steps
        self.micro_batch_size = cfg.training.micro_batch_size
        self.lr = cfg.training.learning_rate
        self.use_checkpointing = cfg.training.use_checkpointing
        self.use_mixed_precision = cfg.training.use_mixed_precision
        self.use_zero = cfg.training.use_zero
        self.seed = cfg.training.seed
        self.hidden_dim = cfg.model.hidden_dim

    def train(self, x, y):
        """
        Run an end-to-end distributed, memory-aware training loop.
        """
        in_dim = x.shape[1]
        out_dim = y.shape[1]
        params = init_mlp_params(in_dim, self.hidden_dim, out_dim, seed=self.seed)
        
        adam_state = init_adam_state(params)
        if self.use_zero:
            worker_states = partition_optimizer_state(adam_state, self.num_workers)
            
        loss_scale = 1024.0
        loss_history = []
        
        for step in range(self.num_steps):
            shards = shard_dataset_across_workers(x, y, self.num_workers)
            
            per_worker_grads = []
            step_loss_sum = 0.0
            total_micro_batches = 0
            
            for w_x, w_y in shards:
                micro_batches = split_into_micro_batches(w_x, w_y, self.micro_batch_size)
                num_mb = len(micro_batches)
                total_micro_batches += num_mb
                
                local_grads = {k: np.zeros_like(v, dtype=np.float32) for k, v in params.items()}
                
                for mb_x, mb_y in micro_batches:
                    if self.use_mixed_precision:
                        fp16_params = cast_to_half_precision(params)
                        mx = mb_x.astype(np.float16)
                        my = mb_y.astype(np.float16)
                        
                        if self.use_checkpointing:
                            y_pred, cache = mlp_forward_checkpointed(mx, fp16_params)
                        else:
                            y_pred, cache = mlp_forward(mx, fp16_params)
                            
                        loss, dy_pred = mse_loss_and_grad(y_pred, my)
                        step_loss_sum += float(loss)
                        
                        scaled_loss, scaled_dy_pred = scale_loss(loss, dy_pred, loss_scale)
                        scaled_grads = mlp_backward(scaled_dy_pred, cache, fp16_params)
                        
                        unscaled_grads = unscale_gradients(scaled_grads, loss_scale)
                        if not has_non_finite_gradients(unscaled_grads):
                            for k in local_grads:
                                local_grads[k] += unscaled_grads[k]
                    else:
                        if self.use_checkpointing:
                            y_pred, cache = mlp_forward_checkpointed(mb_x, params)
                        else:
                            y_pred, cache = mlp_forward(mb_x, params)
                            
                        loss, dy_pred = mse_loss_and_grad(y_pred, mb_y)
                        step_loss_sum += float(loss)
                        
                        grads = mlp_backward(dy_pred, cache, params)
                        for k in local_grads:
                            local_grads[k] += grads[k]
                            
                if num_mb > 0:
                    for k in local_grads:
                        local_grads[k] /= num_mb
                        
                per_worker_grads.append(local_grads)
                
            global_grads = all_reduce_mean(per_worker_grads)
            
            if self.use_zero:
                params, worker_states = zero_optimizer_step(params, global_grads, worker_states, lr=self.lr)
            else:
                adam_state['t'] += 1
                t = adam_state['t']
                for k in params:
                    m = adam_state['m'][k]
                    v = adam_state['v'][k]
                    g = global_grads[k]
                    
                    adam_state['m'][k] = 0.9 * m + 0.1 * g
                    adam_state['v'][k] = 0.999 * v + 0.001 * (g ** 2)
                    
                    m_hat = adam_state['m'][k] / (1 - 0.9 ** t)
                    v_hat = adam_state['v'][k] / (1 - 0.999 ** t)
                    
                    params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
                    
            loss_history.append(step_loss_sum / max(total_micro_batches, 1))
            
        return {'loss_history': loss_history, 'final_params': params}