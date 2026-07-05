import numpy as np

def apply_l2_regularization(params: dict, l2_lambda: float) -> dict:
    """
    Computes L2 penalty gradients for all weight matrices.
    """
    l2_grads = {}
    for k, v in params.items():
        if k.startswith('W'):
            l2_grads[k] = l2_lambda * v
        else:
            l2_grads[k] = np.zeros_like(v)
    return l2_grads
