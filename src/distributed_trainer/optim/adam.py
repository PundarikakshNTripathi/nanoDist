import numpy as np
from .base_optimizer import BaseOptimizer

class AdamOptimizer(BaseOptimizer):
    def __init__(self, params, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.state = {
            'm': {k: np.zeros_like(v) for k, v in params.items()},
            'v': {k: np.zeros_like(v) for k, v in params.items()},
            't': 0
        }

    def step(self, grads):
        self.state['t'] += 1
        t = self.state['t']
        
        for k in self.params.keys():
            m = self.state['m'][k]
            v = self.state['v'][k]
            g = grads[k]
            
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * (g ** 2)
            
            self.state['m'][k] = m
            self.state['v'][k] = v
            
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)
            
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
