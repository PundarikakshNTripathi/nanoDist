from abc import ABC, abstractmethod

class BaseOptimizer(ABC):
    def __init__(self, params, lr: float):
        self.params = params
        self.lr = lr
        self.state = {}

    @abstractmethod
    def step(self, grads):
        pass

    def zero_grad(self):
        pass
