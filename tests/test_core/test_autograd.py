import numpy as np
from distributed_trainer.core.layers import linear_forward, relu_forward
from distributed_trainer.core.autograd import linear_backward
from distributed_trainer.evaluation.metrics import mse_loss_and_grad

def test_linear_forward():
    x = np.ones((2, 3))
    w = np.ones((3, 4))
    b = np.zeros(4)
    out = linear_forward(x, w, b)
    assert out.shape == (2, 4)
    assert np.all(out == 3.0)

def test_relu_forward():
    x = np.array([[-1.0, 0.0, 1.0]])
    out = relu_forward(x)
    assert np.array_equal(out, np.array([[0.0, 0.0, 1.0]]))

def test_mse_loss():
    y_pred = np.array([[1.0, 1.0]])
    y_true = np.array([[1.0, 0.0]])
    loss, grad = mse_loss_and_grad(y_pred, y_true)
    assert loss == 0.25 # (0 + 1) / 2 / 2
    assert grad.shape == y_pred.shape
