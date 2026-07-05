import numpy as np

def mse_loss_and_grad(y_pred, y_true):
    """
    Compute the mean squared error (MSE) loss and its gradient with respect to y_pred.
    
    Args:
        y_pred: Predictions array of shape (N, D).
        y_true: Ground truth targets array of shape (N, D).
        
    Returns:
        loss: Python float representing the overall error.
        dy_pred: Gradient array of shape (N, D) indicating how to change y_pred.
    """
    # y_pred.size gives the total number of elements in the matrix (N * D)
    total_elements = y_pred.size
    
    # 1. Compute the loss (Mean Squared Error)
    # np.mean averages over all elements automatically.
    loss = float(np.mean((y_pred - y_true) ** 2))
    
    # 2. Compute the gradient of the loss with respect to the predictions
    # The derivative of x^2 is 2x. We must divide by total_elements to match 
    # the 'mean' operation used in the loss calculation.
    dy_pred = (2.0 / total_elements) * (y_pred - y_true)
    
    return loss, dy_pred