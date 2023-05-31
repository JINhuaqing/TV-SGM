import torch
import numpy as np
import torch.nn as nn

def lin_R_fn(x, y):
    """
    For both torch and np
    Calculate the linear correlation coefficient (Lin's R) between x and y.
    
    Args:
    x: torch.Tensor, shape (batch_size, num_features)
    y: torch.Tensor, shape (batch_size, num_features)
    
    Returns:
    ccc: torch.Tensor, shape (batch_size,)
    """
    assert x.shape == y.shape, "x and y should have the same shape"
    x_bar = x.mean(axis=-1, keepdims=True)
    y_bar = y.mean(axis=-1, keepdims=True)
    num = ((x-x_bar)*(y-y_bar)).sum(axis=-1);
    den = (x**2).sum(axis=-1) + (y**2).sum(axis=-1) - (2 * x.shape[-1] * x_bar * y_bar).squeeze()
    ccc = num/den;
    return ccc

def lin_R_loss(x, y):
    cccs = lin_R_fn(x, y)
    return -cccs.mean()

def reg_R_fn(x, y):
    """Calculate pearons'r in batch, for both numpy and torch
    Args:
    x: torch.Tensor, shape (batch_size, num_features)
    y: torch.Tensor, shape (batch_size, num_features)
    Returns:
    corrs: torch.Tensor, shape (batch_size,)
    """
    assert x.shape == y.shape, "x and y should have the same shape"
    x_mean = x.mean(axis=-1, keepdims=True)
    y_mean = y.mean(axis=-1, keepdims=True)
    num = ((x- x_mean)*(y-y_mean)).sum(axis=-1)
    if isinstance(x, np.ndarray):
        den = np.sqrt(((x- x_mean)**2).sum(axis=-1)*((y-y_mean)**2).sum(axis=-1))
    else:
        den = torch.sqrt(((x- x_mean)**2).sum(axis=-1)*((y-y_mean)**2).sum(axis=-1))
    corrs = num/den
    return corrs

def reg_R_loss(x, y):
    corrs = reg_R_fn(x, y)
    return -corrs.mean()

def weighted_mse_loss(x, y, ws=None):
    """
    Calculates the weighted mean squared error loss between predicted and target values.

    Args:
        x(torch.Tensor): predicted values
        y(torch.Tensor): target values
        ws (torch.Tensor, optional): weights for each value. Defaults to None.

    Returns:
        torch.Tensor: weighted mean squared error loss
    """
    if ws is None:
        ws = torch.ones_like(x);
        ws_mul = torch.ones(ws.shape[-1])
        ws_mul[:int(len(ws_mul)/2)] = 10
        ws = ws * ws_mul;
    return torch.mean((x-y)**2 * ws)


# it should be the same to reg_R_fn, deprecated
def cos_simi_loss(input_, target):
    """
    Calculates the cosine similarity loss between the input and target tensors.
    
    Args:
    input_ (torch.Tensor): The input tensor.
    target (torch.Tensor): The target tensor.
    
    Returns:
    torch.Tensor: The negative mean of the cosine similarity loss.
    """
    fn = nn.CosineSimilarity(dim=-1)
    losses = fn(input_, target)
    return - losses.mean()
