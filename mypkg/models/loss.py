import torch
import numpy as np
import torch.nn as nn


def weighted_mse_loss(pred, target, ws=None):
    """
    Calculates the weighted mean squared error loss between predicted and target values.

    Args:
        pred (torch.Tensor): predicted values
        target (torch.Tensor): target values
        ws (torch.Tensor, optional): weights for each value. Defaults to None.

    Returns:
        torch.Tensor: weighted mean squared error loss
    """
    if ws is None:
        ws = torch.ones_like(pred[0])
        ws[:, :20] = ws[:, :20]*10
    return torch.mean((pred-target)**2 * ws)

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