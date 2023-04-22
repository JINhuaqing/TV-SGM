import torch
import numpy as np

def generate_position_encode(block_size, nfeature):
    """
    Generate positional encoding for a given block size and number of features.

    Args:
        block_size (int): The size of the block.
        nfeature (int): The number of features.

    Returns:
        pos_enc (torch.Tensor): The positional encoding tensor.
    """
    # create a matrix with shape (blocksize, nfeature)
    position = torch.arange(block_size, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, nfeature, 2).float() * (-np.log(10000.0) / nfeature))
    pos_enc = torch.zeros((block_size, nfeature))
    # apply sine to even indices in the array
    pos_enc[:, 0::2] = torch.sin(position * div_term)
    # apply cosine to odd indices in the array
    # to avoid the case when nfeature is odd
    pos_enc[:, 1::2] = torch.cos(position * div_term[:int(nfeature/2)])
    return pos_enc