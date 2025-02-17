import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('LayerNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('LSTM') != -1 and not (classname.find('LSTM_SGM') != -1):
        nn.init.normal_(m.weight_hh_l0.data, 0.0, 0.02)
        nn.init.normal_(m.weight_ih_l0.data, 0.0, 0.02)
        nn.init.constant_(m.bias_hh_l0.data, 0)
        nn.init.constant_(m.bias_ih_l0.data, 0)
        if m.bidirectional:
            nn.init.normal_(m.weight_ih_l0_reverse.data, 0.0, 0.02)
            nn.init.normal_(m.weight_hh_l0_reverse.data, 0.0, 0.02)
            nn.init.constant_(m.bias_ih_l0_reverse.data, 0)
            nn.init.constant_(m.bias_hh_l0_reverse.data, 0)

            
            
class MyDataset(Dataset):
    #! I try to use SGMfn to generate the PSD dynamically, but it is too slow.
    #! So let me fistly generate the PSD and save it to disk, and then load it.
    def __init__(self, X, 
                 Y=None, 
                 SGMfn=None,
                 is_std=True, 
                 dtype=torch.float32):
        """
        The dataset class for training a NN to approximate the SGM model.

        Args:
            X (torch.Tensor or array-like): the SGM parameters, n x 7 
            Y (torch.Tensor or array-like): the corresponding PSD, n x nroi x nfreq. If None, it will be generated from X.
            SGMfn (callable, optional): A function to generate the PSD from the SGM parameters. Defaults to None. The generated PSD should in abs magnitude.
            is_std (bool, optional): Whether to standardize the PSD across the freq or not. Defaults to True.
        """
        if Y is None:
            assert SGMfn is not None, "If Y is None, SGMfn should be provided."
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)

        self.dtype = dtype
        self.is_std = is_std
        self.SGMfn = SGMfn
        self.X = X.to(dtype)
        self.Y = self._preprocess_Y(Y)


    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Gets the item at the specified index.

        Args:
            idx (int or str): The index of the item to get.

        Returns:
            tuple: A tuple containing the input data and target data.
        """
        if isinstance(idx, int):
            if self.Y is not None:
                return self.X[idx], self.Y[idx]
            else:
                sgm_params = self.X[idx]
                cur_Y = self.SGMfn(sgm_params.cpu().numpy())
                cur_Y = self._preprocess_Y(cur_Y)
                return sgm_params, cur_Y
        elif isinstance(idx, str) and idx.lower().startswith("all"):
            return self.X, self.Y

    def _preprocess_Y(self, Y):
        """
        Preprocesses the target data Y.

        Args:
            Y (torch.Tensor or array-like): The PSD data. n x nroi x nfreq or nroi x nfreq 

        Returns:
            torch.Tensor: The preprocessed PSD data.
        """
        if Y is None:
            return None

        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y)
        # to dB scale
        Y = 20 * torch.log10(Y) # Y from SGM, is unsquared so use 20
        if self.is_std:
            # here I only divide by std of Y to keep the spatial features
            # In fact, it did not work for SGM to real MEG, so let me 
            # still std via the mean and std for each ROI and each data (Apr 3, 2023)
            Y = (Y-Y.mean(axis=-1, keepdims=True))/Y.std(axis=-1, keepdims=True)
        return Y.to(self.dtype)
