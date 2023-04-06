# This file contains the MLP to approximate SGM given the sgm parmaters (Mar 30, 2023). 

import torch.nn as nn
from torch.functional import F


class SGMnet(nn.Module):
    def __init__(self, nroi, nfreq, is_large=True):
        """
        A multi-layer perceptron (MLP) to approximate the SGM model.
    
        Args:
            nroi (int): Number of regions of interest.
            nfreq (int): Number of frequency points.
            is_large (bool): Whether to use a large model architecture.
    
        Attributes:
            nroi (int): Number of regions of interest.
            nfreq (int): Number of frequency bands.
            is_large (bool): Whether to use a large model architecture.
            lay_init (nn.Linear): Initial linear layer.
            layers1 (nn.Sequential): First set of linear layers.
            layers2 (nn.Sequential): Second set of linear layers.
    
        Methods:
            forward(x): Forward pass of the MLP.
    
        """
        super(SGMnet, self).__init__()

        output_size = nroi*nfreq
        self.is_large = is_large
        self.nroi = nroi
        self.nfreq = nfreq
        self.lay_init = nn.Linear(7, 256)
        self.dropout = nn.Dropout(0.5)
        
        self.layers1 = nn.Sequential(
                      nn.Linear(256, 256),
                      nn.ReLU(),
                      nn.Linear(256, 256),
                    )
        self.layers2 = nn.Sequential(
                      nn.Linear(256, 512),
                      nn.ReLU(),
                      nn.Linear(512, 1024),
                      nn.ReLU(),
                      nn.Dropout(0.5),
                      nn.Linear(1024, output_size),
                    )


    def forward(self, x):
        """
        The input x is in the orignal SGM parameters scale (untransformed)
        Args:
            x: tensor, batchsize x 7 
        """
        x = F.relu(self.lay_init(x))
        if self.is_large:
            residual = x
            x = self.layers1(x)
            x = F.relu(x + residual)
            #x = self.dropout(x) # not converage Mar 31, 2023
        x = self.layers2(x)
        x = x.reshape(-1, self.nroi, self.nfreq)
        # std it Apr 3, 2023
        x = (x - x.mean(axis=-1, keepdims=True))/x.std(axis=-1, keepdims=True)
        return x