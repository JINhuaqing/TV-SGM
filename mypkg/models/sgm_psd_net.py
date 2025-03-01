# this file contains the network to approximate the SGM model
# originally, I use MLP to approximate the SGM model 
# but I think I can do it better

import torch.nn as nn
from models.sgm_net_utils import FCBlock, Conv1DResidualBlock, SGMNetBase
from pprint import pprint




class SGMPSDNet(SGMNetBase):
    def __init__(self, nroi, nfreq, 
                 nlayers=2,
                 mid_dim=256, 
                 conv_channels=32,
                 dropout_rate=0.2):

        """
        A neural network to approximate the SGM model for PSD
        !add batch normalization, input batchsize must be larger than 1
        args: 
            - nroi (int): Number of regions of interest.
            - nfreq (int): Number of frequency points.
            - nlayers (int, optional): Number of conv residual blocks. Defaults to 2.
            - mid_dim (int, optional): Dimension of the intermediate features. Defaults to 256.
            - conv_channels (int, optional): Number of channels in the convolutional layer. Defaults to 32.
            - dropout_rate (float, optional): Dropout rate. Defaults to 0.2. if 0, no dropout.
        """
        super().__init__()

        out_dim = nroi*nfreq
        self.nroi = nroi
        self.nfreq = nfreq
        self.lay_init = FCBlock(in_dim=7, out_dim=mid_dim, dropout_rate=dropout_rate)
        self.link_cov = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        
        self.mid_layers = nn.ModuleList()
        for _ in range(nlayers):
            self.mid_layers.append(Conv1DResidualBlock(conv_channels, dropout_rate=dropout_rate))
        self.link_cov2 = nn.Conv1d(conv_channels, 4, kernel_size=3, padding=1)
        self.lay_out = nn.Linear(4*mid_dim, out_dim) 

        self.initialize_weights()
        n_params = self.count_parameters()
        pprint(f"total number of parameters: {n_params['total']/1e6:.2f}M")


    def forward(self, x):
        """
        The input x is in the orignal SGM parameters scale (untransformed)
        Args:
            x: tensor, batchsize x 7
        """
        x = self.lay_init(x)
        x = x.unsqueeze(1)
        x = self.link_cov(x)
        for layer in self.mid_layers:
            x = layer(x)
        x = self.link_cov2(x)
        x = x.view(x.size(0), -1)
        x = self.lay_out(x)
        x = x.view(-1, self.nroi, self.nfreq)
        # std it 
        x = (x - x.mean(axis=-1, keepdims=True))/x.std(axis=-1, keepdims=True)
        return x