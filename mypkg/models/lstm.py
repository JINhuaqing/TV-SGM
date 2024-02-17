# The file contains a LSTM net for prediction SGM parameters from PSD
# only speed is constant

import torch
import torch.nn as nn
from torch.functional import F
from utils.reparam import raw2theta_torch


class LSTM_SGM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, 
                 is_bidirectional=False, 
                prior_bds=None, 
                k=1, 
                dy_mask=[1, 1, 1, 1, 1, 1, 0]):
        super(LSTM_SGM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.prior_bds = prior_bds
        self.is_bidirectional = is_bidirectional
        self.k = k
        # 1 is dynamic, 0 is constant across time
        self.dy_mask = torch.tensor(dy_mask)

        self.lstm = nn.LSTM(self.input_dim,
                            self.hidden_dim,
                            num_layers=1,
                            bidirectional=self.is_bidirectional)
        
        if self.is_bidirectional:
            self.fc1 = nn.Linear(2*self.hidden_dim, 256)
            self.laynorm = nn.LayerNorm(2*self.hidden_dim)
        else:
            self.fc1 = nn.Linear(self.hidden_dim, 256)
            self.laynorm = nn.LayerNorm(self.hidden_dim)
        self.fc2 = nn.Linear(256, self.output_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
            

    def forward(self, seq):
        """
        args:
            seq: should be len_seq x n_batch x len_fs
        return:
            x: x in original scale, should be len_seq x n_batch x output_dim (7)
        """
        x, _ = self.lstm(seq) # len_seq x n_batch x len_fs
        x = self.laynorm(x)
        #x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        #x = self.dropout2(x)
        x = self.fc2(x) # len_seq x n_batch x output_dim(7)
        
        x_c = x.mean(axis=0, keepdims=True).repeat(len(seq), 1, 1)
        one_mat = torch.ones_like(x);
        mask_d = torch.stack([torch.ones_like(x[:, :, 0])*v for v in self.dy_mask], axis=2)
        mask_c = one_mat - mask_d;
        x_raw = x * mask_d + x_c * mask_c;
        
        loss_pen = torch.diff(x_raw, axis=0).abs().mean(axis=(0, 1))
        # convert x_raw to x (original sgm parameters)
        x = self.raw2theta(x_raw)
        return x, loss_pen
    
    def raw2theta(self, thetas_raw, k=None):
        """transform reparameterized theta to orignal theta
            args: thetas_raw: an array with num_sps x 7
                  prior_bds: an array with 7 x 2
        """
        if k is None:
            k = self.k
        thetas = raw2theta_torch(thetas_raw, self.prior_bds, k=k)
        return thetas 


