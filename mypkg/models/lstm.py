# The file contains a LSTM net for prediction SGM parameters from PSD

import torch
import torch.nn as nn
from torch.functional import F
from utils.reparam import raw2theta_torch


class LSTM_SGM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, 
                 is_bidirectional=False, 
                prior_bds=None, 
                k=1):
        super(LSTM_SGM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.prior_bds = prior_bds
        self.is_bidirectional = is_bidirectional
        self.k = k

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
        self.dropout = nn.Dropout(0.5)
            

    def forward(self, seq):
        """
        args:
            seq: should be len_seg x n_batch x len_fs
        return:
            x: x in original scale, should be len_seg x n_batch x output_dim (7)
        """
        x, _ = self.lstm(seq) # len_seg x n_batch x len_fs
        x = self.laynorm(x)
        #x = self.dropout(x)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x) # len_seg x n_batch x output_dim(7)
        
        x = torch.transpose(x, 1, 0) #  n_batch x len_seg x output_dim(7)
        x_last = x[:, :, -1].mean(axis=-1, keepdims=True).repeat(1, len(seq)).unsqueeze(-1) # n_batch x len_seg x 1
        x_raw = torch.cat([x[:, :, :-1], x_last], dim=-1) #  n_batch x len_seg x output_dim(7)
        x_raw = x_raw.transpose(1, 0) # len_seg x n_batch x output_dim(7)
        
        # convert x_raw to x (original sgm parameters)
        x = self.raw2theta(x_raw)
        return x
    
    def raw2theta(self, thetas_raw, k=None):
        """transform reparameterized theta to orignal theta
            args: thetas_raw: an array with num_sps x 7
                  prior_bds: an array with 7 x 2
        """
        if k is None:
            k = self.k
        thetas = raw2theta_torch(thetas_raw, self.prior_bds, k=k)
        return thetas 


