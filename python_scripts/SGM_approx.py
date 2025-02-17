#!/usr/bin/env python
# coding: utf-8

# This file is to train a MLP to approximate MLP
# 
# PSD of SGM is in abs magnitude, 
# 
# but I want to train a MLP to approximate the PSD of SGM in log magnitude (in dB)

# In[23]:


from jin_utils import get_mypkg_path
import sys
mypkg = get_mypkg_path()
sys.path.append(mypkg)

from constants import RES_ROOT, FIG_ROOT, DATA_ROOT, SGM_prior_bds, MIDRES_ROOT


import numpy as np
import scipy
from easydict import EasyDict as edict
from tqdm import trange, tqdm
import time



# In[5]:


from utils.misc import save_pkl_dict2folder, load_pkl_folder2dict, delta_time
from models.mlp import SGMnet


# In[6]:


# pkgs for pytorch ( Mar 27, 2023) 
import torch
import torch.nn as nn
from torch.functional import F
from torch.optim.lr_scheduler import ExponentialLR


df_dtype = torch.float32
torch.set_default_dtype(df_dtype)

if torch.cuda.is_available():
    print("use GPU for training")
    torch.set_default_device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    torch.set_default_device("cpu")


# In[7]:


from torch.utils.data import DataLoader
from models.model_utils import MyDataset
from models.loss import weighted_mse_loss, cos_simi_loss


# load the dataset to get the freqs in real data (Apr 2, 2023)
import netCDF4
fils = list(DATA_ROOT.glob("*s100tp.nc"))
file2read = netCDF4.Dataset(fils[0], 'r')
psd_all = np.array(file2read.variables["__xarray_dataarray_variable__"][:])
time_points = np.array(file2read.variables["timepoints"][:])
freqs = np.array(file2read.variables["frequencies"][:])
ROIs_order = np.array(file2read.variables["regionx"][:])
file2read.close()


data_path = MIDRES_ROOT/"sgm_pairs_ntrain100000_ntest10000"
simu_sgm_data = load_pkl_folder2dict(data_path);


save_path = RES_ROOT/f"sgmnet_{data_path.stem}"

paras_sgm_net = edict()
paras_sgm_net.batchsize = 512
paras_sgm_net.nepoch = 1000

paras_sgm_net.loss_out = 1
paras_sgm_net.lr_step = 200
paras_sgm_net.lr_gamma = 0.1
paras_sgm_net.lr = 1e-3
paras_sgm_net.weight_decay = 0

# We can use Pearsons R as loss, but let me try this later (on Apr 6, 2023)
#loss_fn = nn.MSELoss()
paras_sgm_net.loss_fn = weighted_mse_loss


# In[40]:


# the data loader for training and testing
train_data = MyDataset(
    X = simu_sgm_data.sgm_paramss, 
    Y=simu_sgm_data.PSDs, 
    SGMfn=None, 
    is_std=True, 
    dtype=df_dtype)
train_data_loader = DataLoader(train_data, batch_size=paras_sgm_net.batchsize, shuffle=True)

test_data = MyDataset(
    X=simu_sgm_data.sgm_paramss_test, 
    Y=simu_sgm_data.PSDs_test, 
    SGMfn=None,
    is_std=True, 
    dtype=df_dtype)
test_data_loader = DataLoader(test_data, batch_size=paras_sgm_net.batchsize, shuffle=False)


# In[41]:


# the network
sgm_net = SGMnet(nroi=68, nfreq=len(freqs)).to(df_dtype)
optimizer = torch.optim.Adam(sgm_net.parameters(), 
                             lr=paras_sgm_net.lr, 
                             weight_decay=paras_sgm_net.weight_decay)
scheduler = ExponentialLR(optimizer, gamma=paras_sgm_net.lr_gamma)


# In[42]:


def evaluate(test_data_loader, net, loss_fn=None):
    if loss_fn is None:
        loss_fn = paras_sgm_net.loss_fn
    net.eval()
    losses = []
    with torch.no_grad():
        for X_batch, Y_batch in test_data_loader:
            Y_batch_pred = net(X_batch)
            loss = loss_fn(Y_batch, Y_batch_pred)
            losses.append(loss.item())
    net.train()
    return np.mean(losses)


# In[ ]:


# training
loss_cur = []
# mean losses
losses = []
losses_test = []

t0 = time.time()
for ie in range(paras_sgm_net.nepoch):
    for X_batch, Y_batch in train_data_loader:
        # Zero the gradients
        optimizer.zero_grad()
        
        Y_batch_est = sgm_net(X_batch)
        #print(Y_batch_est.sum(axis=(1, 2)))
        
        loss = paras_sgm_net.loss_fn(Y_batch, Y_batch_est)
        #print(loss)
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        loss_cur.append(loss.item())
        
    if ie % paras_sgm_net.lr_step == (paras_sgm_net.lr_step-1):
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]
        print(f"At epoch {ie+1}/{paras_sgm_net.nepoch}, the learning rate is {cur_lr:.5e}")
    if ie % paras_sgm_net.loss_out == (paras_sgm_net.loss_out-1):
    
        losses.append(np.mean(loss_cur))
        losses_test.append(evaluate(test_data_loader, sgm_net))
        print(f"At epoch {ie+1}/{paras_sgm_net.nepoch},"
              f"the losses are {losses[-1]:.5f} (train)"
              f" and {losses_test[-1]:.5f} (test). "
              f"The time used is {delta_time(t0):.3f}s. "
              )
        loss_cur = []
        t0 = time.time()
    

trained_model = edict()
trained_model.model = sgm_net.cpu()
trained_model.optimizer = optimizer
trained_model.paras = paras_sgm_net
trained_model.loss = losses
trained_model.loss_test = losses_test
trained_model.freqs = freqs
save_pkl_dict2folder(save_path, trained_model, is_force=True)

