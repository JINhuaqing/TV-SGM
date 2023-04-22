#!/usr/bin/env python
# coding: utf-8

# Analyze the results from LSTM

# In[40]:


run_python_script = True


# In[1]:


import sys
sys.path.append("../mypkg")
from constants import RES_ROOT, FIG_ROOT, DATA_ROOT


# In[2]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from easydict import EasyDict as edict
from tqdm import trange, tqdm
import time


# In[3]:


import importlib
import models.lstm
importlib.reload(models.lstm)


# In[4]:


from utils.reparam import theta2raw_torch, raw2theta_torch, raw2theta_np
from spectrome import Brain
from sgm.sgm import SGM
from utils.misc import save_pkl, save_pkl_dict2folder, load_pkl, load_pkl_folder2dict, delta_time
from models.lstm import LSTM_SGM
from models.loss import cos_simi_loss, weighted_mse_loss
from utils.standardize import std_mat, std_vec


# In[5]:


# pkgs for pytorch ( Mar 27, 2023) 
import torch
import torch.nn as nn
from torch.functional import F
from torch.optim.lr_scheduler import ExponentialLR

torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    torch.cuda.set_device(2)
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    torch.backends.cudnn.benchmark = True
else:
    torch.set_default_tensor_type(torch.DoubleTensor)


# In[ ]:





# # Data, fn and paras

# In[6]:


import netCDF4
fils = list(DATA_ROOT.glob("*s100tp.nc"))
file2read = netCDF4.Dataset(fils[0], 'r')
psd_all = np.array(file2read.variables["__xarray_dataarray_variable__"][:])
time_points = np.array(file2read.variables["timepoints"][:])
freqs = np.array(file2read.variables["frequencies"][:])
ROIs_order = np.array(file2read.variables["regionx"][:])
file2read.close()


# In[7]:


# Load the Connectome
brain = Brain.Brain()
brain.add_connectome(DATA_ROOT)
brain.reorder_connectome(brain.connectome, brain.distance_matrix)
brain.bi_symmetric_c()
brain.reduce_extreme_dir()


# In[32]:


# some constant parameters for this file
paras = edict()

## I reorder them in an alphabetical order and I change tauC to tauG (Mar 27, 2023)
## the orginal order is taue, taui, tauC, speed, alpha, gii, gei
## paras.par_low = np.asarray([0.005,0.005,0.005,5, 0.1,0.001,0.001])
## paras.par_high = np.asarray([0.03, 0.20, 0.03,20,  1,    2,  0.7])
##

# alpha, gei, gii, taue, tauG, taui, speed 
paras.par_low = np.array([0.1, 0.001,0.001, 0.005, 0.005, 0.005, 5])
paras.par_high = np.asarray([1, 0.7, 2, 0.03, 0.03, 0.20, 20])
paras.prior_bds = np.array([paras.par_low, paras.par_high]).T
paras.names = np.array(["alpha", "gei", "gii", "Taue", "TauG", "Taui", "Speed"])

paras.C = brain.reducedConnectome
paras.D = brain.distance_matrix
paras.freqs = freqs


# In[16]:


std_fn = lambda x: (x-x.mean(axis=-1, keepdims=True))/x.std(axis=-1, keepdims=True)


# In[11]:


# functions to generate training sample (Apr 1, 2023)
def random_choice(n, batchsize=1, len_seg=None):
    """Randomly select the lower and upper bound of the segment
        args:
            n: len of the total time series
    """
    if len_seg is None:
        len_seg = torch.randint(low=10, high=100, size=(1, ))
    up_bd = torch.randint(low=len_seg.item(), high=n, size=(batchsize, ))
    low_bd = up_bd - len_seg
    return low_bd, up_bd


def random_samples_rnn(X, Y=None, batchsize=1, 
                       bds=None, 
                       is_std=True, 
                       theta2raw_fn=None):
    """Randomly select a sample from the whole segment
        args:
            X: PSD, num_seq x 68 x nfreq or 
               PSD, num_sub x num_seq x 68 x nfreq
            Y: params, num x 7, in original sgm scale
        return:
            X_seqs: len_seq x batchsize x num_fs
            Y_seqs: len_seq x batchsize x 7
            
    """
    if X.ndim == 4:
        # if multiple subjects, pick up a subject
        num_sub = X.shape[0]
        sub_idx = np.random.randint(low=0, high=num_sub)
        X = X[sub_idx]
        
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X)
    if is_std:
        #X = X/X.std(axis=(1, 2), keepdims=True)
        # Let std for each ROI and each data
        X = (X-X.mean(axis=2, keepdims=True))/X.std(axis=2, keepdims=True)
    if Y is not None:
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y)
        if theta2raw_fn: 
            Y = theta2raw_fn(Y)
    if bds is None:
        low_bds, up_bds = random_choice(len(X), batchsize)
    else:
        low_bds, up_bds = bds

    X = X.flatten(1)
    X_seqs = []
    Y_seqs = []
    for low_bd, up_bd in zip(low_bds, up_bds):
        X_seq = X[low_bd:up_bd, :].unsqueeze(1)
        X_seqs.append(X_seq)
        if Y is not None:
            Y_seq = Y[low_bd:up_bd].unsqueeze(1)
            Y_seqs.append(Y_seq)
    if Y is not None:
        return torch.cat(X_seqs, dim=1), torch.cat(Y_seqs, dim=1)
    else:
        return torch.cat(X_seqs, dim=1)
        


# In[ ]:





# # Load the model

# In[13]:


trained_model_sgm = load_pkl_folder2dict(RES_ROOT/"SGM_net", excluding=['opt*']);
sgm_net = trained_model_sgm.model;


# In[18]:


fixed = "_gain"
trained_model_lstm = load_pkl_folder2dict(RES_ROOT/f"LSTM_simu_net{fixed}", excluding=['opt*']);
lstm_net = trained_model_lstm.model;


# # Evaluate

# ## sgm parameters

# In[30]:


dy_mask = np.array(trained_model_lstm.paras.dy_mask)


# In[19]:


trained_model_lstm.model.eval()
sgm_paramss_est = []
for data_idx  in range(36):
    cur_data = psd_all[:, :, :, data_idx].transpose(2, 0, 1)
    X_test = random_samples_rnn(cur_data,
                                bds=[[0], [360]],
                                theta2raw_fn=None)
    with torch.no_grad():
        Y_pred = trained_model_lstm.model(X_test).squeeze()
        sgm_paramss_est.append(Y_pred.numpy())
sgm_paramss_est = np.array(sgm_paramss_est);


# In[22]:


if not run_python_script:
    plt.figure(figsize=[20, 5*np.sum(dy_mask)])
    for ix in range(7):
        if dy_mask[ix] == 1:
            plt.subplot(np.sum(dy_mask), 1, ix+1)
            plt.title(f"Estimation of {paras.names[ix]}", fontsize=25)
            sns.heatmap(sgm_paramss_est[:, :, ix])
            plt.xlabel("Time")
            plt.ylabel("Subject")


# In[34]:


if not run_python_script:
    corr_mat = np.array([np.corrcoef(sgm_params_est[:, dy_mask==1].T) for sgm_params_est in sgm_paramss_est]).mean(axis=0)
    plt.figure(figsize=[10, 10])
    plt.title(f"Average corr between SGM parameters across 36 subjects", fontsize=30)
    sns.heatmap(corr_mat, square=True, annot=True, cbar=False,
                annot_kws=dict(fontsize=15))
    plt.xticks(np.arange(dy_mask.sum())+0.5, paras.names[dy_mask==1])
    plt.yticks(np.arange(dy_mask.sum())+0.5, paras.names[dy_mask==1]);


# In[ ]:





# ### Example

# In[35]:


if not run_python_script:
    sub_idx = 0
    sgm_params_est = sgm_paramss_est[sub_idx]
    plt.figure(figsize=[20, 5])
    plt.suptitle(f"Normalized curves (Subj {sub_idx})", fontsize=30)
    for ix in range(7):
        plt.subplot(2, 4, ix+1)
        plt.plot(sgm_params_est[:, ix])
        plt.title(paras.names[ix], fontsize=20)


# In[36]:


if not run_python_script:
    plt.figure(figsize=[10, 10])
    plt.title(f"Correlation between SGM parameters(Subj {sub_idx})", fontsize=30)
    sns.heatmap(np.corrcoef(sgm_params_est[:, dy_mask==1].T), square=True, annot=True, cbar=False,
                annot_kws=dict(fontsize=15))
    plt.xticks(np.arange(dy_mask.sum())+0.5, paras.names[dy_mask==1])
    plt.yticks(np.arange(dy_mask.sum())+0.5, paras.names[dy_mask==1]);


# In[ ]:





# ### PSD

# In[37]:


sgmmodel = SGM(paras.C, paras.D, paras.freqs)
X_recs = []
for sgm_params_est in sgm_paramss_est:
    X_rec = []
    for sgm_param in tqdm(sgm_params_est):
        cur_PSD = sgmmodel.run_local_coupling_forward(sgm_param)
        X_rec.append(cur_PSD[:68])
    X_recs.append(X_rec)


# In[39]:


# save
trained_model_lstm.Rec_PSD = np.array(X_recs)
save_pkl_dict2folder(RES_ROOT/f"LSTM_simu_net{fixed}", trained_model_lstm, is_force=False)


