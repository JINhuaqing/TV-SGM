#!/usr/bin/env python
# coding: utf-8

# This file is to train a MLP to approximate MLP for eye-close-open data
# 
# Note that I have to retrain it because we have different freqs of the PSD
# 
# Note that you should convert the scale to dB with `20 log10(x)`  (on Jun 2, 2023)

# In[1]:


RUN_PYTHON_SCRIPT = True


# In[2]:


import sys
sys.path.append("../mypkg")
from constants import RES_ROOT, FIG_ROOT, DATA_ROOT


# In[3]:


import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from easydict import EasyDict as edict
from tqdm import trange, tqdm
import time
import multiprocessing as mp
from tqdm import tqdm

if not RUN_PYTHON_SCRIPT:
    plt.style.use(FIG_ROOT/"base.mplstyle")


# In[4]:


import importlib
import models.mlp
importlib.reload(models.mlp)


# In[5]:


from utils.reparam import theta2raw_torch, raw2theta_torch, raw2theta_np
from spectrome import Brain
from sgm.sgm import SGM
from utils.misc import save_pkl, save_pkl_dict2folder, load_pkl, load_pkl_folder2dict, delta_time
from models.mlp import SGMnet


# In[6]:


# pkgs for pytorch 
import torch
import torch.nn as nn
from torch.functional import F
from torch.optim.lr_scheduler import ExponentialLR
from torch.distributions.multivariate_normal import MultivariateNormal


torch.set_default_dtype(torch.float64)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    torch.backends.cudnn.benchmark = True
else:
    torch.set_default_tensor_type(torch.DoubleTensor)


# In[ ]:





# In[ ]:





# ## Data, fn and paras

# In[7]:


# Load the Connectome
brain = Brain.Brain()
brain.add_connectome(DATA_ROOT)
brain.reorder_connectome(brain.connectome, brain.distance_matrix)
brain.bi_symmetric_c()
brain.reduce_extreme_dir()


# In[8]:


# I have need the freqs
meg_root = DATA_ROOT/"MEG-eye-multiFreqs"
fil_paths = list(meg_root.glob("*.pkl"))
data = load_pkl(fil_paths[0]);
freqs = data.freqs


# In[9]:


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
paras.names = ["alpha", "gei", "gii", "Taue", "TauG", "Taui", "Speed"]

paras.C = brain.reducedConnectome
paras.D = brain.distance_matrix
paras.freqs = freqs

paras.saved_folder = RES_ROOT/"simu_sgm_data_ind_eye_close"
if not paras.saved_folder.exists():
    paras.saved_folder.mkdir()

sgmmodel = SGM(paras.C, paras.D, paras.freqs)


# In[11]:


# running parameters
# May change

paras_run = edict()

paras_run.n = 1000
paras_run.k = 0.15 # the parameter for reparameterization in logistic
paras_run.sd = 10 # The std to generate SGM parameters in raw scale (R)


# In[ ]:





# In[ ]:





# ## Generate simulated data

# In[12]:


if not RUN_PYTHON_SCRIPT:
    #check the reparameterization
    sgm_params_raw = np.random.randn(1000, 7)*paras_run.sd
    sgm_paramss = raw2theta_np(sgm_params_raw, paras.prior_bds, k=paras_run.k)
    
    plt.figure(figsize=[20, 5])
    for ix in range(7):
        plt.subplot(2, 4, ix+1)
        sns.kdeplot(sgm_paramss[:, ix])
        plt.xlim(paras.prior_bds[ix, :])
        plt.xlabel(paras.names[ix])


# In[13]:


if not RUN_PYTHON_SCRIPT:
    #check the reparameterization
    plt.fill_between(np.arange(7), paras.par_low, paras.par_high, alpha=0.5)
    plt.yscale("log")
    for cur_ts_sgm in sgm_paramss:
        plt.plot(np.arange(7), cur_ts_sgm, color="red", alpha=0.1)
    plt.xticks(np.arange(7), paras.names);
    plt.show()
    plt.close()


# In[14]:


# the function to generate PSD with SGM
def get_psd(cur_sgm):
    cur_PSD = sgmmodel.run_local_coupling_forward(cur_sgm)
    cur_PSD = cur_PSD[:68, :]
    return cur_PSD


# In[15]:


# demo of PSD
if __name__ == "__main__":
    if not RUN_PYTHON_SCRIPT:
        sgm_params_raw = np.random.randn(100, 7)*paras_run.sd
        sgm_paramss = raw2theta_np(sgm_params_raw, paras.prior_bds, k=paras_run.k)
        
        num_cores = 20
        with mp.Pool(num_cores) as pool:
            PSDs = list(tqdm(pool.imap(get_psd, sgm_paramss), total=sgm_paramss.shape[0]))
        PSDs = np.array(PSDs)


# In[16]:


if not RUN_PYTHON_SCRIPT:
    for ix in range(100):
        dat = PSDs[ix]
        dat_std = (dat-dat.mean(axis=-1, keepdims=True))/dat.std(axis=-1, keepdims=True)
        seq = dat_std.mean(axis=0)
        plt.plot(dat_std.mean(axis=0))


# In[17]:


if not RUN_PYTHON_SCRIPT:
    ix = -20
    dat =PSDs[ix]
    dat_std = (dat-dat.mean(axis=-1, keepdims=True))/dat.std(axis=-1, keepdims=True)
    for ix in range(68):
        plt.plot(dat_std[ix])


# In[ ]:


if __name__ == "__main__":
    if RUN_PYTHON_SCRIPT:
        print(f"save to {paras.saved_folder}") 
        sgm_params_raw = np.random.randn(100000, 7)*paras_run.sd
        sgm_paramss = raw2theta_np(sgm_params_raw, paras.prior_bds, k=paras_run.k)
        
        num_cores = 20
        with mp.Pool(num_cores) as pool:
            PSDs = list(tqdm(pool.imap(get_psd, sgm_paramss), total=sgm_paramss.shape[0]))
            
        simu_sgm_data = edict()
        simu_sgm_data.PSDs = np.array(PSDs)
        simu_sgm_data.sgm_paramss =  sgm_paramss
        simu_sgm_data.freqs = paras.freqs
        save_pkl_dict2folder(paras.saved_folder, simu_sgm_data, is_force=True)


# In[ ]:


if __name__ == "__main__":
    if RUN_PYTHON_SCRIPT:
        print(f"save to {paras.saved_folder}") 
        sgm_params_raw = np.random.randn(10000, 7)*paras_run.sd
        sgm_paramss = raw2theta_np(sgm_params_raw, paras.prior_bds, k=paras_run.k)
        
        num_cores = 20
        with mp.Pool(num_cores) as pool:
            PSDs = list(tqdm(pool.imap(get_psd, sgm_paramss), total=sgm_paramss.shape[0]))
            
        simu_sgm_data = edict()
        simu_sgm_data.PSDs_test = np.array(PSDs)
        simu_sgm_data.sgm_paramss_test =  sgm_paramss
        save_pkl_dict2folder(paras.saved_folder, simu_sgm_data, is_force=True)


# In[ ]:





