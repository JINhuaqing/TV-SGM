
from jin_utils import get_mypkg_path
import sys
mypkg = get_mypkg_path()
sys.path.append(mypkg)

from constants import RES_ROOT, FIG_ROOT, DATA_ROOT


import numpy as np
import numpy.random as npr
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from easydict import EasyDict as edict
from tqdm import trange, tqdm
from joblib import Parallel, delayed
import time


from utils.reparam import theta2raw_torch, raw2theta_torch, raw2theta_np
from spectrome import Brain
from sgm.sgm import SGM
from utils.misc import save_pkl, save_pkl_dict2folder, load_pkl, load_pkl_folder2dict, delta_time
from utils.misc import get_cpt_ts
from utils.stable import paras_stable_check
from models.gpt import myNet
from models.model_utils import generate_position_encode
from models.loss import  weighted_mse_loss, reg_R_loss, lin_R_loss, lin_R_fn, reg_R_fn
from utils.standardize import std_mat, std_vec


# pkgs for pytorch ( Mar 27, 2023) 
import torch
import torch.nn as nn
from torch.functional import F
from torch.optim.lr_scheduler import ExponentialLR

df_dtype = torch.float32
torch.set_default_dtype(df_dtype)
if False:
#if torch.cuda.is_available():
    torch.set_default_device("cuda")
    torch.backends.cudnn.benchmark = True
else:
    torch.set_default_device("cpu")

    
seed = 1
import random
random.seed(seed)
np.random.seed(seed);
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True); 


import netCDF4
fils = list(DATA_ROOT.glob("*s100tp.nc")) # 300/150
file2read = netCDF4.Dataset(fils[0], 'r')
psd_all_full = np.array(file2read.variables["__xarray_dataarray_variable__"][:])
psd_all_full = 10 * np.log10(psd_all_full) # to dB scale, 
# make it num_sub x num_roi x num_freqs x num_ts
psd_all_full = psd_all_full.transpose(3, 0, 1, 2)
time_points = np.array(file2read.variables["timepoints"][:])
freqs = np.array(file2read.variables["frequencies"][:])
ROIs_order = np.array(file2read.variables["regionx"][:])
file2read.close()


# Load the Connectome
brain = Brain.Brain()
brain.add_connectome(DATA_ROOT)
brain.reorder_connectome(brain.connectome, brain.distance_matrix)
brain.bi_symmetric_c()
brain.reduce_extreme_dir()


# some constant parameters for this file
paras = edict()

## I reorder them in an alphabetical order and I change tauC to tauG (Mar 27, 2023)
## the orginal order is taue, taui, tauC, speed, alpha, gii, gei
## paras.par_low = np.asarray([0.005,0.005,0.005,5, 0.1,0.001,0.001])
## paras.par_high = np.asarray([0.03, 0.20, 0.03,20,  1,    2,  0.7])
##

## alpha, gei, gii, taue, tauG, taui, speed 
#paras.par_low = np.array([0.1, 0.001,0.001, 0.005, 0.005, 0.005, 5])
#paras.par_high = np.asarray([1, 0.7, 2, 0.03, 0.03, 0.20, 20])
#paras.prior_bds = np.array([paras.par_low, paras.par_high]).T
#paras.names = ["alpha", "gei", "gii", "Taue", "TauG", "Taui", "Speed"]

# alpha, gei, gii, taue, tauG, taui, speed 
paras.par_low = np.array([0.1, 0.005, 0.005, 0.005, 5])
paras.par_high = np.asarray([1, 0.03, 0.03, 0.20, 20])
paras.prior_bds = np.array([paras.par_low, paras.par_high]).T
paras.names = ["alpha", "Taue", "TauG", "Taui", "Speed"]

paras.C = brain.reducedConnectome
paras.D = brain.distance_matrix
paras.freqs = freqs


# running parameters
# May change

paras_run = edict()

paras_run.n = 360
paras_run.n_sub = 50
# note k can differ from SGM_net
paras_run.k = 1.5 # the parameter for reparameterization in logistic
#paras_run.sd = 10 # The std to generate SGM parameters in raw scale (R)


simu_sgm_data = load_pkl_folder2dict(RES_ROOT/"simu_cptts_sgm_data_conspeed_nogeigii");
trained_model = load_pkl_folder2dict(RES_ROOT/"SGM_net", excluding=['opt*']);
sgm_net = trained_model.model
sgm_net.to(dtype=df_dtype);
sgm_net.eval();

def _evaluate(all_data):
    num_sub, len_seq, _, _ = all_data.shape
    all_data_raw = torch.tensor(all_data, dtype=df_dtype)
    all_data_input = (all_data_raw - all_data_raw.mean(axis=-1, keepdims=True))/all_data_raw.std(axis=-1, keepdims=True);
    all_data_input = all_data_input.flatten(2);
    
    with torch.no_grad():
        Y_pred, _ = net(all_data_input+pos_enc);
        Y_pred_flt = Y_pred.flatten(0, 1)
        geis = torch.ones(Y_pred_flt.shape[0], 1, dtype=df_dtype, device=Y_pred.device) * 0.3 
        giis = torch.ones(Y_pred_flt.shape[0], 1, dtype=df_dtype, device=Y_pred.device) * 1
        Y_pred_flt_full = torch.cat([
                Y_pred_flt[:, :1], 
                geis, 
                giis, 
                Y_pred_flt[:, 1:],
            ], dim=1)
        X_pred = sgm_net(Y_pred_flt_full);
    corrs = reg_R_fn(all_data_raw.flatten(0, 1), X_pred);
    corrs = corrs.reshape(num_sub, len_seq, -1)
    return corrs.detach().numpy()

    
prefix = "GPTnogeigii_larger"

config = edict()
config.nfeature = 39 * 68 # the dim of features at each time point
config.target_dim = 5 # the target dim 
config.ndim = 512 # the output of the first FC layer
config.dropout = 0.0 # the dropout rate
config.n_layer = 8 # the number of self-attention layers
config.n_head = 8 # numher of heads for multi-head attention
config.is_mask = False # Use mask to make the attention causal
config.is_bias = True # Bias  for layernorm
config.block_size = 360 # the preset length of seq, 
config.batch_size = 1 # the batch size
config.k = 1
config.dy_mask = torch.tensor([1, 1, 1, 1, 0], dtype=df_dtype)

pos_enc = generate_position_encode(config.block_size, config.nfeature).unsqueeze(0)

paras_train = edict()
paras_train.niter = 2000
paras_train.loss_out = 1
paras_train.loss_pen_w = 1
paras_train.eval_out = 20
paras_train.clip = 10 # from 
paras_train.lr_step = 100
paras_train.gamma = 0.9 #!!!! 0.5
paras_train.lr = 1e-4 
paras_train.unstable_pen = 10000
paras_train.loss_name = "wmse"


post_fix = "_datataufixed"
stat_part = "_".join(np.array(paras.names)[np.array(config.dy_mask.cpu())==0][:-1])
if len(stat_part) > 0:
    folder_name = f"{prefix}_simu_net_simudatacpt_{paras_train.loss_pen_w*10}_{paras_train.loss_name}_{stat_part}{post_fix}";
else:
    folder_name = f"{prefix}_simu_net_simudatacpt_{paras_train.loss_pen_w*10}_{paras_train.loss_name}{post_fix}";
paras_train.save_dir = RES_ROOT/folder_name
print(paras_train.save_dir)



#  all_data should be num_sub x len_seq x nrois x nfreqs
#  or len_seq x nrois x nfreqs
all_data = simu_sgm_data.PSDss

# to dB
all_data = 20 * np.log10(all_data)

all_data_raw = torch.tensor(all_data, dtype=df_dtype)
all_data_input = (all_data_raw - all_data_raw.mean(axis=-1, keepdims=True))/all_data_raw.std(axis=-1, keepdims=True);
all_data_input = all_data_input.flatten(2);
# input should be num_sub x len_seq unlike lstm


net = myNet(config, 
           prior_bds=torch.tensor(paras.prior_bds, dtype=df_dtype));

net.to(dtype=df_dtype);
if paras_train.loss_name.startswith("corr"):
    loss_fn = reg_R_loss
elif paras_train.loss_name.startswith("linR"):
    loss_fn = lin_R_loss
elif paras_train.loss_name.startswith("wmse"):
    loss_fn = weighted_mse_loss
elif paras_train.loss_name.startswith("mse"):
    loss_fn = nn.MSELoss()
else:
    raise KeyError("No such loss")

optimizer = torch.optim.AdamW(net.parameters(), lr=paras_train.lr, weight_decay=0)
scheduler = ExponentialLR(optimizer, gamma=paras_train.gamma)


# training
loss_cur = 0
loss_pen_cur = 0
losses = []
losses_pen = []
losses_test = []

t0 = time.time()
sgm_net.eval()
loss_add = 0
for ix in range(paras_train.niter):
    net.train()
    X_seq = all_data_input
    # Zero the gradients
    optimizer.zero_grad()
    
    theta_pred, loss_pen = net(X_seq)
    theta_pred_flt = theta_pred.flatten(0, 1)
    geis = torch.ones(theta_pred_flt.shape[0], 1, dtype=df_dtype, device=theta_pred.device) * 0.3 
    giis = torch.ones(theta_pred_flt.shape[0], 1, dtype=df_dtype, device=theta_pred.device) * 1
    theta_pred_flt_full = torch.cat([
            theta_pred_flt[:, :1], 
            geis, 
            giis, 
            theta_pred_flt[:, 1:],
        ], dim=1)
    loss_pen = loss_pen[net.config.dy_mask==1].mean()
    X_pred = sgm_net(theta_pred_flt_full)
    #X_pred = sgm_net(theta_pred.flatten(0, 1))
    loss_main = loss_fn(X_seq.flatten(0, 1).reshape(-1, 68, len(paras.freqs)),
                   X_pred)
    if paras_train.unstable_pen > 0:
        unstable_inds = paras_stable_check(theta_pred_flt_full.detach().numpy());
        unstable_inds = torch.tensor(unstable_inds).reshape(*theta_pred.shape[:2])
        loss_add = (paras_train.unstable_pen * unstable_inds.unsqueeze(-1) * theta_pred).mean();
    loss = loss_main + loss_add + paras_train.loss_pen_w * loss_pen
    
    # Perform backward pass
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(net.parameters(), paras_train.clip)
    # Perform optimization
    optimizer.step()
    
    if ix % paras_train.lr_step == (paras_train.lr_step-1):
        scheduler.step()
        print("Learning rate is",scheduler.get_last_lr())
    
    loss_cur = loss_cur + loss_main.item()
    loss_pen_cur = loss_pen_cur + loss_pen.item()
    if ix % paras_train.loss_out == (paras_train.loss_out-1):
        losses.append(loss_cur/paras_train.loss_out)
        losses_pen.append(loss_pen_cur/paras_train.loss_out)
        print(f"At iter {ix+1}/{paras_train.niter}, "
              f"the losses are {loss_cur/paras_train.loss_out:.5f} (train). "
              f"the pen losses are {loss_pen_cur/paras_train.loss_out:.5f} (train). "
              f"The time used is {delta_time(t0):.3f}s. "
             )
        loss_cur = 0
        loss_pen_cur = 0
        t0 = time.time()
        
    if ix % paras_train.eval_out == (paras_train.eval_out-1):
        net.eval()
        loss_test = _evaluate(all_data).mean()
        losses_test.append(loss_test)
        print(f"="*100)
        print(f"At iter {ix+1}/{paras_train.niter}, "
              f"the losses on all data are {loss_test:.5f}. "
              f"The time used is {delta_time(t0):.3f}s. "
             )
        print(f"="*100)
        t0 = time.time()

if (paras_train.save_dir).exists():
    trained_model = load_pkl_folder2dict(paras_train.save_dir)
else:
    trained_model = edict()
    trained_model.model = net
    trained_model.loss_fn = loss_fn
    trained_model.optimizer = optimizer
    trained_model.paras = paras_train
    trained_model.config = config
    trained_model.loss = losses
    save_pkl_dict2folder(paras_train.save_dir, trained_model, is_force=True)

    
trained_model.model.eval()
with torch.no_grad():
    Y_pred, _ = trained_model.model(all_data_input)
sgm_paramss_est = Y_pred.cpu().numpy()
trained_model.sgm_paramss_est = sgm_paramss_est
save_pkl_dict2folder(paras_train.save_dir, trained_model, is_force=True)