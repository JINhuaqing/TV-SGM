# this file is to generate SGM pairs. 
# SGM pairs depends on the SC, distmat, as well as the freq range.
# the saved PSD is in abs magnitude (not std)
# the save FC is in original scale (not minmax), diagonal is removed, FC >=0
#! If you change the data, you need to change the freq range in this file.

from jin_utils import get_mypkg_path
import sys
mypkg = get_mypkg_path()
sys.path.append(mypkg)

from constants import RES_ROOT, DATA_ROOT, SGM_prior_bds, MIDRES_ROOT

import numpy as np
from easydict import EasyDict as edict
from tqdm import tqdm
from joblib import Parallel, delayed

from utils.reparam import ThetaTransform
from spectrome import Brain
from sgm.sgm import SGM
from utils.misc import save_pkl_dict2folder, load_pkl

np.random.seed(423)
n_jobs = 20

# load the dataset to get the freqs in real data 
import netCDF4
fils = list(DATA_ROOT.glob("*s100tp.nc"))
file2read = netCDF4.Dataset(fils[0], 'r')
psd_all = np.array(file2read.variables["__xarray_dataarray_variable__"][:])
time_points = np.array(file2read.variables["timepoints"][:])
freqs = np.array(file2read.variables["frequencies"][:])
ROIs_order = np.array(file2read.variables["regionx"][:])
file2read.close()

# Load the Connectome and Distance Matrix
brain = Brain.Brain()
brain.add_connectome(DATA_ROOT)
brain.reorder_connectome(brain.connectome, brain.distance_matrix)
brain.bi_symmetric_c()
brain.reduce_extreme_dir()



# some constant parameters for this file
paras = edict()

# alpha, gei, gii, taue, tauG, taui, speed 
paras.names = ["alpha", "gei", "gii", "taue", "taug", "taui", "speed"]
paras.prior_bds = np.array([SGM_prior_bds[key] for key in paras.names])

paras.C = brain.reducedConnectome
paras.D = brain.distance_matrix
paras.freqs = freqs
sgm = SGM(paras.C, paras.D, paras.freqs)


paras_run = edict()
paras_run.n_train = 100000
paras_run.n_test = 10000
paras_run.k = 0.15 # the parameter for reparameterization in logistic
paras_run.sd = 10 # The std to generate SGM parameters in raw scale (R)
theta_trans_fn = ThetaTransform(prior_bds=paras.prior_bds, k=paras_run.k)
folder_name = f"sgm_pairs_ntrain{paras_run.n_train}_ntest{paras_run.n_test}"
simu_sgm_data = edict()


# the function to generate PSD with SGM
def get_psd(cur_sgm):
    input_sgm = {}
    for ix, key in enumerate(paras.names):
        input_sgm[key] = cur_sgm[ix]
    cur_PSD = sgm.forward_psd(input_sgm)
    cur_PSD = cur_PSD[:68, :]
    return cur_PSD

def get_fc(cur_sgm):
    input_sgm = {}
    for ix, key in enumerate(paras.names):
        input_sgm[key] = cur_sgm[ix]
    cur_PSD = sgm.forward_fc(input_sgm)
    cur_PSD = cur_PSD[:68, :68]
    return cur_PSD

# generate the training and testing data
if (MIDRES_ROOT/f"{folder_name}/sgm_paramss.pkl").exists():
    sgm_paramss = load_pkl(MIDRES_ROOT/f"{folder_name}/sgm_paramss.pkl")
else:
    sgm_params_raw = np.random.randn(paras_run.n_train, 7)*paras_run.sd
    sgm_paramss = theta_trans_fn.raw2theta(sgm_params_raw)

simu_sgm_data.sgm_paramss = sgm_paramss

if not (MIDRES_ROOT/f"{folder_name}/PSDs.pkl").exists():
    with Parallel(n_jobs=n_jobs) as parallel:
        psds = parallel(delayed(get_psd)(cur_sgm) for cur_sgm in tqdm(sgm_paramss, 
                                                                      total=paras_run.n_train, 
                                                                      desc="Generating training PSDs"))
    PSDs = np.array(psds)
    simu_sgm_data.PSDs = PSDs

if not (MIDRES_ROOT/f"{folder_name}/FCs.pkl").exists():
    with Parallel(n_jobs=n_jobs) as parallel:
        fcs = parallel(delayed(get_fc)(cur_sgm) for cur_sgm in tqdm(sgm_paramss, 
                                                                      total=paras_run.n_train, 
                                                                      desc="Generating training FCs"))
    FCs = np.array(fcs)
    simu_sgm_data.FCs = FCs

if (MIDRES_ROOT/f"{folder_name}/sgm_paramss_test.pkl").exists():
    sgm_paramss_test = load_pkl(MIDRES_ROOT/f"{folder_name}/sgm_paramss_test.pkl")
else:
    sgm_params_test_raw = np.random.randn(paras_run.n_test, 7)*paras_run.sd
    sgm_paramss_test = theta_trans_fn.raw2theta(sgm_params_test_raw)
simu_sgm_data.sgm_paramss_test = sgm_paramss_test

if not (MIDRES_ROOT/f"{folder_name}/PSDs_test.pkl").exists():
    with Parallel(n_jobs=n_jobs) as parallel:
        psds = parallel(delayed(get_psd)(cur_sgm) for cur_sgm in tqdm(sgm_paramss_test, 
                                                                      total=paras_run.n_test, 
                                                                      desc="Generating testing PSDs"))
    PSDs_test = np.array(psds)
    simu_sgm_data.PSDs_test = PSDs_test

if not (MIDRES_ROOT/f"{folder_name}/FCs_test.pkl").exists():
    with Parallel(n_jobs=n_jobs) as parallel:
        fcs = parallel(delayed(get_fc)(cur_sgm) for cur_sgm in tqdm(sgm_paramss_test, 
                                                                      total=paras_run.n_test, 
                                                                      desc="Generating testing FCs"))
    FCs_test = np.array(fcs)
    simu_sgm_data.FCs_test = FCs_test
    


simu_sgm_data.freqs = paras.freqs
save_pkl_dict2folder(MIDRES_ROOT/folder_name, simu_sgm_data, is_force=True, verbose=True)
