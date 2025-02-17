import numpy as np
from pathlib import Path

# some global paths
_cur_dir = Path(__file__).parent
RES_ROOT = Path(_cur_dir/"../results")
DATA_ROOT = Path(_cur_dir/"../data")
FIG_ROOT = Path(_cur_dir/"../figs")
MIDRES_ROOT = Path(_cur_dir/"../mid_results")


SGM_prior_bds = {}
SGM_prior_bds["alpha"] = [0.01, 1]
SGM_prior_bds["gei"] = [0.001, 0.7]
SGM_prior_bds["gii"] = [0.001, 2]
SGM_prior_bds["taue"] = [0.005, 0.03]
SGM_prior_bds["taug"] = [0.005, 0.03]
SGM_prior_bds["taui"] = [0.005, 0.20]
SGM_prior_bds["speed"] = [5, 20]