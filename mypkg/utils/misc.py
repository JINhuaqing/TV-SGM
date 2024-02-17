import numpy as np
import numpy.random as npr
import pickle
from easydict import EasyDict as edict
import time
import logging


def _set_verbose_level(verbose, logger):
    if verbose == 0:
        verbose_lv = logging.ERROR
    elif verbose == 1:
        verbose_lv = logging.WARNING
    elif verbose == 2:
        verbose_lv = logging.INFO
    elif verbose == 3:
        verbose_lv = logging.DEBUG
    if len(logger.handlers)>0:
        logger.handlers[0].setLevel(verbose_lv)
    else:
        logger.setLevel(verbose_lv)

def _update_params(input_params, def_params, logger):
    for ky, v in input_params.items():
        if ky not in def_params.keys():
            logger.warning(f"Check your input, {ky} is not used.")
        else:
            if v is not None:
                def_params[ky] = v
    return edict(def_params)

def get_cpt_ts(mean_diff, cpts, err_sd, length):
    """
    Generate the time series with change points.

    Parameters:
    mean_diff (list): List of mean differences for each segment.
    cpts (list): List of change points.
                 from 0, cpt is the first point of the segment 
                 cpt should beween (0, length)
    err_sd (float): Standard deviation of the error term.
    length (int): Length of the time series.

    Returns:
    numpy.ndarray: Time series with change points.
    """
    if not isinstance(cpts, list):
        cpts = list(cpts)
    if not isinstance(mean_diff, list):
        mean_diff = list(mean_diff)
    cpts0 = np.array([0] + cpts)
    cpts1 = np.array(cpts + [length])
    
    means = np.cumsum([0]+mean_diff)
    seg_lens = cpts1 - cpts0
    seqs = []
    for m, seg_len in zip(means, seg_lens):
        seqs.append([m]*seg_len)
    seqs = np.concatenate(seqs)    
    
    ts = seqs+npr.randn(length)*err_sd
    return ts

def mag2db(y):
    """Convert magnitude response to decibels for a simple array.
    Args:
        y (numpy array): Power spectrum, raw magnitude response. no squared
    Returns:
        dby (numpy array): Power spectrum in dB
    """
    dby = 20 * np.log10(y)
    return dby

def delta_time(t):
    """Return the time diff from t
    """
    delta_t = time.time() - t
    return delta_t


def load_pkl_folder2dict(folder, excluding=[], including=["*"], verbose=True):
    """The function is to load pkl file in folder as an edict
        args:
            folder: the target folder
            excluding: The files excluded from loading
            including: The files included for loading
            Note that excluding override including
    """
    if not isinstance(including, list):
        including = [including]
    if not isinstance(excluding, list):
        excluding = [excluding]
        
    if len(including) == 0:
        inc_fs = []
    else:
        inc_fs = list(set(np.concatenate([list(folder.glob(nam+".pkl")) for nam in including])))
    if len(excluding) == 0:
        exc_fs = []
    else:
        exc_fs = list(set(np.concatenate([list(folder.glob(nam+".pkl")) for nam in excluding])))
    load_fs = np.setdiff1d(inc_fs, exc_fs)
    res = edict()
    for fil in load_fs:
        res[fil.stem] = load_pkl(fil, verbose)                                                                                                                                  
    return res

# save a dict into a folder
def save_pkl_dict2folder(folder, res, is_force=False, verbose=True):
    assert isinstance(res, dict)
    for ky, v in res.items():
        save_pkl(folder/f"{ky}.pkl", v, is_force=is_force, verbose=verbose)

# load file from pkl
def load_pkl(fil, verbose=True):
    if verbose:
        print(f"Load file {fil}")
    with open(fil, "rb") as f:
        result = pickle.load(f)
    return result

# save file to pkl
def save_pkl(fil, result, is_force=False, verbose=True):
    if not fil.parent.exists():
        fil.parent.mkdir()
        if verbose:
            print(fil.parent)
            print(f"Create a folder {fil.parent}")
    if is_force or (not fil.exists()):
        if verbose:
            print(f"Save to {fil}")
        with open(fil, "wb") as f:
            pickle.dump(result, f)
    else:
        if verbose:
            print(f"{fil} exists! Use is_force=True to save it anyway")
        else:
            pass

