# this file contains fns for reparameterizing SGM
import numpy as np
import torch
from scipy.stats import norm as sci_normal
from torch.distributions.normal import Normal as tor_normal
from numbers import Number

def normcdf_np(x, sd=10):
    return sci_normal(loc=0, scale=sd).cdf(x)

def normcdf_torch(x, sd=10):
    return tor_normal(loc=0, scale=sd).cdf(x)

def logistic_torch(x, k=0.10):
    """k=0.1 fits prior N(0, 100)
    """
    if isinstance(x, Number):
        x = torch.tensor(float(x))
    num = torch.exp(k*x)
    den = torch.exp(k*x) + 1
    
    # fix inf issue
    res = num/den
    if torch.isinf(num).sum() > 0:
        res[torch.isinf(num)] = 1
    return res

def logit_torch(z, k):
    """ x = logit_torch(logistic_torch(x))
    """
    if isinstance(z, Number):
        z = torch.tensor(float(z))
    x = torch.logit(z, eps=1e-10)/k
    return x


def logistic_np(x, k=0.10):
    """k=0.1 fits prior N(0, 100)
    """
    num = np.exp(k*x)
    den = np.exp(k*x) + 1
    # fix inf issue
    res = num/den
    res[np.isinf(num)] = 1
    return res

def raw2theta_np(thetas_raw, prior_bds, k):
    """transform reparameterized theta to orignal theta
        args: thetas_raw: an array with num_sps x 7
              prior_bds: an array with 7 x 2
    """
    assert prior_bds.shape[0] == 7
    assert thetas_raw.shape[-1] == 7
    thetas = logistic_np(thetas_raw, k=k)*(prior_bds[:, 1] -  prior_bds[:, 0]) + prior_bds[:, 0]
    return thetas

def raw2theta_torch(thetas_raw, prior_bds, k):
    """transform reparameterized theta to orignal theta
        args: thetas_raw: an array with num_sps x 7
              prior_bds: an array with 7 x 2
    """
    assert prior_bds.shape[0] == 7
    assert thetas_raw.shape[-1] == 7
    thetas = logistic_torch(thetas_raw, k=k)*(prior_bds[:, 1] -  prior_bds[:, 0]) + prior_bds[:, 0]
    return thetas

def theta2raw_torch(thetas, prior_bds, k):
    """transform orignal theta to reparameterized theta 
        args: thetas_raw: an array with num_sps x 7
              prior_bds: an array with 7 x 2
    """
    assert prior_bds.shape[0] == 7
    assert thetas.shape[-1] == 7
    theta_raws = logit_torch((thetas-prior_bds[:, 0])/(prior_bds[:, 1]-prior_bds[:, 0]), k=k)
    return theta_raws
