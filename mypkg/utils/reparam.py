# this file contains fns for reparameterizing SGM
import numpy as np
import torch
from scipy.stats import norm as sci_normal
from torch.distributions.normal import Normal as tor_normal
from numbers import Number
from constants import SGM_prior_bds

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
    if isinstance(x, Number):
        x = np.array([x], dtype=np.float64)
    num = np.exp(k*x)
    den = np.exp(k*x) + 1
    # fix inf issue
    res = num/den
    res[np.isinf(num)] = 1
    return res

def logit_np(z, k=0.1):
    if isinstance(z, Number):
        z = np.array([z], dtype=np.float64)
    eps = 1e-10
    z = np.clip(z, eps, 1-eps) 
    return np.log(z/(1-z))/k

# DEP To be deprecated
def raw2theta_np(thetas_raw, prior_bds, k):
    """transform reparameterized theta to orignal theta
        args: thetas_raw: an array with num_sps x *
              prior_bds: an array with * x 2
    """
    thetas = logistic_np(thetas_raw, k=k)*(prior_bds[:, 1] -  prior_bds[:, 0]) + prior_bds[:, 0]
    return thetas

# DEP To be deprecated
def raw2theta_torch(thetas_raw, prior_bds, k):
    """transform reparameterized theta to orignal theta
        args: thetas_raw: an array with num_sps x *
              prior_bds: an array with * x 2
    """
    thetas = logistic_torch(thetas_raw, k=k)*(prior_bds[:, 1] -  prior_bds[:, 0]) + prior_bds[:, 0]
    return thetas

# DEP To be deprecated
def theta2raw_torch(thetas, prior_bds, k):
    """transform orignal theta to reparameterized theta 
        args: thetas_raw: an array with num_sps x *
              prior_bds: an array with * x 2
    """
    theta_raws = logit_torch((thetas-prior_bds[:, 0])/(prior_bds[:, 1]-prior_bds[:, 0]), k=k)
    return theta_raws


class ThetaTransform:
    """
    Transform between reparameterized and original theta
    theta is the SGM parameters
    """
    def __init__(self, 
                 params_mask=None,
                 prior_bds=None, 
                 k=0.15):
        """
        by default, the order of parameters is alphabetical order
        args: 
            params_mask: which parameters to be transformed
            prior_bds: an array with * x 2
            k: the steepness of the logistic fn
        """
        if params_mask is None:
            params_mask = [True]*len(SGM_prior_bds)
        if prior_bds is None:
            keys = ["alpha", "gei", "gii", "taue", "taug", "taui", "speed"]
            prior_bds = np.array([SGM_prior_bds[key] for key in keys])
        
        self.prior_bds = np.array([prior_bds[i] for i in range(len(prior_bds)) if params_mask[i]])
        self.k = k

    def theta2raw(self, thetas):
        """
        transform orignal theta to reparameterized theta
        args: thetas_raw: an array/torch.Tensor with num_sps x *
        """
        prior_bds = self.prior_bds
        k = self.k
        if isinstance(thetas, torch.Tensor):
            prior_bds = torch.tensor(prior_bds)
            theta_raws = logit_torch((thetas-prior_bds[:, 0])/(prior_bds[:, 1]-prior_bds[:, 0]), k=k)
        else:
            theta_raws = logit_np((thetas-prior_bds[:, 0])/(prior_bds[:, 1]-prior_bds[:, 0]), k=k)
        return theta_raws
    
    def raw2theta(self, thetas_raw):
        """
        transform reparameterized theta to orignal theta
        args: thetas_raw: an array/torch.Tensor with num_sps x *
        """
        prior_bds = self.prior_bds
        k = self.k
        if isinstance(thetas_raw, torch.Tensor):
            prior_bds = torch.tensor(prior_bds)
            thetas = logistic_torch(thetas_raw, k=k)*(prior_bds[:, 1] -  prior_bds[:, 0]) + prior_bds[:, 0]
        else:
            thetas = logistic_np(thetas_raw, k=k)*(prior_bds[:, 1] -  prior_bds[:, 0]) + prior_bds[:, 0]
        return thetas
