# This file contains functions for checking whether 
# the SGM parameter is stable or not
# Refer to Verma_et_al_NNS_2022.pdf, formula (28) in appendix
import numpy as np


def obt_poly_coefs(theta):
    """The fn to obtain the polynomial coefs based on SGM parameters
            args: theta: parameters: num of sps x dim
                order: ['alpha', 'gei', 'gii', 'taue', 'tauG', 'taui', 'speed']
    """
    if len(theta.shape) == 1:
        theta = theta.reshape(1, -1)
    tes = 1/theta[:, 3]
    tis = 1/theta[:, 5]
    geis = theta[:, 1]
    giis = theta[:, 2]
    gee = 1.0
    # part 1
    p1 = np.array([
        gee*tes**3*tis**2,
        tis**2*tes**2+2*gee*tes**3*tis, 
        gee*tes**3+2*tes*tis**2+2*tis*tes**2, 
        tes**2+tis**2+4*tes*tis, 
        2*(tes+tis), 
        np.ones(len(tes)),
    ])
    
    p2 = np.array([
        giis*tis**3*tes**2,
        tis**2*tes**2+2*giis*tis**3*tes, 
        giis*tis**3+2*tes*tis**2+2*tis*tes**2, 
        tes**2+tis**2+4*tes*tis, 
        2*(tes+tis), 
        np.ones(len(tes)),
    ])
    
    p1p2 = np.array([
        np.ones(len(tes)),
        2*p1[4, :], 
        2*p1[3, :] + p1[4, :]**2, 
        p1[2, :] + p2[2, :] + 2*p1[3, :]*p1[4, :], 
        p1[1, :] + p2[1, :] + p1[4, :]*(p1[2, :]+p2[2, :]) + p1[3, :]**2, 
        p1[0, :] + p2[0, :] + p1[4, :]*(p1[1, :]+p2[1, :]) + p1[3, :]*(p1[2, :]+p2[2, :]), 
        p1[4, :]*(p1[0, :]+p2[0, :]) + p1[3, :]*(p1[1, :]+p2[1, :]) + p1[2, :]*p2[2, :],
        p1[3, :]*(p1[0, :]+p2[0, :]) + p1[2, :]*p2[1, :] + p1[1, :]*p2[2, :], 
        p1[2, :]*p2[0, :] + p2[2, :]*p1[0, :] + p1[1, :]*p2[1, :],
        p1[1, :]*p2[0, :]+p2[1, :]*p1[0, :],
        p1[0, :]*p2[0, :]
    ])
    
    coefs = p1p2
    coefs[-1, :] = coefs[-1, :] + geis**2*tes**5*tis**5
    return coefs.T


def paras_stable_check(theta):
    """The fn to obtain the polynomial coefs based on SGM parameters
            args: theta: parameters: num of sps x dim
                order: ['alpha', 'gei', 'gii', 'taue', 'tauG', 'taui', 'speed']
        return: a vec of 0 or 1.  0 if stable, 1 if not stable
    """
    if len(theta.shape) == 1:
        theta = theta.reshape(1, -1)
    coefs = obt_poly_coefs(theta)
    stb_idxs = []
    for ix in range(coefs.shape[0]):
        res = np.roots(coefs[ix])
        stb_idxs.append(int(np.sum(res.real >0) >0))
    stb_idxs = np.array(stb_idxs)
    return stb_idxs
