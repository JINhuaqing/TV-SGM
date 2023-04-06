# +
# A collection of the standardization functions
# -

import numpy as np


# +
def std_mat(mat, nroi=None):
    """
    Standardize a 2-D array by subtracting the mean and dividing by the standard deviation along the second axis.

    Args:
        mat (numpy.ndarray): A 2-D array of shape (num_rois, len_seq).
        nroi (int, optional): The number of ROIs. Defaults to None.

    Returns:
        numpy.ndarray: The standardized 2-D array.
    """ 
    assert mat.ndim == 2, "Only 2-D array is permitted."
    if nroi is not None:
        assert mat.shape[0] == nroi, "The first dim should be num of ROIs."
    mat_std = (mat-mat.mean(axis=1, keepdims=1))/mat.std(axis=1, keepdims=1)
    return mat_std


# -

def std_vec(vec):
    """
    Standardize a 1-D array by subtracting the mean and dividing by the standard deviation.

    Args:
        vec (numpy.ndarray): A 1-D array.

    Returns:
        numpy.ndarray: The standardized 1-D array.
    """
    return (vec-np.mean(vec))/np.std(vec)


