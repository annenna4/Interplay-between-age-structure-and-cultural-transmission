import numpy as np
import scipy.sparse as sp


def reindex_array(x):
    _, x = np.unique(x, return_inverse=True)
    return x
