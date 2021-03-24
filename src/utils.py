import numpy as np


def reindex_array(x):
    _, x = np.unique(x, return_inverse=True)
    return x


def check_random_state(seed):
    if seed is np.random:
        return np.random.mtrand._rand
    if seed is None:
        return np.random.RandomState()
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
    )


def check_array(x):
    x = np.asarray(x)
    return np.array([x]) if x.ndim == 1 else x
