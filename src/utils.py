import multiprocessing as mp
import numpy as np
import torch
import tqdm


class Constant:
    def __init__(self, value):
        self._value = value

    def sample(self, sample_shape=torch.Size()):
        return torch.ones(sample_shape) * self._value


class IndependentPriors:
    def __init__(self, *distributions):
        self.distributions = distributions

    def sample(self, sample_shape=torch.Size()):
        return torch.stack(
            [d.sample(sample_shape) for d in self.distributions],
            int(len(sample_shape) > 0),
        )


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


def bincount2d(a):
    N = a.max() + 1
    a_offs = a + np.arange(a.shape[0])[:,None] * N
    return np.bincount(a_offs.flatten(), minlength=a.shape[0] * N).reshape(-1, N)


class Parallel:
    def __init__(self, n_workers, n_tasks):
        self.pool = mp.Pool(n_workers)
        self._results = []
        self._pb = tqdm.tqdm(total=n_tasks)

    def apply_async(self, fn, args=None):
        self.pool.apply_async(fn, args=args, callback=self._completed)

    def _completed(self, result):
        self._results.append(result)
        self._pb.update()

    def join(self):
        self.pool.close()
        self.pool.join()

    def result(self):
        self._pb.close()
        self.pool.close()
        return self._results
