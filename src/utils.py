import inspect
import itertools
import multiprocessing as mp
import numbers

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as dists
import tqdm


def get_arguments():
    """Returns tuple containing dictionary of calling function's
    named arguments and a list of calling function's unnamed
    positional arguments.
    """
    posname, kwname, args = inspect.getargvalues(inspect.stack()[1][0])[-3:]
    posargs = args.pop(posname, [])
    args.update(args.pop(kwname, []))
    if "self" in args:
        del args["self"]
    if "__class__" in args:
        del args["__class__"]
    return args


class Randint:
    def __init__(self, low, high):
        probs = torch.ones(high)
        probs[:low] = 0
        probs = probs / probs.sum()
        self.dist = dists.Categorical(probs)

    def sample(self, sample_shape=torch.Size()):
        return self.dist.sample(sample_shape)


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


class ParamSweep:
    def __init__(self, *params, num_samples=100):
        self._num_simulations = np.prod([len(p) for p in params]) * num_samples
        self._num_samples = num_samples
        self.params = params

    def __iter__(self):
        for _ in range(self._num_samples):
            for params in itertools.product(*self.params):
                yield np.array(params)

    def __len__(self):
        return self._num_simulations


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
    a_offs = a + np.arange(a.shape[0])[:, None] * N
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


def tsplot(
    y,
    x=None,
    n=50,
    percentile_min=1,
    percentile_max=99,
    color="C0",
    location="median",
    line_color="k",
    axis=0,
    ax=None,
    label=None,
    **kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()
    # calculate the lower and upper percentile groups, skipping 50 percentile
    perc1 = np.nanpercentile(
        y, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=axis
    )
    perc2 = np.nanpercentile(
        y, np.linspace(50, percentile_max, num=n + 1)[1:], axis=axis
    )

    if x is None:
        x = np.arange(y.shape[1])

    if "alpha" in kwargs:
        alpha = kwargs.pop("alpha")
    else:
        alpha = 1 / n
    # fill lower and upper percentile groups
    for p1, p2 in zip(perc1, perc2):
        ax.fill_between(x, p1, p2, alpha=alpha, color=color, edgecolor=None)

    if location == "mean":
        ax.plot(x, np.nanmean(y, axis=axis), color=line_color, label=label)
    elif location == "median":
        ax.plot(x, np.nanmedian(y, axis=axis), color=line_color, label=label)
    else:
        raise ValueError(f"Location `{location}` is not supported.")

    return ax
