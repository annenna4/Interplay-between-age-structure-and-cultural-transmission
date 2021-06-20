import logging
import numpy as np

import torch
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, Uniform

import utils


class Compose:
    def __init__(self, *transformers):
        self.transformers = transformers

    def __call__(self, x):
        for transformer in self.transformers:
            x = transformer(x)
        return x

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for transformer in self.transformers:
            format_string += '\n'
            format_string += '    {0}'.format(transformer)
        format_string += '\n)'
        return format_string


class TimeseriesTransformer:
    def __init__(self):
        self.logger = logging.getLogger(f"{self}")
        self.logger.setLevel(logging.INFO)

    def __call__(self, x):
        self.logger.info("transforming series")
        x = utils.check_array(x)
        return self.transform(x)

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def transform(self, x):
        raise NotImplementedError


class Sampler(TimeseriesTransformer):
    def __init__(self, n, random_state: int = None):
        self.n = n
        self.rng = utils.check_random_state(random_state)
        super().__init__()

    def transform(self, x):
        return np.array([
            np.bincount(self.rng.choice(v.shape[0], self.n, p=v / v.sum()), minlength=x.shape[1])
            for v in x
        ])

    
class Rescaler(TimeseriesTransformer):
    def __init__(self, timesteps: int = None):
        self.timesteps = timesteps
        super().__init__()

    def transform(self, x):
        return np.vstack([self._scale(v) for v in x])

    def _scale(self, x):
        return np.interp(
            np.linspace(0, 1, self.timesteps), np.linspace(0, 1, len(x)), x
        )


class Evenness(TimeseriesTransformer):
    def transform(self, x):
        return np.vstack([self._evenness_score(v) for v in x])

    def _evenness_score(self, x):
        return (x - 1) / (x[0] - 1)


class Cutter(TimeseriesTransformer):
    def __init__(self, prior: Distribution, random_state: int = None):
        if isinstance(prior, float):
            prior = Uniform(prior, prior)
        self.prior = prior
        self.rng = utils.check_random_state(random_state)
        super().__init__()

    def transform(self, x):
        keep = ~self.rng.binomial(1, p=self.prior.sample(), size=x.shape[1]).astype(bool)
        return x[:, keep]


class Randomizer(TimeseriesTransformer):
    def __init__(self, sigma: float = 0.1, random_state: int = None):
        self.sigma = sigma
        self.rng = utils.check_random_state(random_state)
        super().__init__()

    def transform(self, x):
        noise = self.rng.normal(loc=0.0, scale=self.sigma, size=x.shape)
        return np.clip(x - noise, 0, 1)


class Normalizer(TimeseriesTransformer):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def transform(self, x):
        x = x + self.eps
        return x / x.sum(0)

    
class Standardizer(TimeseriesTransformer):
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        super().__init__()

    def transform(self, x):
        return (x - np.mean(x, axis=1, keepdims=True)) / (
            np.std(x, axis=1, keepdims=True) + self.eps
        )


class BinningPrior(Categorical):
    def __init__(self, timesteps: int, min_bins: int = 10):
        self.bins = np.arange(min_bins, timesteps + 1)
        probs = torch.tensor([1 / len(self.bins)] * len(self.bins))
        super().__init__(probs, None, None)

    def sample(self, sample_shape=torch.Size([])):
        samples = super().sample(sample_shape)
        return self.bins[samples]


class Binner(TimeseriesTransformer):
    def __init__(self, prior: BinningPrior, n_agents: int):
        self.prior = prior
        self.n_agents = n_agents
        super().__init__()

    def transform(self, x):
        pop_size = np.full(x.shape[1], self.n_agents)
        bins = np.array_split(np.arange(x.shape[1]), self.prior.sample())
        return np.array([x[:, b].sum(1) / pop_size[b].sum() for b in bins]).T


class Trimmer(TimeseriesTransformer):
    def __init__(self, fraction: Distribution, random_state: int = None):
        if isinstance(fraction, float):
            fraction = Uniform(fraction, fraction)
        self.fraction = fraction
        self.rng = utils.check_random_state(random_state)
        super().__init__()

    def transform(self, x):
        window = int((1 - self.fraction.sample()) * x.shape[1])
        start = self.rng.randint(0, x.shape[1] - window)
        end = start + window
        return x[:, start:end]


class Padder(TimeseriesTransformer):
    def __init__(self, timesteps: int):
        self.timesteps = timesteps
        super().__init__()

    def transform(self, x):
        x = torch.FloatTensor(x)
        return F.pad(x, (0, self.timesteps - x.shape[1]))


def hill_number(x, q, p):
    return (
        (x > 0).sum()
        if q == 0
        else np.exp(-np.sum(p * np.log(p)))
        if q == 1
        else np.exp(1 / (1 - q) * np.log(np.sum(p ** q)))
    )


class HillNumbers(TimeseriesTransformer):
    def __init__(self, min_q=0, max_q=3, q_step=1):
        self.q = np.arange(min_q, max_q + q_step, step=q_step)
        super().__init__()

    def transform(self, X):
        return np.array([self._transform(x) for x in X])

    def _transform(self, x):
        p = x[x > 0] / x.sum()
        return np.array([self._hn(x, q, p) for q in self.q])

    def _hn(self, x, q, p):
        return hill_number(x, q, p)