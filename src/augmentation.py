import logging
import numpy as np

import utils


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class Normalizer(TimeseriesTransformer):
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        super().__init__()

    def transform(self, x):
        return (x - np.mean(x, axis=1, keepdims=True)) / (
            np.std(x, axis=1, keepdims=True) + self.eps
        )
