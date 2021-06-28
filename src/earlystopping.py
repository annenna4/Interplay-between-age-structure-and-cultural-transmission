import numpy as np

import augmentation
import simulation
import utils


class EarlyStopping:
    def __init__(self, model: simulation.Simulator, verbose=False, **kwargs):
        self.model = model
        self.verbose = verbose
        self.log = []

    def __call__(self):
        return self._criterion()

    def _criterion(self) -> bool:
        raise NotImplementedError


class TurnoverEarlyStopping(EarlyStopping):
    def __init__(self, model: simulation.Simulator, verbose=False, **kwargs):
        super().__init__(model=model, verbose=verbose)

    def _criterion(self) -> bool:
        return self.model.population.min() >= self.model.initial_traits


class MaxIterEarlyStopping(EarlyStopping):
    def __init__(
        self, model: simulation.Simulator, warmup=10_000, verbose=False, **kwargs
    ):
        self.model = model
        self.warmup = warmup
        super().__init__(model=model, verbose=verbose)

    def _criterion(self) -> bool:
        self.warmup -= 1
        return self.warmup <= 0


class DiversityEarlyStopping(EarlyStopping):
    def __init__(
        self,
        model: simulation.Simulator,
        diversity_order=3.0,
        poll_interval=1,
        verbose=False,
        **kwargs,
    ):
        super().__init__(model=model, verbose=verbose)
        self.diversity_order = diversity_order
        self.poll_interval = poll_interval
        args = model.input_args
        args["initial_traits"] = int(model.n_agents / 10)
        self.alternative_model = simulation.Simulator(**args)

    def __call__(self):
        self.alternative_model.step()
        return (
            False
            if not self.model.timestep % self.poll_interval == 0
            else self._criterion()
        )

    def _criterion(self):
        Qa, Qb = self.diversity(self.model), self.diversity(self.alternative_model)
        if self.verbose:
            self.pp.update([[Qa, Qb]])
        self.log.append({"homogeneous": Qa, "heterogeneous": Qb})
        return Qa > Qb

    def diversity(self, model: simulation.Simulator):
        x = np.bincount(utils.reindex_array(model.population))
        p = x[x > 0] / x.sum()
        return augmentation.hill_number(x, self.diversity_order, p)


EARLYSTOPPERS = {
    "turnover": TurnoverEarlyStopping,
    "max_iter": MaxIterEarlyStopping,
    "diversity": DiversityEarlyStopping,
}
