import json
import operator

import numpy as np
import tqdm

import augmentation
import utils


class Simulator:
    def __init__(
        self,
        n_agents: int = 100_000,
        beta: float = 0.0,
        mu: float = 0.0005,
        p_death: float = 0.1,
        eta: int = 2,
        earlystopper: str = "diversity",
        initial_traits: int = 2,
        random_state: int = None,
        disable_pbar: bool = False,
        diversity_order: float = 5.0,
        warmup_iterations: int = 10_000,
        minimum_timesteps: int = 1000,
        poll_interval: int = 100,
        q_step: float = 0.25,
    ):
        self.n_agents = n_agents
        self.beta = beta
        self.mu = mu
        self.p_death = p_death
        self.eta = eta
        self.initial_traits = initial_traits
        self.rng = utils.check_random_state(random_state)
        self.disable_pbar = disable_pbar
        self.earlystopper = earlystopper
        self.diversity_order = diversity_order
        self.warmup_iterations = warmup_iterations
        self.minimum_timesteps = minimum_timesteps
        self.poll_interval = poll_interval
        self.q_step = q_step

        self._input_args = utils.get_arguments()

        # Initialize the population with n traits equally distributed over the agents
        self.population = self.rng.choice(self.initial_traits, size=self.n_agents)
        # Randomly associate birth dates with each of the traits
        self.birth_date = np.ceil(self.rng.random(self.n_agents) * 50).astype(np.int64)
        # Compute the number of unique traits
        self.n_traits = len(np.unique(self.population))
        self.timestep = self.birth_date.max() + 1

    def fit(self):
        self.earlystop = EARLYSTOPPERS[self.earlystopper](
            self,
            warmup=self.warmup_iterations,
            diversity_order=self.diversity_order,
            minimum_timesteps=self.minimum_timesteps,
            poll_interval=self.poll_interval,
            verbose=False,
        )

        with tqdm.tqdm(desc="Burn-in period", disable=self.disable_pbar) as pbar:
            while not self.earlystop():
                self.step()
                pbar.update()
            for _ in range(self.warmup_iterations):
                self.step()
                pbar.update()
        return self

    def sample(self, timesteps=1):
        sample = np.zeros((timesteps, self.n_agents), dtype=np.int64)
        self.population = utils.reindex_array(self.population)
        self.n_traits = len(np.unique(self.population))
        for timestep in tqdm.trange(
            timesteps,
            desc="Generating populations",
            disable=self.disable_pbar,
        ):
            self.step()
            sample[timestep] = self.population

        sample = utils.bincount2d(sample)
        return sample

    def step(self):
        novel = self.rng.binomial(1, self.p_death, self.n_agents).astype(bool)
        copy_pool = np.arange(self.n_agents)
        # Limit the copy pool if age window is specified
        if self.p_death < 1.0:
            age_low, age_high = 0, self.eta
            parents, offset = np.array([]), 0
            while parents.sum() == 0:
                parents = (self.birth_date < (self.timestep - (age_low - offset))) & (
                    self.birth_date > (self.timestep - (age_high + offset))
                )
                offset += 1
            copy_pool = copy_pool[parents]
        traits, counts = np.unique(self.population[copy_pool], return_counts=True)
        counts = counts ** (1 + self.beta)
        self.population[novel] = self.rng.choice(
            traits, novel.sum(), p=counts / counts.sum()
        )
        innovators = np.flatnonzero(novel)[self.rng.binomial(1, self.mu, novel.sum()).astype(bool)]
        n_innovations = len(innovators)
        assert self.population.max() < self.n_traits
        self.population[innovators] = np.arange(
            self.n_traits, self.n_traits + n_innovations
        )
        self.n_traits = self.n_traits + n_innovations
        assert self.timestep > self.birth_date.max()
        self.birth_date[novel] = self.timestep
        self.timestep += 1

    def to_dict(self):
        return (
            {
                "population": self.population.list(),
                "n_traits": self.n_traits,
                "birth_date": self.birth_date.list(),
                "params": self.input_args,
                "burnin-step": self.timestep,
            },
        )

    def save(self, filepath: str):
        if not filepath.endswith(".json"):
            filepath += ".json"
        with open(filepath, "w") as fp:
            json.dump(self.to_dict(), fp)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath) as fp:
            data = json.load(fp)
        model = Simulator(**data["params"])
        model.population = np.array(data["population"])
        model.n_traits = data["n_traits"]
        model.birth_date = data["birth_date"]
        model.timestep = data["timestep"]
        return model

    @property
    def input_args(self):
        return {k: v for k, v in self._input_args.items()}


class EarlyStopping:
    def __init__(self, model: Simulator, verbose=False, **kwargs):
        self.model = model
        self.verbose = verbose
        self.log = []

    def __call__(self):
        return self._criterion()

    def _criterion(self) -> bool:
        raise NotImplementedError


class TurnoverEarlyStopping(EarlyStopping):
    def __init__(self, model: Simulator, verbose=False, **kwargs):
        super().__init__(model=model, verbose=verbose)

    def _criterion(self) -> bool:
        return self.model.population.min() >= self.model.initial_traits


class MaxIterEarlyStopping(EarlyStopping):
    def __init__(self, model: Simulator, warmup=10_000, verbose=False, **kwargs):
        self.model = model
        self.warmup = warmup
        super().__init__(model=model, verbose=verbose)

    def _criterion(self) -> bool:
        self.warmup -= 1
        return self.warmup <= 0


class DiversityEarlyStopping(EarlyStopping):
    def __init__(
        self,
        model: Simulator,
        diversity_order=3.0,
        poll_interval=1,
        minimum_timesteps=1000,
        verbose=False,
        **kwargs,
    ):
        super().__init__(model=model, verbose=verbose)
        self.diversity_order = diversity_order
        self.minimum_timesteps = minimum_timesteps
        self.poll_interval = poll_interval
        self.crossings = 0
        self.patience = 3
        args = model.input_args
        args["initial_traits"] = int(model.n_agents / 10)
        self.alternative_model = Simulator(**args)

    def __call__(self):
        self.alternative_model.step()
        return (
            False
            if not self.model.timestep % self.poll_interval == 0
            else self._criterion()
        )

    def _criterion(self):
        compare = (operator.lt, operator.gt)[int(self.crossings % 2 == 0)]
        Qa, Qb = self.diversity(self.model), self.diversity(self.alternative_model)
        self.log.append({"homogeneous": Qa, "heterogeneous": Qb})
        if self.model.timestep > self.minimum_timesteps:
            self.crossings += int(compare(Qa, Qb))
        return self.crossings == self.patience

    def diversity(self, model: Simulator):
        x = np.bincount(utils.reindex_array(model.population))
        p = x[x > 0] / x.sum()
        return augmentation.hill_number(x, self.diversity_order, p)


EARLYSTOPPERS = {
    "turnover": TurnoverEarlyStopping,
    "max_iter": MaxIterEarlyStopping,
    "diversity": DiversityEarlyStopping,
}
