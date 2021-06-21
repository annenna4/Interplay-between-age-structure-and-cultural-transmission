import collections
import math
import json

import numpy as np
import torch
import tqdm

import augmentation
import earlystopping
import utils


class Simulator:
    def __init__(
        self,
        n_agents: int = 100_000,
        beta: float = 0.0,
        mu: float = 0.0005,
        p_death: float = 0.1,
        eta: int = 10,
        earlystopper: str = "diversity",
        initial_traits: int = 2,
        random_state: int = None,
        disable_pbar: bool = False,
        diversity_order: float = 3.0,
        warmup_iterations: int = 10_000,
        poll_interval: int = 1,
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
        self.poll_interval = poll_interval
        self.q_step = q_step

        self._input_args = utils.get_arguments()

        # Initialize the population with n traits equally distributed over the agents
        self.population = self.rng.choice(self.initial_traits, size=self.n_agents)
        # Randomly associate birth dates with each of the traits
        self.birth_date = np.ceil(self.rng.random(self.n_agents) * 50).astype(np.int64)
        # Compute the number of unique traits
        self.n_traits = len(np.unique(self.population))
        self.timestep = self.birth_date.max()

    def fit(self):
        self.earlystop = earlystopping.EARLYSTOPPERS[self.earlystopper](
            self,
            warmup=self.warmup_iterations,
            diversity_order=self.diversity_order,
            poll_interval=self.poll_interval,
            verbose=False,
        )

        with tqdm.tqdm(desc="Burn-in period", disable=self.disable_pbar) as pbar:
            while not self.earlystop():
                self.step()
                pbar.update()
        return self

    def sample(self, timesteps=1):
        sample = np.zeros((timesteps, self.n_agents), dtype=np.int64)
        self.population = utils.reindex_array(self.population)
        self.n_traits = len(np.unique(self.population))
        self.birth_date = self.birth_date - (self.birth_date.min() - 1)
        init = self.birth_date.max() + 1
        for timestep in tqdm.trange(
            init,
            timesteps + init,
            desc="Generating populations",
            disable=self.disable_pbar,
        ):
            self.step()
            sample[timestep - init] = self.population

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

        innovators = np.where(novel)[0][self.rng.random(novel.sum()) < self.mu]
        n_innovations = len(innovators)
        self.population[innovators] = np.arange(
            self.n_traits, self.n_traits + n_innovations
        )
        self.n_traits = self.n_traits + n_innovations
        self.birth_date[novel] = self.timestep  # + 1
        self.timestep += 1

    def save(self, filepath: str):
        if not filepath.endswith(".json"):
            filepath += ".json"
        with open(filepath, "w") as fp:
            json.dump(
                {
                    "population": self.population.list(),
                    "n_traits": self.n_traits,
                    "birth_date": self.birth_date.list(),
                    "params": self.input_args,
                    "burnin-step": self.timestep,
                },
                fp,
            )

    @property
    def input_args(self):
        return {k: v for k, v in self._input_args.items()}
