import collections
import math

import numpy as np
import torch
import tqdm

from augmentation import TimeseriesTransformer, HillNumbers
from utils import reindex_array, check_random_state, bincount2d


class Simulator:
    def __init__(
        self,
        n_agents,
        timesteps=10_000,
        warmup=1000,
        restricted_age_window=False,
        initial_traits=2,
        top_n=0,
        random_state=None,
        disable_pbar=False,
        q_step=0.25,
        summarize=False,
    ):
        self.n_agents = n_agents
        self.timesteps = timesteps
        self.warmup = warmup
        self.restricted_age_window = restricted_age_window
        self.initial_traits = initial_traits
        self.top_n = top_n
        self.rng = check_random_state(random_state)
        self.disable_pbar = disable_pbar
        self.q_step = q_step
        self.summarize = summarize

    def __call__(self, theta, random_state=None):
        if random_state is not None:
            self.rng = check_random_state(random_state)
        if isinstance(theta, torch.Tensor):
            theta = theta.numpy()
        beta, mu, p_death, eta = theta
        # Initialize the population with n traits equally distributed over the agents
        population = self.rng.choice(self.initial_traits, size=self.n_agents)
        # Randomly associate birth dates with each of the traits
        birth_date = np.ceil(self.rng.random(self.n_agents) * 50).astype(np.int64)

        # We start with a warming-up phase, in which we run the model until all
        # initial traits have gone extinct
        n_traits = len(np.unique(population))
        # with tqdm.tqdm(desc="Burn-in period", disable=self.disable_pbar) as pbar:
        #     while population.min() < self.initial_traits:
        #         population, birth_date, n_traits, novel = self._get_dynamics(
        #             beta, mu, p_death, population, birth_date, n_traits
        #         )
        #         pbar.update()
        init = birth_date.max()
        for timestep in tqdm.trange(
                init, self.warmup + init, desc="Warming up", disable=self.disable_pbar
        ):
            population, birth_date, n_traits, novel = self._get_dynamics(
                timestep, beta, mu, p_death, eta, population, birth_date, n_traits
            )

        # Following the burn-in period, we sample n populations.
        sample = np.zeros((self.timesteps, self.n_agents), dtype=np.int64)
        population = reindex_array(population)
        n_traits = len(np.unique(population))
        birth_date = birth_date - birth_date.min()
        init = birth_date.max()
        for timestep in tqdm.trange(
                init, self.timesteps + init, desc="Generating populations", disable=self.disable_pbar
        ):
            population, birth_date, n_traits, novel = self._get_dynamics(
                timestep, beta, mu, p_death, eta, population, birth_date, n_traits
            )
            sample[timestep - init] = population

        sample = bincount2d(sample)

        if self.top_n > 0:
            sample = sample[:, sample.sum(0).argsort()[-self.top_n :]]
        sample = sample.T

        if self.summarize:
            sample = HillNumbers(q_step=self.q_step)(sample)
        return sample

    def _get_dynamics(self, timestep, beta, mu, p_death, eta, population, birth_date, n_traits):
        novel = self.rng.binomial(1, p_death, self.n_agents).astype(bool)
        # novel = self.rng.random(self.n_agents) < p_death

        copy_pool = np.arange(self.n_agents)
        # Limit the copy pool if age window is specified
        if self.restricted_age_window:
            age_low, age_high = 0, int(math.ceil(eta * birth_date.max()))  # self.age_window
            parents, offset = np.array([]), 0
            while parents.sum() == 0:
                parents = ((birth_date < (timestep - (age_low - offset)))
                    & (birth_date > (timestep - (age_high + offset))))
                offset += 1
            copy_pool = copy_pool[parents]
        traits, counts = np.unique(population[copy_pool], return_counts=True)
        counts = counts ** (1 + beta)
        population[novel] = self.rng.choice(
            traits, novel.sum(), p=counts / counts.sum()
        )

        innovators = (self.rng.random(self.n_agents) < mu) & novel
        n_innovations = innovators.sum()
        population[innovators] = np.arange(n_traits, n_traits + n_innovations)
        birth_date[novel] = timestep # + 1
        return population, birth_date, n_traits + n_innovations, novel
    
