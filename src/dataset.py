from typing import NamedTuple, Dict, Iterable, Tuple

import numpy as np
import torch

from torch.utils.data import Dataset, IterableDataset


class PresimulatedDataset(Dataset):
    def __init__(self, theta, samples):
        self.dataset = torch.FloatTensor(samples)
        self.theta = torch.FloatTensor(theta)

    @classmethod
    def load(cls, fname, summary_statistics=False, random_sample=None):
        theta = np.load(f"{fname}.theta.npy")
        if summary_statistics:
            samples = np.load(f"{fname}.features.npy")
        else:
            samples = np.load(f"{fname}.samples.npy")
        if random_sample is not None:
            if isinstance(random_sample, float):
                random_sample = int(random_sample * samples.shape[0])
            indices = np.random.randint(0, samples.shape[0], size=random_sample)
            theta, samples = theta[indices], samples[indices]
        return PresimulatedDataset(theta, samples)

    def __getitem__(self, i):
        theta = self.theta[i]
        if theta.dim() == 0:
            theta = theta.view(-1)
        return theta, self.dataset[i]

    def __len__(self):
        return len(self.dataset)


class SimulationDataset(Dataset):
    
    def __init__(self, simulator, prior, num_simulations):
        
        self.simulator = simulator
        self.theta = prior.sample([num_simulations])
        self.theta[np.random.choice(num_simulations, num_simulations // 2, replace=False)] = 0.0
        self.num_simulations = num_simulations
    
    def __getitem__(self, i):
        return self.theta[i], self.simulator(self.theta[i])

    def __len__(self):
        return self.num_simulations


class LazySimulationDataset(IterableDataset):
    def __init__(self, simulator, prior, num_simulations):
        
        self.simulator = simulator
        self.prior = prior
        self.num_simulations = num_simulations

    def __iter__(self):
        biased = np.random.random(self.num_simulations) < 0.5
        for i in range(self.num_simulations):
            theta = 0.0 if not biased[i] else self.prior.sample()
            yield theta, self.simulator(theta)


def lazy_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.num_simulations = dataset.num_simulations // worker_info.num_workers
