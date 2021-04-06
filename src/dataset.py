from typing import NamedTuple, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import Dataset, IterableDataset


class PresimulatedDataset(Dataset):
    def __init__(self, ids, theta, samples):
        self.dataset = torch.FloatTensor(samples)
        self.theta = torch.FloatTensor(theta)
        self.ids = np.array(ids)

    @classmethod
    def load(cls, fname, transform=None):
        data = np.load(fname)
        theta, samples, ids = data["theta"], data["samples"], data["ids"]
        samples = samples.astype(np.float64)
        if transform is not None:
            for id in np.unique(ids):
                idx = ids == id
                samples[idx] = transform(samples[idx])
        return PresimulatedDataset(ids, theta, samples)

    def __getitem__(self, i):
        theta = self.theta[i]
        if theta.dim() == 0:
            theta = theta.view(-1)
        return theta, self.dataset[i]

    def __len__(self):
        return len(self.dataset)

    def train_test_split(self, test_size=0.1):
        ids = np.unique(self.ids)
        train_ids, test_ids = train_test_split(ids, test_size=test_size, shuffle=True)
        train_ids, test_ids = np.isin(self.ids, train_ids), np.isin(self.ids, test_ids)
        return (
            PresimulatedDataset(self.ids[train_ids], self.theta[train_ids], self.dataset[train_ids]),
            PresimulatedDataset(self.ids[test_ids], self.theta[test_ids], self.dataset[test_ids])
        )


class SimulationDataset(Dataset):
    
    def __init__(self, simulator, prior, num_simulations, transform=None):
        
        self.simulator = simulator
        self.theta = prior.sample([num_simulations])
        for index in np.random.choice(num_simulations, num_simulations // 2, replace=False):
            self.theta[index, 0] = 0.0
        self.num_simulations = num_simulations
        self.transform = transform
    
    def __getitem__(self, i):
        theta, sample = self.theta[i], self.simulator(self.theta[i])
        if self.transform is not None:
            sample = self.transform(sample)
        return theta, sample

    def __len__(self):
        return self.num_simulations


class LazySimulationDataset(IterableDataset):
    def __init__(self, simulator, prior, num_simulations, transform=None):
        
        self.simulator = simulator
        self.prior = prior
        self.num_simulations = num_simulations
        self.transform = transform

    def __iter__(self):
        biased = np.random.random(self.num_simulations) < 0.5
        for i in range(self.num_simulations):
            theta = self.prior.sample()
            if not biased[i]:
                theta[0] = 0.0
            sample = self.simulator(theta)
            if self.transform is not None:
                sample = self.transform(sample)            
            yield theta, sample

    def __len__(self):
        return self.num_simulations


def lazy_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.num_simulations = len(dataset) // worker_info.num_workers


def collate_timeseries(series):
    theta, series = zip(*series)
    length = max(len(seq) for seq in series)
    series = [F.pad(torch.FloatTensor(seq), (0, length - len(seq))) for seq in series]
    return torch.stack(theta), torch.stack(series)
