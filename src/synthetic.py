import argparse
import json
import multiprocessing as mp
import os

from datetime import datetime

import numpy as np
import torch

from augmentation import Normalizer
from simulation import Simulator
from utils import Parallel


def simulate(theta, n_agents, timesteps):
    simulator = Simulator(n_agents, timesteps=timesteps, disable_pbar=True)
    return theta.numpy(), simulator(theta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agents", type=int, default=10_000)
    parser.add_argument("-t", "--timesteps", type=int, default=1000)
    parser.add_argument("-s", "--simulations", type=int, default=10_000)
    parser.add_argument("-w", "--workers", type=int, default=1)
    parser.add_argument("-b", "--burn_in", type=int, default=1000)
    parser.add_argument("-p", "--top_n", type=int, default=10)

    args = parser.parse_args()

    prior = torch.distributions.Uniform(
        torch.tensor([
            -0.1,   # beta low
             0.001, # mu low
             0.001, # p_death low
        ]),
        torch.tensor([
             0.1,   # beta high
             0.01, # mu high
             0.1,   # p_death high
        ])
    )

    pool = Parallel(args.workers, args.simulations)
    biased = np.random.random(args.simulations) < 0.5
    for i in range(args.simulations):
        theta = prior.sample()
        if not biased[i]:
            theta[0] = 0.0
            print(theta)
        pool.apply_async(simulate, args=(theta, args.agents, args.timesteps))
    pool.join()

    theta, samples = zip(*pool.result())
    ids = np.array([0] + [len(sample) for sample in samples]).cumsum()
    theta, samples = np.vstack(theta), np.vstack(samples)

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    if not os.path.exists("../data"):
        os.mkdir("../data")
    np.save(f"../data/{now}.theta.npy", theta)
    np.save(f"../data/{now}.samples.npy", samples)
    np.save(f"../data/{now}.ids.npy", ids)

    with open(f"../data/{now}.params.json", "w") as fp:
        json.dump(args.__dict__, fp)

    
