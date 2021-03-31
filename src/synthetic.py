# NOTE: running with many cores might lead to a "too many files open" error. Increase ulimit
#       to prevent that from happening.

import argparse
import json
import multiprocessing as mp
import os

from datetime import datetime

import numpy as np
import torch
from torch.distributions import Uniform

from augmentation import Normalizer
from simulation import Simulator
import utils



def simulate(theta, n_agents, timesteps, top_n, summarize):
    simulator = Simulator(
        n_agents,
        timesteps=timesteps,
        top_n=top_n,
        disable_pbar=True,
        summarize=summarize,
    )
    sample = simulator(theta)
    return np.repeat(theta.numpy()[None, :], len(sample), axis=0), sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agents", type=int, default=1000)
    parser.add_argument("-t", "--timesteps", type=int, default=1000)
    parser.add_argument("-s", "--simulations", type=int, default=10_000)
    parser.add_argument("-w", "--workers", type=int, default=1)
    parser.add_argument("-b", "--burn_in", type=int, default=1000)
    parser.add_argument("-p", "--top_n", type=int, default=0)
    parser.add_argument("-d", "--summarize", action="store_true")

    args = parser.parse_args()

    prior = utils.IndependentPriors(
        Uniform(0.001, 0.01), # beta prior
        Uniform(0.001, 0.01), # mu prior
        Uniform(0.001, 0.1)   # p_death prior
    )

    pool = utils.Parallel(args.workers, args.simulations)
    biased = np.random.random(args.simulations) < 0.5
    negative_bias = np.random.random(args.simulations) < 0.5
    for i in range(args.simulations):
        theta = prior.sample()
        if not biased[i]:
            theta[0] = 0.0
        elif negative_bias[i]:
            theta[0] = -theta[0]
        pool.apply_async(
            simulate,
            args=(theta, args.agents, args.timesteps, args.top_n, args.summarize),
        )
    pool.join()

    theta, samples = zip(*pool.result())
    ids = np.array([[i] * len(sample) for i, sample in enumerate(samples)])
    theta, samples, ids = np.vstack(theta), np.vstack(samples), np.hstack(ids)

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    if not os.path.exists("../data"):
        os.mkdir("../data")
    np.save(f"../data/{now}.theta.npy", theta)
    np.save(f"../data/{now}.samples.npy", samples)
    np.save(f"../data/{now}.ids.npy", ids)

    with open(f"../data/{now}.params.json", "w") as fp:
        json.dump(args.__dict__, fp)
