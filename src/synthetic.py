import argparse
import json
import multiprocessing as mp
import os

from datetime import datetime

import numpy as np
import torch
import torch.distributions as dists

from simulation import Simulator
import utils


def simulate(theta, n_agents, timesteps, top_n, summarize):
    simulator = Simulator(
        n_agents,
        timesteps=timesteps,
        top_n=top_n,
        restricted_age_window=True,
        disable_pbar=True,
        summarize=summarize,
    )
    sample = simulator(theta)
    if isinstance(theta, torch.Tensor):
        theta = theta.numpy()
    return np.repeat(theta[None, :], len(sample), axis=0), sample


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agents", type=int, default=1000)
    parser.add_argument("-t", "--timesteps", type=int, default=1000)
    parser.add_argument("-s", "--simulations", type=int, default=10_000)
    parser.add_argument("-w", "--workers", type=int, default=1)
    parser.add_argument("--beta", nargs=2, type=float, default=(0.0, 0.01))
    parser.add_argument("--mu", nargs=2, type=float, default=(0.0001, 0.1))
    parser.add_argument("--p_death", nargs=2, type=float, default=(0.1, 1.0))
    parser.add_argument("--age_max", nargs=2, type=float, default=(0.001, 1.0))
    parser.add_argument("-b", "--burn_in", type=int, default=1000)
    parser.add_argument("-p", "--top_n", type=int, default=0)
    parser.add_argument("--parameter_sweep", action="store_true")
    parser.add_argument("-d", "--summarize", action="store_true")

    args = parser.parse_args()

    if not args.parameter_sweep:
        prior = utils.IndependentPriors(
            dists.Normal(*args.beta),         # beta prior
            dists.Uniform(*args.mu),          # mu prior
            dists.Uniform(*args.p_death),     # p_death prior
            dists.Uniform(*args.age_max),     # threshold_high prior
        )
    else:
        prior = utils.ParamSweep(
            np.linspace(*args.beta, num=51),
            [0.01],
            np.linspace(*args.p_death, num=51),
            [0.001, 0.01, 1.0],
            num_samples=100
        )

    if not args.parameter_sweep:
        pool = utils.Parallel(args.workers, args.simulations)
        biased = np.random.random(args.simulations) < 0.5
        for i in range(args.simulations):
            theta = prior.sample().numpy()
            if not biased[i]:
                theta[0] = 0.0
            pool.apply_async(
                simulate,
                args=(theta, args.agents, args.timesteps, args.top_n, args.summarize)
            )
        pool.join()
    else:
        pool = utils.Parallel(args.workers, len(prior))
        for theta in prior:
            pool.apply_async(
                simulate,
                args=(theta, args.agents, args.timesteps, args.top_n, args.summarize)
            )
        pool.join()

    theta, samples = zip(*pool.result())
    ids = np.array([[i] * len(sample) for i, sample in enumerate(samples)])
    theta, samples, ids = np.vstack(theta), np.vstack(samples), np.hstack(ids)

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    if not os.path.exists("../data"):
        os.mkdir("../data")
    np.savez_compressed(
        f"../data/{now}.npz",
        theta=theta.astype(np.float64),
        samples=samples,
        ids=ids.astype(np.int64)
    )

    with open(f"../data/{now}.params.json", "w") as fp:
        json.dump(args.__dict__, fp)
