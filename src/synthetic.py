import argparse
import json
import multiprocessing as mp
import os

from datetime import datetime

import numpy as np
import torch.distributions as dists

from simulation import Simulator
import utils


def simulate(n_agents, beta, mu, p_death, eta, earlystopper, timesteps):
    simulator = Simulator(
        n_agents=n_agents,
        beta=beta,
        mu=mu,
        p_death=p_death,
        eta=eta,
        earlystopper=earlystopper,
        disable_pbar=True,
    ).fit()
    sample = simulator.sample(timesteps=timesteps)
    theta = np.array([beta, mu, p_death, eta])
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
    parser.add_argument("--age_max", nargs=2, type=int, default=(1, 21))
    parser.add_argument(
        "--burnin_strategy",
        type=str,
        choices=["diversity", "max_iter", "turnover"],
        default="diversity",
    )
    parser.add_argument("-b", "--burn_in", type=int, default=1000)
    parser.add_argument("-p", "--top_n", type=int, default=0)
    parser.add_argument("--parameter_sweep", action="store_true")

    args = parser.parse_args()

    if not args.parameter_sweep:
        prior = utils.IndependentPriors(
            dists.Normal(*args.beta),  # beta prior
            dists.Uniform(*args.mu),  # mu prior
            dists.Uniform(*args.p_death),  # p_death prior
            utils.Randint(*args.age_max),  # threshold_high prior
        )
    else:
        prior = utils.ParamSweep(
            np.linspace(*args.beta, num=16),
            [0.0005],
            [0.02, 0.1],
            [1, 5, 10, 20],
            num_samples=100,
        )

    if not args.parameter_sweep:
        pool = utils.Parallel(args.workers, args.simulations)
        biased = np.random.random(args.simulations) < 0.5
        for i in range(args.simulations):
            beta, mu, p_death, eta = prior.sample().numpy()
            if not biased[i]:
                beta = 0.0
            pool.apply_async(
                simulate,
                args=(
                    args.agents,
                    beta,
                    mu,
                    p_death,
                    eta,
                    args.burnin_strategy,
                    args.timesteps,
                ),
            )
        pool.join()
    else:
        pool = utils.Parallel(args.workers, len(prior))
        for theta in prior:
            beta, mu, p_death, eta = theta
            pool.apply_async(
                simulate,
                args=(
                    args.agents,
                    beta,
                    mu,
                    p_death,
                    eta,
                    args.burnin_strategy,
                    args.timesteps,
                ),
            )
        pool.join()

    theta, samples = zip(*pool.result())
    ids = np.array([[i] * len(sample) for i, sample in enumerate(samples)])
    max_len = max(len(sample[0]) for sample in samples)
    samples = [
        np.pad(sample[0], (0, max_len - len(sample[0])), "constant")
        for sample in samples
    ]
    theta, samples, ids = np.vstack(theta), np.vstack(samples), np.hstack(ids)

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    if not os.path.exists("../data"):
        os.mkdir("../data")
    np.savez_compressed(
        f"../data/{now}.npz",
        theta=theta.astype(np.float64),
        samples=samples,
        ids=ids.astype(np.int64),
    )

    with open(f"../data/{now}.params.json", "w") as fp:
        json.dump(args.__dict__, fp)
