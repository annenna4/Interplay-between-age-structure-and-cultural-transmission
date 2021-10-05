"""Script to convert samples from matlab to the same format in NumPy"""

import os
import re
import numpy as np

params_re = re.compile(r"samples_N\d+_pMut(.*?)_pDeath(.*?)_b(.*?)_thLow0_thHigh(\d+)_nSample(\d+).txt")

def extract_parameters(fname):
    return np.array(params_re.findall(fname)).astype(float)

FOLDERS = [f"thHigh{d}{f}" for d in (2, 6, 11, 21) for f in ("", "_pDeath01")]

theta, samples = [], []
for folder in FOLDERS:
    for fname in os.scandir(f"../data/{folder}"):
        if fname.name.endswith(".txt"):
            data = np.loadtxt(fname.path).T
            samples.append(data)
            params = extract_parameters(fname.name)
            theta.append(np.repeat(params, data.shape[0], 0))
max_len = max(len(sample[0]) for sample in samples)
samples = [
    np.pad(sample, [(0, 0), (0, max_len - len(sample[0]))], "constant")
    for sample in samples
]
theta, samples = np.vstack(theta), np.vstack(samples)
ids = np.arange(samples.shape[0])
np.savez_compressed(
    f"../data/matlab-testset.npz",
    theta=theta.astype(np.float64),
    samples = samples,
    ids=ids.astype(np.int64),
)
