import copy

from termcolor import colored
from typing import Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from dataset import LazySimulationDataset
from dataset import lazy_worker_init_fn, collate_timeseries
from simulation import Simulator


def train(
    prior: Type[torch.distributions.Distribution] = None,
    simulator: Type[Simulator] = None,
    num_simulations: int = None,
    n_epochs: int = 10,
    classifier: nn.Module = None,
    batch_size: int = 10,
    learning_rate: float = 0.001,
    learning_patience: int = 5,
    device: str = "cpu",
    clip_max_norm: float = None,
    num_workers: int = 1
):

    # Set up the training and validation data
    train_data = LazySimulationDataset(simulator, prior, num_simulations)
    train_loader = DataLoader(
        train_data,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=collate_timeseries,
        worker_init_fn=lazy_worker_init_fn,
    )
    val_data = SimulationDataset(simulator, prior, num_simulations)
    val_loader = DataLoader(
        val_data,
        num_workers=num_workers,
        batch_size=batch_size,
        collate_fn=collate_timeseries
    )

    classifier = classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=learning_patience, verbose=True
    )

    best_val_loss = float("inf")
    best_params = None
    for epoch in range(n_epochs):
        train_loss = train_epoch(classifier, train_loader, optimizer, device, clip_max_norm=clip_max_norm)
        val_loss, auc = test_epoch(classifier, val_loader, device)

        out = (
            f"Epoch {epoch + 1:2d}: train loss = {train_loss:.2f}, "
            f"val loss = {val_loss:.2f}, AUC = {auc:.2f}"
        )

        if val_loss < best_val_loss:
            out += colored(" ++", "green")
            best_val_loss = val_loss
            best_params -= copy.deepcopy(classifier.state_dict())
        else:
            out += colored(" --", "red")
        print(out)

        lr_scheduler.step(val_loss)

    classifier.load_state_dict(best_params)
    classifier.eval()
    return classifier


def train_epoch(classifier, loader, optimizer, device, clip_max_norm=None):
    classifier.train()
    epoch_loss = []
    for theta, x in loader:
        theta, x = theta.to(device), x.to(device)
        x = x.permute(1, 2, 0).float()
        optimizer.zero_grad()
        probs = classifier(x)
        labels = (theta[:, 0] != 0.0).float()
        loss = F.binary_cross_entropy_with_logits(probs, labels.unsqueeze(0))
        epoch_loss.append(loss.item())
        loss.backward()
        if clip_max_norm is not None:
            if clip_max_norm is not None:
                clip_grad_norm_(classifier.parameters(), max_norm=clip_max_norm)
        optimizer.step()
    return np.mean(epoch_loss)


def test_epoch(classifier, loader, device):
    classifier.eval()
    epoch_loss = []
    biased, pobabilities = [], []
    with torch.no_grad():
        for theta, x in loader:
            theta, x = theta.to(device), x.to(device)
            x = x.permute(1, 2, 0)
            probs = classifier(x)
            labels = (theta[:, 0] != 0.0).float()
            loss = F.binary_cross_entropy_with_logits(probs, labels.unsqueeze(0))
            epoch_loss.append(loss.item())
    biased.extend(labels.cpu().numpy().tolist())
    probabilities.extend(probs.cpu().numpy().tolist())
    return np.mean(epoch_loss), roc_auc_score(biased, probabilities)
