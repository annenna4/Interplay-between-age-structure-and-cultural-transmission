import copy

from termcolor import colored
from typing import Type, Callable

import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from dataset import LazySimulationDataset, SimulationDataset, PresimulatedDataset
from dataset import lazy_worker_init_fn, collate_timeseries
from simulation import Simulator


def train(
    data: PresimulatedDataset = None,
    prior: Type[torch.distributions.Distribution] = None,
    simulator: Type[Simulator] = None,
    num_simulations: int = None,
    n_epochs: int = 10,
    classifier: nn.Module = None,
    transform: Callable = None,
    batch_size: int = 10,
    learning_rate: float = 0.001,
    test_size=0.1,
    learning_patience: int = 5,
    device: str = "cpu",
    clip_max_norm: float = None,
    num_workers: int = 1
):

    # Set up the training and validation data
    if data is None:
        train_data = LazySimulationDataset(simulator, prior, num_simulations, transform=transform)
        val_data = SimulationDataset(simulator, prior, num_simulations, transform=transform)
    else:
        train_data, val_data = data.train_test_split(test_size=test_size)

    train_loader = DataLoader(
        train_data,
        num_workers=num_workers,
        batch_size=batch_size,
        worker_init_fn=lazy_worker_init_fn,
        # collate_fn=collate_timeseries
    )
    val_loader = DataLoader(
        val_data,
        num_workers=num_workers,
        batch_size=batch_size,
        # collate_fn=collate_timeseries        
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
            best_params = copy.deepcopy(classifier.state_dict())
        else:
            out += colored(" --", "red")
        print(out)

        lr_scheduler.step(val_loss)

    classifier.load_state_dict(best_params)
    classifier.eval()
    return classifier, train_loader, val_loader


def train_epoch(classifier, loader, optimizer, device, clip_max_norm=None):
    classifier.train()
    epoch_loss = []
    for theta, x in loader:
        theta, x = theta.to(device), x.to(device)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        optimizer.zero_grad()
        output = classifier(x.float())
        labels = (theta[:, 0] != 0.0).float()
        loss = F.binary_cross_entropy(output.view(-1), labels.view(-1))
        epoch_loss.append(loss.item())
        loss.backward()
        if clip_max_norm is not None:
            if clip_max_norm is not None:
                clip_grad_norm_(classifier.parameters(), max_norm=clip_max_norm)
        optimizer.step()
    return np.mean(epoch_loss)


def test_epoch(classifier, loader, device):
    classifier.eval()
    epoch_loss, roc = [], []
    with torch.no_grad():
        for theta, x in loader:
            theta, x = theta.to(device), x.to(device)
            if x.dim() == 2:
                x = x.unsqueeze(1)
            output = classifier(x.float())
            labels = (theta[:, 0] != 0.0).float()
            loss = F.binary_cross_entropy(output.view(-1), labels.view(-1))
            epoch_loss.append(loss.item())
            roc.append(roc_auc_score(labels.view(-1).cpu().numpy(), output.cpu().numpy()))
    return np.mean(epoch_loss), np.mean(roc)


def classify_example(x, classifier, device="cpu"):
    classifier = classifier.to(device)
    return classifier(x.unsqueeze(1))
    
