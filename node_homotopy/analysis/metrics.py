import numpy as np
import torch
from torch.utils.data import DataLoader

from node_homotopy.learners import BasicLearner
from node_homotopy.datasets import IVPDataset
from node_homotopy.odesolve import odesolve
from node_homotopy.pyhessian.hessian import hessian, density_generate


def calculate_mse(learner: BasicLearner, dataset: IVPDataset, **odesolve_kwargs):
    u_pred = odesolve(learner.model, dataset.u0, dataset.t, **odesolve_kwargs)
    return torch.nn.functional.mse_loss(u_pred, dataset.u, reduction="mean")


def calculate_train_loss(learner: BasicLearner, dataset: IVPDataset, **odesolve_kwargs):
    u_pred = odesolve(learner, dataset.u0, dataset.t, **odesolve_kwargs)
    return torch.nn.functional.mse_loss(u_pred, dataset.u, reduction="mean")


def calculate_hessian_eigspectrum(
    learner: BasicLearner, dataset: IVPDataset, eps: float = 1e-7
) -> tuple[np.ndarray, np.ndarray]:
    dl = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True, shuffle=True)
    hess = hessian(learner, dl, cuda=False)
    densities, grids = density_generate(*hess.density())
    return grids, densities + eps


def calculate_hessian_trace(learner: BasicLearner, dataset: IVPDataset) -> float:
    dl = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True, shuffle=True)
    hess = hessian(learner, dl, cuda=False)
    return np.mean(hess.trace())
