from typing import Sequence, Callable
import copy
from functools import reduce, cached_property
from operator import mul
from collections import OrderedDict

from tqdm import tqdm
from tqdm.contrib.itertools import product
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

from node_homotopy.pyhessian.hessian import hessian


def flatten_model_params(model: nn.Module, only_requires_grad: bool = True) -> Tensor:
    params = model.parameters()
    if only_requires_grad:
        params = filter(lambda p: p.requires_grad, params)
    return torch.cat([p.flatten() for p in params])


def unflatten_model_params(
    model: nn.Module,
    param_vec: Sequence,
    in_place: bool = True,
    only_requires_grad: bool = True,
) -> nn.Module:
    if not in_place:
        model = copy.deepcopy(model)
    param_vec = torch.tensor(param_vec)

    names, params_old = list(zip(*model.named_parameters()))
    if only_requires_grad:
        params_old = filter(lambda p: p.requires_grad, params_old)
    param_shapes = [p.shape for p in params_old]
    param_lengths = [reduce(mul, s) for s in param_shapes]

    params_new = torch.split(param_vec, param_lengths)
    params_new = [p.view(s) for p, s in zip(params_new, param_shapes)]
    model.load_state_dict(OrderedDict(zip(names, params_new)), strict=False)
    return model


def make_1d_landscape(
    model: nn.Module,
    scalar_func: Callable[[nn.Module], Tensor],
    basis: Sequence,
    coeffs: Sequence,
) -> np.ndarray:
    param_orig = flatten_model_params(model)
    basis = torch.tensor(basis)
    # In the future, need to validate the shapes of the bases
    # Also check if they are normalized and orthogonal here.
    model_ = copy.deepcopy(model)
    N = len(coeffs)

    losses = np.empty((N,), dtype=float)
    with torch.no_grad():
        for i, c in enumerate(tqdm(coeffs)):
            param_new = param_orig.add(basis, alpha=c)
            unflatten_model_params(model_, param_new)
            losses[i] = scalar_func(model_).item()
    return losses


def make_2d_landscape(
    model: nn.Module,
    scalar_func: Callable[[nn.Module], Tensor],
    basis_x: Sequence,
    basis_y: Sequence,
    coeffs_x: Sequence,
    coeffs_y: Sequence,
) -> np.ndarray:
    param_orig = flatten_model_params(model)
    basis_x, basis_y = torch.tensor(basis_x), torch.tensor(basis_y)
    # In the future, need to validate the shapes of the bases
    # Also check if they are normalized and orthogonal here.
    model_ = copy.deepcopy(model)
    N_x, N_y = len(coeffs_x), len(coeffs_y)

    losses = np.empty((N_x * N_y), dtype=float)
    with torch.no_grad():
        for i, (c_x, c_y) in enumerate(product(coeffs_x, coeffs_y)):
            param_new = param_orig.add(basis_x, alpha=c_x)
            param_new.add_(basis_y, alpha=c_y)
            unflatten_model_params(model_, param_new)
            losses[i] = scalar_func(model_).item()
    return losses.reshape((N_x, N_y))


class PCALandscapeAnalyzer:
    def __init__(self, model_params: np.ndarray, best_epoch: int):
        self.model_params = np.array(model_params)
        self._pca = PCA(n_components=2)
        self.best_epoch = best_epoch
        self._find_basis_vectors()

    def _find_basis_vectors(self) -> None:
        delta_params = (
            self.model_params[: self.best_epoch] - self.model_params[self.best_epoch]
        )
        self._pca.fit(delta_params)

    @property
    def basis_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        bases = self._pca.components_
        return bases[0], bases[1]

    @cached_property
    def parameter_trajectory(self) -> np.ndarray:
        param_traj = self._pca.transform(self.model_params)
        param_traj = param_traj - param_traj[self.best_epoch]
        return param_traj[: self.best_epoch + 1]


class HessianLandscapeAnalyzer:
    def __init__(self, model: nn.Module, dataloader: DataLoader):
        self._hessian = hessian(model, dataloader, cuda=False)
        eigvals, eigvecs = self._compute_hessian_eigenbases()
        self.basis_vectors = eigvecs
        self.eigenvalues = eigvals

    def _compute_hessian_eigenbases(
        self,
    ) -> tuple[tuple[float, float], tuple[np.ndarray, np.ndarray]]:
        eigvals, eigvecs = self._hessian.eigenvalues(top_n=2)
        eigvec0 = torch.cat([x.flatten() for x in eigvecs[0]]).numpy()
        eigvec1 = torch.cat([x.flatten() for x in eigvecs[1]]).numpy()
        return tuple(eigvals), (eigvec0, eigvec1)
