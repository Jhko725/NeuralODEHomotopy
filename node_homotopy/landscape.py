import copy
from functools import cached_property

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import Dataset, DataLoader
import lightning
from lightning import LightningModule
import more_itertools

from node_homotopy.typealiases import ArrayOrTensor
from node_homotopy.pyhessian.hessian import hessian, density_generate


def calculate_hessian_eigenbasis(
    model: LightningModule, dataset: Dataset, cuda: bool = True
) -> tuple[Tensor, Tensor]:
    """Computes the top two eigenvectors of the loss Hessian.

    This is useful when using these eigenvectors as axes in which to visualize the loss landscape.

    Args:
        model: LightningModule representing the model to calculate the eigenvectors for.
        dataset: The dataset to compute the loss with.
        cuda: Boolean indicating whether to use the gpu for the calculation.

    Returns:
        eigvecs: Tuple of two 1D Tensors representing the flattened top two dominant eigenvectors of the loss Hessian.
    """
    dataloader = DataLoader(dataset)
    hess = hessian(model, dataloader, cuda=cuda)
    _, eigvecs = hess.eigenvalues(top_n=2)
    eigvecs = tuple(parameters_to_vector(e) for e in eigvecs)
    return eigvecs


def calculate_hessian_eigspectrum(
    model: LightningModule,
    dataset: Dataset,
    eps: float = 1e-7,
    cuda: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the approximate eigenvalue spectrum of the loss Hessian.

    Args:
        model: LightningModule representing the model to calculate the eigenvalue spectrum for.
        dataset: The dataset to compute the loss with.
        eps: A small positive float value denoting the floor of the spectrum.
            Otherwise, the zero portion of the spectrum, when plotted in log-y scale, drops down to negative infinity.
        cuda: Boolean indicating whether to use the gpu for the calculation.

    Returns:
        grids: Location of the eigenvalues - corresponds to the x-axis of the spectrum.
        densities: Density of the eigenvalues at each grid value - corresponds to the y-axis of the spectrum.
    """
    dataloader = DataLoader(dataset)
    hess = hessian(model, dataloader, cuda=cuda)
    densities, grids = density_generate(*hess.density())
    return grids, densities + eps


def calculate_hessian_trace(
    model: LightningModule,
    dataset: Dataset,
    cuda: bool = True,
) -> float:
    """Computes the trace of the loss Hessian.

    Args:
        model: LightningModule representing the model to calculate the Hessian trace for.
        dataset: The dataset to compute the loss with.
        cuda: Boolean indicating whether to use the gpu for the calculation.

    Returns:
        trace: Float corresponding to the trace of the loss Hessian.
    """
    dataloader = DataLoader(dataset)
    hess = hessian(model, dataloader, cuda=cuda)
    return np.mean(hess.trace())


def flatten_model_params(model: nn.Module, only_requires_grad: bool = True) -> Tensor:
    """Given a nn.Module, flatten and concatenate its parameters into a 1D Tensor and return it.

    Args:
        model: The nn.Module whose parameters are to be flattened.
        only_requires_grad: Boolean value indicating whether to only consider parameters with requires_grad = True,
            or flatten all parameters regardless of its requires_grad status.

    Returns:
        flattened_params: 1D Tensor corresponding to the flattened and concatenated parameters.
    """
    params = model.parameters()
    if only_requires_grad:
        params = filter(lambda p: p.requires_grad, params)
    return parameters_to_vector(params)


def unflatten_model_params(
    model: nn.Module,
    param_vec: ArrayOrTensor,
    in_place: bool = True,
    only_requires_grad: bool = True,
) -> nn.Module:
    """Given a nn.Module and a 1D vector representing a new set of flattened parameters, load the new parameters into the model.

    Args:
        model: The nn.Module to load the new parameters into.
        param_vec: 1D numpy array or torch Tensor representing the flattened vector of new parameters to be loaded into the model.
        in_place: Boolean indicating whether to mutate the given model or create a copy.
        only_require_grad: Boolean indicating whether the parameters to be loaded correspond to only those that have required_grad=True or all parameters are to be loaded.

    Returns:
        model: The nn.Module with the new set of parameters.
    """
    if not in_place:
        model = copy.deepcopy(model)
    param_vec = torch.as_tensor(param_vec)

    param_dict = dict(model.named_parameters())
    if only_requires_grad:
        param_dict = {k: v for k, v in param_dict.items() if v.requires_grad}

    parameters = param_dict.values()
    vector_to_parameters(param_vec, parameters)  # Mutates parameters in place
    model.load_state_dict(dict(zip(param_dict.keys(), parameters)), strict=False)
    return model


def make_2d_landscape(
    model: LightningModule,
    dataset: Dataset,
    bases: tuple[ArrayOrTensor, ArrayOrTensor],
    coeffs: tuple[ArrayOrTensor, ArrayOrTensor],
    cuda: bool = True,
) -> np.ndarray:
    """Calculates the 2D loss landscape of the given model in the plane spanned by the given bases.

    The grid points at which the 2D loss landscape is evaluated is given by the two vectors in coeffs.
    Note that the function assumes the batch size to be one.

    Args:
        model: LightningModule representing the model to compute the loss landscape for.
        dataset: Dataset to compute the loss with.
        bases: Tuple of two 1D numpy arrays or torch Tensors indicating the flattened basis vectors spanning the plane to compute the loss landscape on.
        coeffs: Tuple of two 1D numpy arrays or torch Tensors denoting the grid points to compute the loss at.
        cuda: Boolean indicating whether to use gpu for the computation.

    Return:
        losses: 2D Tensor corresponding to the computed 2D loss landscape.
    """
    # Use lightning fabric to manage device
    accelerator = "cuda" if cuda else "cpu"
    fabric = lightning.Fabric(accelerator=accelerator, devices=1)
    param_orig = flatten_model_params(model).to(fabric.device)

    model_: lightning.LightningModule = fabric.setup_module(copy.deepcopy(model))
    dataloader = fabric.setup_dataloaders(DataLoader(dataset))

    with torch.device(param_orig.device):
        basis_x, basis_y = torch.as_tensor(bases[0]), torch.as_tensor(bases[1])
        # In the future, need to validate the shapes of the bases
        # Also check if they are normalized and orthogonal here.
        losses = torch.empty((len(coeffs[0]), len(coeffs[1])), dtype=float)

    with torch.no_grad():
        for i, c_x in enumerate(coeffs[0]):
            param_new_ = param_orig.add(basis_x, alpha=c_x)
            for j, c_y in enumerate(coeffs[1]):
                param_new = param_new_.add(basis_y, alpha=c_y)
                model_ = unflatten_model_params(model_, param_new)
                batch = more_itertools.one(dataloader)
                losses[i, j] = model_.training_step(batch, 0)
    return losses


class PCALandscapeAnalyzer:
    def __init__(self, model_params: np.ndarray, best_epoch: int):
        try:
            from sklearn.decomposition import PCA

        except ImportError:
            raise ImportError(
                """To perform the PCA loss landscape analysis, need to install the scikit-learn package."""
            )

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
