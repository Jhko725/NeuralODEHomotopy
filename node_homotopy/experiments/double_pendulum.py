# ruff: noqa: F722
from typing import Literal
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from node_homotopy.typealiases import FilePath, ElementwiseFunction
from node_homotopy.datasets import DynamicsDataset
from node_homotopy.models import FullyConnectedNetwork, Blackbox, SecondOrderBlackbox

DATADIR = Path(__file__).parent / "data"


def preprocess_pendulum_data(
    filepath: FilePath,
) -> list[Float[np.ndarray, "_timepoints 5"]]:
    filepath = Path(filepath)
    raw_data = np.genfromtxt(filepath, delimiter=" ", skip_header=1)
    raw_data = raw_data[:, 0:6]  # Discard unneeded columns

    sample_ids = np.unique(raw_data[:, 0])
    trajectories = [raw_data[raw_data[:, 0] == id, 1:] for id in sample_ids]
    return trajectories


def make_pendulum_dataset(
    num_timepoints: int = 100,
    filepath: FilePath = DATADIR / "invar_datasets/real_double_pend_h_1.txt",
) -> DynamicsDataset:
    trajectories = preprocess_pendulum_data(filepath)
    trajectory = trajectories[0]  # Only use the first trajectory
    t: Float[np.ndarray, "{num_timepoints}"] = trajectory[:num_timepoints, 0]
    t = t - t[0]  # Make time start from zero
    t = torch.as_tensor(t, dtype=torch.float32)

    u: Float[np.ndarray, "{num_timepoints} 4"] = trajectory[:num_timepoints, 1:]
    u: Float[np.ndarray, "1 4 {num_timepoints}"] = np.expand_dims(
        u.T, axis=0
    )  # Make into proper shape for DynamicsDataset
    u = torch.as_tensor(u, dtype=torch.float32)

    dataset = DynamicsDataset(t, u)
    return dataset


def make_pendulum_model(
    model_type: Literal["blackbox", "secondorder"],
    nodes_per_layer: int = 50,
    activation: ElementwiseFunction = torch.nn.functional.gelu,
) -> Blackbox | SecondOrderBlackbox:
    """Creates the appropriate NeuralODE model for the double pendulum experimental data from
    Schmidt and Lipson, Science 324, 5923 (2009).

    Two model types are available.
    1) blackbox - A simple NeuralODE with a neural network modeling the entire right hand side.
    2) secondorder - A NeuralODE with second-order dynamics inductive bias, as discussed in
    Gruver et al. Deconstructing the Inductive Biases of Hamiltonian Neural Networks. ICLR 2021.

    Args:
        model_type: String describing the type of the NeuralODE model. Must be either "blackbox" or "secondorder".
        nodes_per_layer: Integer describing the number of nodes per hidden layer.
        activation: Activation function to be used in the fully connected network.
            Defaults to the GeLU function (torch.nn.functional.gelu).

    Returns:
        model: NeuralODE model for the double pendulum dataset.

    Raises:
        ValueError: The given model_type is not one of ["blackbox", "secondorder"].
    """
    match model_type:
        case "blackbox":
            surrogate = FullyConnectedNetwork(
                [4, nodes_per_layer, nodes_per_layer, 4], activation
            )
            model = Blackbox(surrogate)
        case "secondorder":
            surrogate = FullyConnectedNetwork(
                [4, nodes_per_layer, nodes_per_layer, 2], activation
            )
            model = SecondOrderBlackbox(surrogate)
        case _:
            raise ValueError(
                """Unsupported model type - must be one of ["blackbox", "secondorder"]."""
            )
    return model
