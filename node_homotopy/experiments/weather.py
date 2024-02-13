from pathlib import Path

import numpy as np
import torch

from node_homotopy.typealiases import FilePath, ElementwiseFunction
from node_homotopy.datasets import DynamicsDataset
from node_homotopy.models import FullyConnectedNetwork, Blackbox

DATADIR = Path(__file__).parent / "data"


def make_weather_dataset(
    filepath: FilePath = DATADIR / "tde_temperature_MoreSmoothedNoYearShort.npy",
    t_end: int = 2000,
    dt: int = 40,
    scale_factor: float = 0.1,
) -> DynamicsDataset:
    raw_data = np.load(filepath)
    raw_data = np.expand_dims(raw_data.T, axis=0)

    u = raw_data[..., 0:t_end:dt] * scale_factor
    u = torch.as_tensor(u)
    t = torch.arange(u.shape[-1])
    return DynamicsDataset(t, u)


def make_weather_model(
    nodes_per_layer: int = 50,
    activation: ElementwiseFunction = torch.nn.functional.gelu,
) -> Blackbox:
    """Creates the Blackbox NeuralODE model for the weather dataset.

    Args:
        nodes_per_layer: Integer describing the number of nodes per hidden layer.
            Defaults to 50, which is the value used in our paper.
        activation: Activation function to be used in the fully connected network.
            Defaults to the GeLU function (torch.nn.functional.gelu).

    Returns:
        model: Blackbox NeuralODE model for the weather dataset.
    """
    surrogate = FullyConnectedNetwork(
        [5, nodes_per_layer, nodes_per_layer, 5], activation
    )
    model = Blackbox(surrogate)

    return model
