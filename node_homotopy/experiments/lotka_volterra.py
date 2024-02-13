from typing import Literal

import torch

from node_homotopy.dynamics import LotkaVolterra
from node_homotopy.datasets import DynamicsDataset
from node_homotopy.typealiases import ElementwiseFunction
from node_homotopy.models import FullyConnectedNetwork, Blackbox, LotkaVolterraGrayBox

ODESOLVE_KWARGS = {"method": "dopri5", "rtol": 1e-7, "atol": 1e-9}
DYNAMICS_PARAMS = {"a": 1.3, "b": 0.9, "c": 0.8, "d": 1.8}


def make_lotka_dataset(
    t_end: float,
    dt: float = 0.1,
    noise_amplitude: float = 0.05,
) -> DynamicsDataset:
    """Creates the dataset for the Lotka-Volterra system for the given set of experimental parameters.

    The coefficients of the Lotka-Volterra system, as well as the initial conditions are set to be identical
    to that of C. Rackauckas et al. Universal Differential Equations for Scientific Machine Learning. arXiv:2001.04385 (2021).
    Again, following the above paper, the noise magnitude is calculated relative to the mean of each state trajectory (e.g. 5% of the mean).

    Args:
        t_end: The end timepoint of the training dataset.
        dt: The sampling period of the training dataset.
        noise_amplitude: The ratio of the amplitude of the noise, relative to the mean of the trajectory per state.

    Returns:
        dataset: Dataset used to train the model.
    """
    lotkavolterra = LotkaVolterra(**DYNAMICS_PARAMS)
    u0 = torch.tensor([0.44249296, 4.6280594]).view(1, -1)
    t = torch.arange(0, t_end, dt)
    dataset = DynamicsDataset.from_dynamics(lotkavolterra, u0, t)
    dataset.add_noise(noise_amplitude, relative_to="mean")
    return dataset


def make_lotka_model(
    model_type: Literal["blackbox", "graybox"],
    nodes_per_layer: int,
    activation: ElementwiseFunction = torch.nn.functional.gelu,
) -> Blackbox | LotkaVolterraGrayBox:
    """Creates the appropriate NeuralODE model for the Lotka-Volterra dataset.

    Two model types are available.
    1) blackbox - A simple NeuralODE with a neural network modeling the entire right hand side.
    2) graybox - A "graybox" model or a "universal differential equation", with the model structure identical to that of
        C. Rackauckas et al. Universal Differential Equations for Scientific Machine Learning. arXiv:2001.04385 (2021).

    Args:
        model_type: String describing the type of the NeuralODE model. Must be either "blackbox" or "graybox".
        nodes_per_layer: Integer describing the number of nodes per hidden layer.
        activation: Activation function to be used in the fully connected network.
            Defaults to the GeLU function (torch.nn.functional.gelu).

    Returns:
        model: NeuralODE model for the Lotka-Volterra dataset.

    Raises:
        ValueError: The given model_type is not one of ["blackbox", "graybox"].
    """
    match model_type:
        case "blackbox":
            surrogate = FullyConnectedNetwork([2, nodes_per_layer, 2], activation)
            model = Blackbox(surrogate)
        case "graybox":
            surrogate = FullyConnectedNetwork(
                [2, nodes_per_layer, nodes_per_layer, 2], activation
            )
            model = LotkaVolterraGrayBox(
                surrogate, known_params={k: DYNAMICS_PARAMS[k] for k in ("a", "d")}
            )
        case _:
            raise ValueError(
                """Unsupported model type - must be one of ["blackbox", "graybox"]."""
            )
    return model
