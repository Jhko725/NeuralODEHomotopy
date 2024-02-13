import torch

from node_homotopy.dynamics import Lorenz3D
from node_homotopy.datasets import DynamicsDataset
from node_homotopy.typealiases import ElementwiseFunction
from node_homotopy.models import FullyConnectedNetwork, Blackbox

ODESOLVE_KWARGS = {"method": "dopri5", "rtol": 1e-7, "atol": 1e-9}
DYNAMICS_PARAMS = {"s": 10.0, "r": 28.0, "b": 8.0 / 3.0}


def make_lorenz_dataset(
    t_end: float,
    dt=0.1,
    noise_amplitude: float = 0.25,
) -> DynamicsDataset:
    lorenz = Lorenz3D(**DYNAMICS_PARAMS)
    u0 = torch.as_tensor([1.2, 2.1, 1.7]).view(1, -1)
    t = torch.arange(0, t_end, dt)
    dataset = DynamicsDataset.from_dynamics(lorenz, u0, t)
    dataset.add_noise(noise_amplitude, relative_to=None)
    return dataset


def make_lorenz_model(
    nodes_per_layer: int = 50,
    activation: ElementwiseFunction = torch.nn.functional.gelu,
) -> Blackbox:
    """Creates the Blackbox NeuralODE model for the Lorenz3D dataset.

    Args:
        nodes_per_layer: Integer describing the number of nodes per hidden layer.
            Defaults to 50, which is the value used in our paper.
        activation: Activation function to be used in the fully connected network.
            Defaults to the GeLU function (torch.nn.functional.gelu).

    Returns:
        model: Blackbox NeuralODE model for the Lorenz3D dataset.
    """
    surrogate = FullyConnectedNetwork(
        [3, nodes_per_layer, nodes_per_layer, 3], activation
    )
    model = Blackbox(surrogate)

    return model
