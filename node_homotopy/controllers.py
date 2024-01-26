import torch
from torch import nn

from node_homotopy.datasets import IVPDataset

from .interpolation import (
    CubicSplineSmoothing,
    get_interpolation,
)
from .typealiases import Tensorable


class Controller(nn.Module):
    def __init__(self, interpolant: CubicSplineSmoothing, k: Tensorable):
        super().__init__()
        self.k = nn.Parameter(torch.as_tensor(k))
        self.interpolant = interpolant

    def forward(self, t, u):
        u_interp = self.interpolant(t).squeeze(-1)
        return self.k * (u_interp - u)


class ControlledDynamics(nn.Module):
    def __init__(self, dynamics: nn.Module, control: Controller):
        super().__init__()
        self.dynamics = dynamics
        self.control = control

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.dynamics(t, u) + self.control(t, u)


def make_controller(
    dataset: IVPDataset, k: Tensorable, interp_alg: str = "cubic", **interpolant_kwargs
) -> Controller:
    interpolation = get_interpolation(interp_alg)
    interpolant = interpolation(dataset.t, dataset.u, **interpolant_kwargs)
    controller = Controller(interpolant, k)
    return controller
