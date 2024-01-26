from abc import ABC, abstractmethod, abstractproperty

import torch
from torch import Tensor, nn

from .typealiases import Tensorable
from .odesolve import odesolve


class AbstractDynamics(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def _register_params(self, param_dict: dict[str, Tensorable]) -> None:
        # Convert a given python dictionary of parameters into a proper nn.ParameterDict to be registered
        self.params = nn.ParameterDict(
            {k: nn.Parameter(torch.as_tensor(v)) for k, v in param_dict.items()}
        )

    @abstractproperty
    def dof(cls) -> int:
        # Degree of freedom of the dynamics to be implemented as class attributes in all child classes
        pass

    @abstractmethod
    def forward(self, t: Tensor, u: Tensor) -> Tensor:
        # The actual dynamics
        pass

    def solve(self, u0: Tensor, t: Tensor, **odesolve_kwargs) -> Tensor:
        u = odesolve(self, u0, t, **odesolve_kwargs)
        return u


class LotkaVolterra(AbstractDynamics):
    dof = 2  # Class attribute since this is same for all instances

    def __init__(self, a: float, b: float, c: float, d: float):
        super().__init__()
        param_dict = {"a": a, "b": b, "c": c, "d": d}
        self._register_params(param_dict)

    def forward(self, t: Tensor, u: Tensor) -> Tensor:
        # t: 1D tensor, (time)
        # u: 2D tensor (samples, dof); dof: degree of freedom
        # x = u[:, 0], y = u[:, 1]
        Dx = self.params["a"] * u[:, 0] - self.params["b"] * u[:, 0] * u[:, 1]
        Dy = self.params["c"] * u[:, 0] * u[:, 1] - self.params["d"] * u[:, 1]
        return torch.stack([Dx, Dy], dim=-1)


# True Lorenz System
class LorenzSystem(AbstractDynamics):
    dof = 3

    def __init__(self, s: float, r: float, b: float):
        super().__init__()
        param_dict = {"s": s, "r": r, "b": b}
        self._register_params(param_dict)

    # This is the actual equations for the Lorenz system, written in torch
    # u[:, 0] = x, u[:, 1] = y, u[:, 2] = z
    def forward(self, t: Tensor, u: Tensor) -> Tensor:
        Dx = self.params["s"] * (u[:, 1] - u[:, 0])
        Dy = u[:, 0] * (self.params["r"] - u[:, 2]) - u[:, 1]
        Dz = u[:, 0] * u[:, 1] - self.params["b"] * u[:, 2]
        return torch.stack((Dx, Dy, Dz), dim=-1)
