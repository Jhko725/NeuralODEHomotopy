# ruff: noqa: F722
from abc import abstractmethod, abstractproperty

import torch
from torch import Tensor, nn
from jaxtyping import Float

from .typealiases import Tensorlike


class AbstractDynamics(nn.Module):
    """Abstact base class for continuous dynamical systems.

    Assumes that the dynamical system of interest is governed by an ordinary differential equation (ODE) of the form,
    du/dt = f(t, u), which is numerically solved via node_homotopy.odesolve.odesolve.
    To create a concrete subclass, override the self.forward() method with an implementation of f(t, u),
    and set the ChildClass.dof class attribute to be the degree of freedom (dimensionality) of the dynamics.

    Attributes:
        dof: An integer corresponding to the degree of freedom (dimensionality) of the dynamics.
        params: torch.nn.ParameterDict (PyTorch compatible dictionary) of the coefficients, constants, etc. of the dynamical system.
                Can be set to None.
    """

    def __init__(self):
        """Initializes the class."""
        super().__init__()
        self.params: None | nn.ParameterDict = None

    def _register_params(self, param_dict: dict[str, Tensorlike]) -> None:
        """Convenience method to register dictionary of parameters to the underlying torch.nn.Module.

        Usually, param_dict would be used to store a dictionary of coefficients for the dynamical system.
        This method properly binds this dictionary to the class, so that when the model is moved from cpu
        to gpu or vice versa, the parameter dictionary is also moved accordingly.
        The registered dictionary is written to self.params attribute.

        Args:
            param_dict: Dictionary of parameters of the dynamical system.
        """
        self.params = nn.ParameterDict(
            {k: nn.Parameter(torch.as_tensor(v)) for k, v in param_dict.items()}
        )

    @abstractproperty
    def dof(cls) -> int:
        """Degree of freedom (i.e. dimensionality) of the dynamical system.

        As such, must be an integer.

        Returns:
            dof: Integer corresponding to the degree of freedom of the dynamics.
        """
        pass

    @abstractmethod
    def forward(
        self, t: Float[Tensor, ""], u: Float[Tensor, "batch {self.dof}"]
    ) -> Float[Tensor, "batch {self.dof}"]:
        """Abstract method for the dynamics of the system.

        Corresponds to an implementation of f(t, u) for the ODE du/dt = f(t, u).

        Args:
            t: A scalar Tensor of shape (,) corresponding to time.
            u: Tensor of shape (batch, degree of freedom) corresponding to the state at time t.

        Returns:
            dudt: Tensor of shape (batch, degree of freedom) corresponding to the time derivative of the state at time t.
        """
        pass


class LotkaVolterra(AbstractDynamics):
    """Lotka-Volterra system.

    The Lotka-Volterra system is a simplified model of predator-prey population dynamics
    and is given by the equation:
    dx/dt = ax-bxy, dy/dt = cxy-dy

    Attributes:
        dof: Degree of freedom of the system. Equal to 2.
        params: torch.nn.ParameterDict of the coefficients a, b, c, d of the dynamics.
    """

    dof: int = 2  # Class attribute since this is same for all instances

    def __init__(self, a: float = 1.3, b: float = 0.9, c: float = 0.8, d: float = 1.8):
        """Initializes the class.

        Args:
            a: Float value for the coefficient a.
            b: Float value for the coefficient b.
            c: Float value for the coefficient c.
            d: Float value for the coefficient d.
        """
        super().__init__()
        param_dict = {"a": a, "b": b, "c": c, "d": d}
        self._register_params(param_dict)

    def forward(
        self, t: Float[Tensor, ""], u: Float[Tensor, "batch {self.dof}"]
    ) -> Float[Tensor, "batch {self.dof}"]:
        """Dynamics of the system.

        Args:
            t: A scalar Tensor of shape (,) corresponding to time.
            u: Tensor of shape (batch, degree of freedom = 2) corresponding to the state u = [x, y].

        Returns:
            dudt: Tensor of shape (batch, degree of freedom = 2) corresponding to the du/dt = [dx/dt, dy/dt].
        """
        del t
        Dx = self.params["a"] * u[:, 0] - self.params["b"] * u[:, 0] * u[:, 1]
        Dy = self.params["c"] * u[:, 0] * u[:, 1] - self.params["d"] * u[:, 1]
        return torch.stack([Dx, Dy], dim=-1)


class Lorenz3D(AbstractDynamics):
    """3D Lorenz system.

    The Lorenz system is a simplified model of atmospheric convection.
    Under proper choice of coefficients and initial conditions, the system displays chaotic dynamics,
    which leads to the well-known Lorenz attractor or the "butterfly attractor".
    The system is given by the equation:
    dx/dt = s(y-x), dy/dt = x(r-z)-y, dz/dt = xy-bz

    Attributes:
        dof: Degree of freedom of the system. Equal to 3.
        params: torch.nn.ParameterDict of the coefficients s, r, b of the dynamics.
    """

    dof: int = 3

    def __init__(self, s: float = 10.0, r: float = 28.0, b: float = 8.0 / 3.0):
        """Initializes the class.

        Args:
            s: Float value for the coefficient s.
            r: Float value for the coefficient r.
            b: Float value for the coefficient b.
        """
        super().__init__()
        param_dict = {"s": s, "r": r, "b": b}
        self._register_params(param_dict)

    def forward(
        self, t: Float[Tensor, ""], u: Float[Tensor, "batch {self.dof}"]
    ) -> Float[Tensor, "batch {self.dof}"]:
        """Dynamics of the system.

        Args:
            t: A scalar Tensor of shape (,) corresponding to time.
            u: Tensor of shape (batch, degree of freedom = 3) corresponding to the state u = [x, y, z].

        Returns:
            dudt: Tensor of shape (batch, degree of freedom = 3) corresponding to the du/dt = [dx/dt, dy/dt, dz/dt].
        """
        del t
        Dx = self.params["s"] * (u[:, 1] - u[:, 0])
        Dy = u[:, 0] * (self.params["r"] - u[:, 2]) - u[:, 1]
        Dz = u[:, 0] * u[:, 1] - self.params["b"] * u[:, 2]
        return torch.stack([Dx, Dy, Dz], dim=-1)


class VanderPol(AbstractDynamics):
    """Van der Pol oscillator.

    The Van der Pol oscillator is a harmonic oscillator with a nonlinear damping term.
    For positive values of the coefficient mu, the system displays a limit cycle behavior.
    Large positive values of mu leads to a stiff differential equation as well.
    The system is given by the equation:
    dx/dt = v, dv/dt = mu(1-x^2)v-x

    Attributes:
        dof: Degree of freedom of the system. Equal to 2.
        params: torch.nn.ParameterDict of the coefficient mu of the dynamics.
    """

    dof: int = 2

    def __init__(self, mu: float = 1000.0):
        """Initializes the class.

        Args:
            mu: Float value for the coefficient mu.
        """
        super().__init__()
        param_dict = {"mu": mu}
        self._register_params(param_dict)

    def forward(
        self, t: Float[Tensor, ""], u: Float[Tensor, "batch {self.dof}"]
    ) -> Float[Tensor, "batch {self.dof}"]:
        """Dynamics of the system.

        Args:
            t: A scalar Tensor of shape (,) corresponding to time.
            u: Tensor of shape (batch, degree of freedom = 2) corresponding to the state u = [x, v].

        Returns:
            dudt: Tensor of shape (batch, degree of freedom = 2) corresponding to the du/dt = [dx/dt, dv/dt].
        """
        del t
        Dx = u[:, 1]
        Dv = self.params["mu"] * (1 - u[:, 0] ** 2) * u[:, 1] - u[:, 0]
        return torch.stack((Dx, Dv), dim=-1)


class RobertsonChemReaction(AbstractDynamics):
    """Robertson's chemical reaction system.

    Robertson's chemical reaction system is a canonical example of a system of stiff differential equations.
    The system is given by the equation:
    dy1/dt = -ay1+by2y3, dy2/dt = ay1-by2y3-cy2^2, dy3/dt = cy2^2

    Attributes:
        dof: Degree of freedom of the system. Equal to 3.
        params: torch.nn.ParameterDict of the coefficients a, b, c of the dynamics.
    """

    dof: int = 3

    def __init__(self, a: float = 0.04, b: float = 1.0e4, c: float = 3.0e7):
        """Initializes the class.

        Args:
            a: Float value for the coefficient a.
            b: Float value for the coefficient b.
            c: Float value for the coefficient c.
        """
        super().__init__()
        param_dict = {"a": a, "b": b, "c": c}
        self._register_params(param_dict)

    def forward(
        self, t: Float[Tensor, ""], u: Float[Tensor, "batch {self.dof}"]
    ) -> Float[Tensor, "batch {self.dof}"]:
        """Dynamics of the system.

        Args:
            t: A scalar Tensor of shape (,) corresponding to time.
            u: Tensor of shape (batch, degree of freedom = 3) corresponding to the state u = [y1, y2, y3].

        Returns:
            dudt: Tensor of shape (batch, degree of freedom = 3) corresponding to the du/dt = [dy1/dt, dy2/dt, dy3/dt].
        """
        del t
        Dy1 = -self.params["a"] * u[:, 0] + self.params["b"] * u[:, 1] * u[:, 2]
        Dy3 = self.params["c"] * u[:, 1] ** 2
        Dy2 = -Dy1 - Dy3
        return torch.stack((Dy1, Dy2, Dy3), axis=-1)
