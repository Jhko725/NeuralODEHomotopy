# ruff: noqa: F722
from __future__ import annotations

from typing_extensions import override
import torch
from torch import Tensor, nn
from jaxtyping import Float

from node_homotopy.dynamics import AbstractDynamics
from node_homotopy.datasets import DynamicsDataset
from node_homotopy.smoothing import (
    CubicSplineSmoothing,
)
from node_homotopy.typealiases import Tensorlike


class CouplingTerm(nn.Module):
    """Class representing a proportional coupling term, used for synchronization.

    Given an interpolant \hat{u}(y) of some target trajectory, the proportional coupling term
    is given by g(t, u) = -k(u-\hat{u}).

    Attributes:
        n_dim: Corresponds to the dimensionality of the interpolated trajectory the coupling is based on.
        k: Strength of the coupling term. Can either be a positive scalar, or a positive definite matrix of size n_dim by n_dim.
        interpolant: An interpolant for the target trajectory to base the control on.
    """

    def __init__(self, interpolant: CubicSplineSmoothing, k: Tensorlike):
        super().__init__()
        self.k = nn.Parameter(torch.as_tensor(k))
        self.interpolant = interpolant

    @property
    def n_dim(self) -> int:
        """Dimensionality of the interpolated trajectory the coupling is based on.

        Returns:
            n_dim: Integer corresponding to the dimensionality of the coupling.
        """
        return self.interpolant.n_dim

    def forward(
        self, t: Float[Tensor, ""], u: Float[Tensor, "batch {self.n_dim}"]
    ) -> Float[Tensor, "batch {self.n_dim}"]:
        """Implementation of the coupling term g(t, u) = -k(u-\hat{u}).

        Args:
            t: A scalar Tensor of shape (,) corresponding to time.
            u: Tensor of shape (batch, dimensionality) corresponding to the state at time t.

        Returns:
            g: Tensor of shape (batch, dimensionality) representing the coupling term at time t.
        """
        u_interp = self.interpolant.forward(t).squeeze(-1)
        return self.k * (u_interp - u)

    @classmethod
    def from_dataset(
        cls,
        dataset: DynamicsDataset,
        k: Tensorlike,
    ) -> CouplingTerm:
        interpolant = CubicSplineSmoothing(dataset.t, dataset.u)
        return CouplingTerm(interpolant, k)


class SynchronizedDynamics(AbstractDynamics):
    """Class representing synchronized dynamics.

    Simply augments the dynamics of self.dynamics with an extra coupling represented by self.coupling.

    Attributes:
        dof: Degree of freedom of the augmented dynamics. Identical to self.dynamics.dof.
        dynamics: A subclass of AbstractDynamics representing the base dynamical system to be synchronized.
        coupling: A CouplingTerm that synchonizes the dynamics to some desired trajectory.
        scale: A keyword only argument denoting a float value to scale the coupling term by.
                Intended use is for homotopy optimization, where the scale starts at 1 and is gradually decreased to 0.
    """

    def __init__(
        self, dynamics: AbstractDynamics, coupling: CouplingTerm, scale: float = 1.0
    ):
        """
        Initializes the class.

        Will throw an error if the dimensionality of the dynamics is incompatible with that of the coupling.

        Args:
            dynamics: A subclass of AbstractDynamics representing the base dynamical system.
            coupling: A CouplingTerm to control the dynamics.
        """
        super().__init__()
        if dynamics.dof != coupling.n_dim:
            raise ValueError(
                f"""The degree of freedom of the dynamics ({dynamics.dof}) is different 
                from the degree of freedom of the coupling term ({coupling.n_dim})."""
            )
        self.dynamics = dynamics
        self.coupling = coupling
        self.scale = scale

    @override
    @property
    def dof(self) -> int:
        return self.dynamics.dof

    @property
    def k(self) -> nn.Parameter:
        """Returns the coupling strength of self.coupling term.

        Returns:
            k: Coupling strength of self.coupling term.
        """
        return self.coupling.k

    def forward(
        self,
        t: Float[Tensor, ""],
        u: Float[Tensor, "batch {self.dof}"],
    ) -> Float[Tensor, "batch {self.dof}"]:
        """Synchronized dynamics of the system.

        Corresponds to an implementation of f(t, u) + g(t, u) for the ODE du/dt = f(t, u) + g(t, u)
        where f is the original dynamics and g is the coupling term.
        A typical form of the coupling term is a proportional control term of the form: g(t, u) = -k(u-\hat{u}),
        where \hat{u} is the interpolated trajectory to bind the original dynamics to.

        Args:
            t: A scalar Tensor of shape (,) corresponding to time.
            u: Tensor of shape (batch, degree of freedom) corresponding to the state at time t.

        Returns:
            dudt: Tensor of shape (batch, degree of freedom) corresponding to the time derivative of the state at time t.
        """
        return self.dynamics(t, u) + self.scale * self.coupling(t, u)
