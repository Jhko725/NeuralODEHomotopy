# ruff: noqa: F722
from __future__ import annotations
from typing import Any, TypedDict, Literal

import torch
from torch import Tensor
from jaxtyping import Float

from .dynamics import AbstractDynamics
from .odesolve import odesolve


class DynamicsDatasetBatch(TypedDict):
    t: Float[Tensor, "*batch time"]
    u: Float[Tensor, "*batch dof time"]
    u0: Float[Tensor, "*batch dof"]


class DynamicsDataset(torch.utils.data.Dataset):
    """PyTorch dataset of trajectories generated from dynamical systems.

    The trajectories can be multidimensional, but assumes that all trajectories share
    the same time points.
    The initial conditions for each trajectories can either be directly given or be omitted.
    In the latter case, the initial conditions will be set to the trajectory values at the first time point.

    Attributes:
        t: Tensor of time points for the trajectories. Must be monotonically increasing.
           Does not have to be equispaced. Has shape (N, ); N = number of time points.
        u: Tensor of the state values for the trajectories.
            Has shape (B, M, N); B = number of trajectories, M = dimensionality of the state vector, N = number of time points.
        u0: Tensor of the initial conditions for the trajectories.
            Has shape (B, M); ; B = number of trajectories, M = dimensionality of the state vector
        random_seed: Seed value used to generate the noise values added on to the trajectories
            Not relevant if self.add_noise() is not used. Can be set to None, which corresponds to not using a fixed random seed.
        noise_amplitude: Tensor denoting the amplitude (standard deviation) of the Gaussian noise added to the original trajectory via self.add_noise().
            Not relevant if self.add_noise() is not used.
        n_dim: Integer value denoting the dimensionality of the state vector. If the states are from a fully observed dynamical system,
            this value is equal to the degree of freedom of the dynamics.
    """

    def __init__(
        self,
        t: Float[Tensor, " time"],
        u: Float[Tensor, "samples dof time"],
        u0: Float[Tensor, "samples dof"] | None = None,
    ):
        """Initializes the class by constructing the dataset.

        Args:
            t: Tensor of time points for the trajectories. Must be monotonically increasing.
                Does not have to be equispaced. Has shape (N, ); N = number of time points.
            u: Tensor of the state values for the trajectories.
                Has shape (B, M, N); B = number of trajectories, M = dimensionality of the state vector, N = number of time points.
            u0: Tensor of the initial conditions for the trajectories. Defaults to None, in which case, it is set to the value of u
                corresponding to the first time point of t. Has shape (B, M); B = number of trajectories, M = dimensionality of the state vector
        """
        super().__init__()
        self.t = Tensor(t)
        self.u = Tensor(u)
        self.u0 = Tensor(u0) if u0 is not None else self.u[..., 0]
        self.random_seed: int | None = None
        self.noise_amplitude: Float[Tensor, ""] | Float[
            Tensor, "samples dof 1"
        ] = torch.tensor(0.0)

    @property
    def n_dim(self) -> int:
        """Dimensionality of the state vector.

        Returns:
            n_dim: Integer corresponding to the dimensionality of the state vector.
        """
        return self.u0.shape[1]

    def __len__(self) -> int:
        """Corresponds to the number of samples in the dataset: i.e., the number of trajectories.

        Returns:
            n_samples: Integer corresponding to the number of trajectories in the dataset.
        """
        return self.u0.shape[0]

    def __getitem__(self, idx: Any) -> DynamicsDatasetBatch:
        """Fetches the trajectories in the dataset corresponding to idx.

        Args:
            idx: Any object that can index into a PyTorch Tensor

        Returns:
            sample: Dictionary containing the trajectories corresponding to idx.
                The dictionary has three fields: t, u, and u0 with shapes (N, ), (*, M, N), and (*, M)
                where N = number of time points, M = dimensionality of the state vector, and
                * can be zero or one dimension corresponding to number of trajectories in the sample.
        """
        sample = {"t": self.t, "u": self.u[idx], "u0": self.u0[idx]}
        return sample

    def add_noise(
        self,
        amplitude: float,
        relative_to: Literal["mean"] | None = None,
        random_seed: int | None = 10,
    ) -> None:
        """Adds Gaussian noise to the trajectories in the dataset.

        Does not add noise to the initial condition values. Calling this function modifies the self.random_seed and self.noise_amplitude attributes.

        Args:
            amplitude: Determines the standard deviation of the Gaussian noise to be added.
                If relative_to = None, standard deviation = amplitude for all sample and state dimension.
                If relative_to = "mean", standard deviation = (mean of the trajectory over time) * amplitude, for each sample and state dimension.
        """
        if random_seed is not None:
            self.random_seed = random_seed
            torch.manual_seed(self.random_seed)

        match relative_to:
            case None:
                self.noise_amplitude = torch.tensor(amplitude)
            case "mean":
                mean = torch.mean(self.u, dim=-1, keepdim=True)
                self.noise_amplitude = amplitude * mean
            case _:
                raise ValueError(
                    "Unsupported value for relative to: currently supports None or mean"
                )

        noise = self.noise_amplitude * torch.normal(
            mean=0.0, std=torch.ones_like(self.u)
        )
        self.u += noise

    @classmethod
    def from_dynamics(
        cls,
        dynamics: AbstractDynamics,
        u0: Float[Tensor, "samples n_dims"],
        t: Float[Tensor, " time"],
        **odesolve_kwargs,
    ) -> DynamicsDataset:
        """Constructs the dataset from a given AbstractDynamics.

        The given dynamics is solved using the initial condition u0 and the time points t, then the result is packaged into a DynamicsDataset.

        Args:
            dynamics: AbstractDynamics to create the dataset from.
            u0: Tensor of initial condition to solve the dynamics with.
            t: Tensor of time points to solve the dynamics at.
            **odesolve_kwargs: Keyword arguments to be passed to node_homotopy.odesolve.odesolve.

        Returns:
            dataset: DynamicsDataset containing the corresponding solution to the dynamics.
        """
        if dynamics.dof != u0.shape[1]:
            raise ValueError(
                "The dimensionality of the initial condition tensor u0 is different from the degree of freedom of the dynamics"
            )
        with torch.no_grad():
            u = odesolve(dynamics, u0, t, **odesolve_kwargs)

        return cls(t, u, u0)
