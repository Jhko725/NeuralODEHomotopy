from __future__ import annotations
from pathlib import Path
from typing import Sequence
from functools import reduce
from operator import add

import numpy as np
import torch
from torch import Tensor

from .dynamics import AbstractDynamics


class IVPDataset(torch.utils.data.Dataset):
    def __init__(self, t: Tensor, u0: Tensor, u: Tensor):
        super().__init__()
        self.t = Tensor(t)  # (time)
        self.u0 = Tensor(u0)  # (samples, dof)
        self.u = Tensor(u)  # (samples, dof, time)
        self.random_seed = None
        self.noise_amplitude = 0.0

    def __len__(self) -> int:
        return self.u0.size(0)  # array.shape[0]

    @property
    def dof(self) -> int:
        return self.u0.size(1)

    def __getitem__(self, idx) -> dict:
        sample = {"t": self.t, "u0": self.u0[idx], "u": self.u[idx]}
        return sample

    def add_noise(
        self,
        strength: float,
        relative_to: str | None = None,
        random_seed: int | None = 10,
    ) -> None:
        if random_seed is not None:
            self.random_seed = random_seed
            torch.manual_seed(self.random_seed)

        noise = strength * torch.normal(mean=0.0, std=torch.ones_like(self.u))
        self.noise_amplitude = strength

        match relative_to:
            case None:
                pass
            case "mean":
                mean = torch.mean(self.u, dim=-1).unsqueeze(-1)
                noise *= mean
                self.noise_amplitude = self.noise_amplitude * mean
            case _:
                raise ValueError(
                    "Unsupported value for relative to: currently supports None or mean"
                )
        self.u += noise

    def subset_by_batch(self, batch_idx) -> IVPDataset:
        t_subset = self.t if len(self.t.size()) == 1 else self.t[batch_idx]
        u0_subset = self.u0[batch_idx]
        u_subset = self.u[batch_idx]

        match batch_idx:
            case int:
                u0_subset = u0_subset.unsqueeze(0)
                u_subset = u_subset.unsqueeze(0)

        return IVPDataset(t_subset, u0_subset, u_subset)

    def subset_by_time(self, time_idx) -> IVPDataset:
        t_subset = self.t[..., time_idx]
        u_subset = self.u[..., time_idx]
        return IVPDataset(t_subset, self.u0, u_subset)


def make_dynamics_dataset(
    dynamics: AbstractDynamics,
    u0: Sequence,
    tspan: tuple[float, float],
    dt: float,
    **odesolve_kwargs,
) -> IVPDataset:
    u0 = torch.as_tensor(u0)
    if len(u0.size()) == 1:
        u0 = u0.view(1, -1)
    t = torch.arange(*tspan, dt)
    u = dynamics.solve(u0, t, enable_grad=False, **odesolve_kwargs)
    return IVPDataset(t, u0, u)


class SchmidtDataset(IVPDataset):
    def __init__(self, filepath: Path | str, segment_length: int = 300):
        t, u = read_schmidt_data(filepath, segment_length=segment_length)
        u0 = u[..., 0]
        super().__init__(t, u0, u)

    @property
    def dof(self):
        return self.u0.size(1)


def read_schmidt_data(
    filepath: Path | str, segment_length: int
) -> tuple[np.ndarray, np.ndarray]:
    raw_data = np.genfromtxt(Path(filepath), delimiter=" ", skip_header=1)
    raw_data = raw_data[:, 0:6]  # Discard unneeded columns
    trajs = _split_data_by_id(raw_data)
    segments = _segment_trajectories(trajs, segment_length)
    stacked = np.stack(segments, axis=0)
    # (sample(1), num_data(819), (time, theta1, theta2, w1, w2) = 5)
    # (sample(2), num_data(701), (time, theta1, theta2, w1, w2) = 5)
    t, u = stacked[..., 0], stacked[..., 1:].transpose(0, 2, 1)
    t = t - np.expand_dims(t[:, 0], axis=-1)
    # t.shape: (sample, min_time) = (2, 701)
    # without transpose ->
    # u.shape: (sample, min_time, dof = 4 = (theta1, theta2, w1, w2))
    # Need to transpose (0, 2, 1) to match the IVPDataset u shape: (sample, dof, time) = (2,4,701)
    return t, u


def _split_data_by_id(raw_data: np.ndarray) -> list[np.ndarray]:
    trial_ids = np.unique(raw_data[:, 0])
    trajectories = [raw_data[raw_data[:, 0] == id, 1:] for id in trial_ids]
    return trajectories


def _segment_trajectories(
    trajectories: list[np.ndarray], segment_length: int
) -> list[np.ndarray]:
    segment_list = reduce(
        add,
        [_segment_trajectory(traj, segment_length) for traj in trajectories],
    )
    return segment_list


def _segment_trajectory(
    trajectory: np.ndarray, segment_length: int
) -> list[np.ndarray]:
    n_segments = len(trajectory) // segment_length
    return np.split(trajectory[: n_segments * segment_length], n_segments)


def _stack_trajectories(trajectories: list[np.ndarray]) -> np.ndarray:
    min_length = np.min([traj.shape[0] for traj in trajectories])
    stacked = np.stack([traj[0:min_length, :] for traj in trajectories], axis=0)
    return stacked
