from typing import Callable

import torch
from torch import Tensor
from torchdiffeq import odeint, odeint_adjoint
from torch_symplectic_adjoint import (
    odeint_onecheckpoint,
    odeint_checkpoint,
    odeint_symplectic_adjoint,
)

ODEINT_ALGORITHMS = {
    "backprop": odeint,
    "adjoint": odeint_adjoint,
    "single_checkpoint_adjoint": odeint_onecheckpoint,
    "adaptive_checkpoint_adjoint": odeint_checkpoint,
    "symplectic_adjoint": odeint_symplectic_adjoint,
}


def select_odeint_alg(name: str | None = None):
    alg = odeint if name is None else ODEINT_ALGORITHMS[name]
    return alg


def odesolve(
    func,
    u0: Tensor,
    t: Tensor,
    enable_grad: bool = False,
    method: str = "dopri5",
    rtol: float = 1e-7,
    atol: float = 1e-9,
    backend: str = "backprop",
    **odeint_kwargs
):
    odeint = select_odeint_alg(backend)

    with torch.set_grad_enabled(enable_grad):
        sol = odeint(
            func, u0, t, method=method, rtol=rtol, atol=atol, **odeint_kwargs
        )  # (time, samples, dof)
        sol = sol.permute(1, 2, 0)  # (samples, dof, time)
    return sol
