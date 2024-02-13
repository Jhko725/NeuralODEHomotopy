# ruff: noqa: F722
from typing import TypedDict, Callable, Literal

from torch import Tensor
from jaxtyping import Float
from torchdiffeq import odeint, odeint_adjoint

AdjointOption = Literal[
    "backprop", "adjoint", "single_ckpt", "adaptive_ckpt", "symplectic"
]


def select_odeint_alg(adjoint_type: AdjointOption) -> Callable:
    """Selects the ODE solve function depending on the adjoint type, i.e., how to calculate the gradient.

    Five different options are possible: backprop, adjoint, single_ckpt, adaptive_ckpt, symplectic.
    The options backprop and adjoint are supported by torchdiffeq, whereas the other three require installing
    the torch_symplectic_adjoint package from https://github.com/tksmatsubara/symplectic-adjoint-method.

    Args:
        adjoint_type: String value indicating the one of the five valid adjoint options.

    Returns:
        alg: Corresponding algorithm to numerically integrate an ordinary differential equation.
    """
    if adjoint_type == "backprop":
        alg = odeint
    elif adjoint_type == "adjoint":
        alg = odeint_adjoint
    else:
        try:
            from torch_symplectic_adjoint import (
                odeint_onecheckpoint,
                odeint_checkpoint,
                odeint_symplectic_adjoint,
            )

        except ImportError:
            raise ImportError(
                """To use the options [single_ckpt, adaptive_ckpt, symplectic], 
                need to install the torch_symplectic_adjoint package."""
            )

        ALGO_DICT = {
            "single_ckpt": odeint_onecheckpoint,
            "adaptive_ckpt": odeint_checkpoint,
            "symplectic": odeint_symplectic_adjoint,
        }
        alg = ALGO_DICT[adjoint_type]
    return alg


class ODESolveKwargs(TypedDict):
    solver: str
    rtol: float
    atol: float
    adjoint: AdjointOption


def odesolve(
    func: Callable[
        [Float[Tensor, ""], Float[Tensor, "batch dof"]], Float[Tensor, "batch dof"]
    ],
    u0: Float[Tensor, "batch dof"],
    t: Float[Tensor, " time"],
    method: str = "dopri5",
    rtol: float = 1e-7,
    atol: float = 1e-9,
    adjoint: AdjointOption = "backprop",
    **odeint_kwargs
) -> Float[Tensor, "batch dof time"]:
    """Solves the given ordinary differential equation.

    Note that for gradient compuations to work, the ODE must be implemented as a torch.nn.Module.

    Args:
        func: A callable that computes the right-hand-side of the ODE to solve.
        u0: A tensor of initial conditions with the shape (batch, degree of freedom)
        t: A tensor of time points at which the solution values will be computed at. Has shape (time,)
        method: A string denoting the specific ODE solver algorithm to be used. For possible choices, see
                documentations for either torchdiffeq or torch_symplectic_adjoint.
        rtol: Relative tolerance value for the ODE solver. The method must be an adaptive algorithm such as dopri5.
        atol: Absolute tolerance value for the ODE solver. The method must be an adaptive algorithm such as dopri5.
        adjoint: String value indicating the one of the five valid adjoint options. Determines how the gradient will be calculated.
        **odeint_kwargs: Additional arguments to be passed to the underlying odeint function.

    Returns:
        sol: A tensor of the ODE solution trajectories. Has the shape (batch, dof, time).
    """
    odeint = select_odeint_alg(adjoint)

    sol: Float[Tensor, "time batch dof"] = odeint(
        func, u0, t, method=method, rtol=rtol, atol=atol, **odeint_kwargs
    )
    sol = sol.permute(1, 2, 0)
    return sol
