# ruff: noqa: F722
from typing import Sequence, Callable
import os

import numpy as np
import torch
from torch import Tensor
from jaxtyping import Float

# types that can be coerced to a torch.FloatTensor via torch.tensor
Tensorlike = float | Sequence[float]

# Useful when annotating plotting functions that can work with either.
ArrayOrTensor = np.ndarray | torch.Tensor

FilePath = str | bytes | os.PathLike

Sizelike = tuple | torch.Size

ElementwiseFunction = Callable[[Float[Tensor, " *shape"]], Float[Tensor, " *shape"]]
