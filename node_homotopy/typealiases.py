from typing import Sequence
from pathlib import Path
import torch

# types that can be coerced to a torch.FloatTensor via torch.tensor
Tensorable = float | Sequence[float]

Pathlike = str | Path

Sizelike = tuple | torch.Size
