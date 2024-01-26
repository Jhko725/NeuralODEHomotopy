import torch
from torch import nn, Tensor
from more_itertools import pairwise


class _FullyConnectedNetwork(nn.Module):
    def __init__(self, num_nodes: list[int], activation):
        super().__init__()
        self.nodes = list(num_nodes)
        layers = self._build_layers()
        self.hidden_layers = nn.ModuleList(layers[:-1])
        self.output_layer = layers[-1]
        self.activation = activation

    def _build_layers(self):
        return [nn.Linear(n_in, n_out) for n_in, n_out in pairwise(self.nodes)]

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)  # output activation is identity

        return x


def FullyConnectedNetwork(num_nodes: list[int], activation):
    return torch.jit.script(_FullyConnectedNetwork(num_nodes, activation))


class LotkaVolterraUDE(nn.Module):
    def __init__(self, true_params: dict, surrogate: nn.Module):
        super().__init__()
        self.true_params = true_params
        a, d = true_params["a"], true_params["d"]
        self.register_buffer("coeff", torch.tensor([a, -d]))

        self.surrogate = surrogate  # NN with input node 2, output node 2 (maps u = [x, y] -> [-bxy, cxy])

    def forward(self, t, u):
        return torch.addcmul(self.surrogate(u), self.coeff, u)


class ScaledBlackbox(nn.Module):
    def __init__(
        self,
        surrogate: nn.Module,
        scale: tuple[float, float, float],
        true_params: dict,
    ):
        super().__init__()
        self.surrogate = surrogate
        # we make this a class attribute so that we can remember
        # what parameters this is supposed to learn
        self.register_buffer("scale", torch.tensor(scale))
        self.true_params = true_params

    def forward(self, t, u):  # u = (n_s, 3)
        return self.surrogate(u / self.scale) * self.scale


class Blackbox(nn.Module):
    def __init__(
        self,
        surrogate: nn.Module,
        true_params: dict,
    ):
        super().__init__()
        self.surrogate = surrogate
        # we make this a class attribute so that we can remember
        # what parameters this is supposed to learn
        self.true_params = true_params

    def forward(self, t, u):  # u = (n_s, 3)
        return self.surrogate(u)


class SecondOrderBlackbox(nn.Module):
    def __init__(
        self,
        surrogate: nn.Module,
        true_params: dict,
        degree_of_freedom: int = 4,
    ):
        super().__init__()
        self.surrogate = surrogate  # Batch, dof -> Batch, dof/2
        self.true_params = true_params
        self.dof = degree_of_freedom

    def forward(self, t, u):
        return torch.cat((u[..., self.dof // 2 :], self.surrogate(u)), dim=-1)
