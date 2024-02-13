# ruff: noqa: F722
from typing import Callable, TypedDict

import torch
from torch import nn, Tensor
from jaxtyping import Float
from more_itertools import pairwise

from .typealiases import ElementwiseFunction
from .dynamics import AbstractDynamics


def build_linear_layers(nodes: list[int]) -> list[nn.Linear]:
    """Given a list containing the number of nodes per layer, construct a list containing the corrsponding torch.nn.Linear layers.

    Args:
        nodes:  A list containing the number of nodes per layer.

    Returns:
        layers: A list containing the corresponding Linear layers.
    """
    return [nn.Linear(n_in, n_out) for n_in, n_out in pairwise(nodes)]


class FullyConnectedNetwork(nn.Module):
    """A class representing a fully connected neural network.

    The output layer activation is set to be identity.

    Attributes:
        nodes: A list containing the number of nodes for each layer.
        hidden_layers: A torch.nn.ModuleList containing all layers but the output layer
        output_layer: A torch.nn.Linear layer corresponding to the last layer of the neural network.
        activation: The activation function used inbetween the layers.
            Note that this activation is not applied to the results of the output_layer.
            Instead, the output activation is fixed to be the identity function.
        input_dim: Number of nodes in the input layer.
        output_dim: Number of nodes in the output layer.
    """

    def __init__(self, nodes: list[int], activation: ElementwiseFunction):
        """Initializes the class.

        Args:
            nodes: A list of integers corresponding to the number of nodes per layer.
            activation: A PyTorch compatible function, used as the activation function of the neural network.
        """
        super().__init__()
        self.nodes = list(nodes)
        layers = build_linear_layers(self.nodes)
        self.hidden_layers = nn.ModuleList(layers[:-1])
        self.output_layer = layers[-1]
        self.activation = activation

    @property
    def input_dim(self) -> int:
        """Returns the input dimension of the network, i.e. the number of nodes in the input layer.

        Returns:
            input_dim: An integer corresponding to the number of nodes in the input layer.
        """
        return self.nodes[0]

    @property
    def output_dim(self) -> int:
        """Returns the output dimension of the network, i.e. the number of nodes in the output layer.

        Returns:
            output_dim: An integer corresponding to the number of nodes in the output layer.
        """
        return self.nodes[-1]

    def forward(
        self, x: Float[Tensor, "*batch {self.input_dim}"]
    ) -> Float[Tensor, "*batch {self.output_dim}"]:
        """Compute the forward pass of the network.

        Args:
            x: Tensor corresponding to the (batched) input to the neural network.

        Returns:
            y: Tensor corresponding to the (batched) output of the neural network.
        """
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)  # output activation is identity

        return x


class Blackbox(AbstractDynamics):
    """A class representing a black-box NeuralODE, which considers the entire equation unknown and
    models it using a neural network.

    Attributes:
        surrogate: Neural network used to model the unknown differential equation.
        dof: Integer corresponding to the degree of freedom of the dynamics.
        params: Inherited from the parent class, but is irrelevant here. Is fixed to None.
    """

    def __init__(
        self,
        surrogate: FullyConnectedNetwork,
    ):
        """Initializes the class.

        Args:
            surrogate: A FullyConnectedNetwork used to model the unknown differential equation.
                The input and output dimensions of surrogate must be the same, as both are equal to the degree of freedom of the neuralode.
        Raises:
            ValueError: The input and output dimensions of the surrogate are different.
        """
        super().__init__()
        if surrogate.input_dim != surrogate.output_dim:
            raise ValueError(
                """The input and output dimensions of the neural network must be equal, 
                as they correspond to the degree of freedom of the neuralode dynamics."""
            )
        self.surrogate = surrogate

    @property
    def dof(self) -> int:
        """Degree of freedom of the NeuralODE.

        Identical to input and output dimensions of self.surrogate.

        Returns:
            dof: Integer corresponding to the degree of freedom.
        """
        return self.surrogate.input_dim

    def forward(
        self, t: Float[Tensor, ""], u: Float[Tensor, "batch {self.dof}"]
    ) -> Float[Tensor, "batch {self.dof}"]:
        """Compute the forward pass of the black-box NeuralODE

        Args:
            t: A scalar Tensor of shape (,) corresponding to time.
            u: Tensor of shape (batch, degree of freedom) corresponding to the state u.

        Returns:
            dudt: Tensor of shape (batch, degree of freedom) corresponding to the du/dt.
        """
        del t
        return self.surrogate(u)


class SecondOrderBlackbox(AbstractDynamics):
    """A class representing a second-order NeuralODE.

    As discussed in N. Gruver et al., “Deconstructing the Inductive Biases of Hamiltonian Neural Networks,” ICLR 2021.,
    baking second-order dynamics into the NeuralODE makes them very effective in modeling physical systems, which are usually second order in nature.

    Attributes:
        surrogate: Neural network used to model the unknown differential equation.
        dof: Integer corresponding to the degree of freedom of the dynamics.
        params: Inherited from the parent class, but is irrelevant here. Is fixed to None.
    """

    def __init__(
        self,
        surrogate: FullyConnectedNetwork,
    ):
        """Initializes the class.

        Args:
            surrogate: A FullyConnectedNetwork used to model the unknown differential equation.
                The input dimension of the surrogate must be half of its output dimension.
        Raises:
            ValueError: The output dimension of the surrogate is not equal to the input dimension doubled.
        """
        super().__init__()
        if surrogate.input_dim != surrogate.output_dim * 2:
            raise ValueError(
                """The input dimension of the neural network, which is equal to the degree of freedom of the dynamics of the system,
                must be twice the output dimension of the neural network"""
            )
        self.surrogate = surrogate

    @property
    def dof(self) -> int:
        """Degree of freedom of the NeuralODE.

        Identical to input dimension or the 2*output dimension of self.surrogate.

        Returns:
            dof: Integer corresponding to the degree of freedom.
        """
        return self.surrogate.input_dim

    def forward(
        self, t: Float[Tensor, ""], u: Float[Tensor, "batch {self.dof}"]
    ) -> Float[Tensor, "batch {self.dof}"]:
        """Compute the forward pass of the second-order NeuralODE

        Args:
            t: A scalar Tensor of shape (,) corresponding to time.
            u: Tensor of shape (batch, degree of freedom) corresponding to the state u = [x, v=dx/dt],
                where both x and v have the shape (batch, degree of freedom/2)

        Returns:
            dudt: Tensor of shape (batch, degree of freedom) corresponding to du/dt = [dx/dt=v, dv/dt].
        """
        del t
        return torch.cat((u[..., self.dof // 2 :], self.surrogate(u)), dim=-1)


class PartialLotkaVolterraCoeffs(TypedDict):
    a: float
    d: float


class LotkaVolterraGrayBox(AbstractDynamics):
    """A class representing a Gray-box NeuralODE or a Universal Differential Equation for the Lotka-Volterra system,
    as discussed in C. Rackauckas et al., Universal Differential Equations for Scientific Machine Learning, arXiv.2001.04385 (2021).

    The model is given by [dx/dt; dy/dt] = [ax; -dy] + U([x,y]), where the neural network U has both input and output dimensions of 2
    and serves as proxy for the term [-bxy; cxy]

    Attributes:
        surrogate: Neural network used to model the unknown portion of the dynamics.
        dof: Degree of freedom of the dynamics. Is equal to 2, which is the degree of freedom for the Lotka-Volterra system.
        params: Inherited from the parent class, but is irrelevant here. Is fixed to None.
        known_params: A dictionary containing values of the known parameters a and d.
        coeff: A PyTorch buffer containing the tensor [a, -d], which is used in the forward pass.
    """

    dof: int = 2

    def __init__(
        self,
        surrogate: Callable[
            [Float[Tensor, "batch {self.dof}"]], Float[Tensor, "batch {self.dof}"]
        ],
        known_params: PartialLotkaVolterraCoeffs,
    ):
        """Initializes the class.

        Args:
            surrogate: Neural network used to model the unknown portion of the dynamics.
            known_params: A dictionary containing values of the known parameters a and d.
        """
        super().__init__()
        self.surrogate = surrogate

        self.known_params = known_params
        a, d = known_params["a"], known_params["d"]
        self.register_buffer("coeff", torch.tensor([a, -d]))

    def forward(
        self, t: Float[Tensor, ""], u: Float[Tensor, "batch {self.dof}"]
    ) -> Float[Tensor, "batch {self.dof}"]:
        """Compute the forward pass of the Lotka-Volterra gray-box NeuralODE.

        Args:
            t: A scalar Tensor of shape (,) corresponding to time.
            u: Tensor of shape (batch, degree of freedom=2) corresponding to the state u = [x, y].

        Returns:
            dudt: Tensor of shape (batch, degree of freedom=2) corresponding to du/dt = [dx/dt, dy/dt].
        """
        del t
        return torch.addcmul(self.surrogate(u), self.coeff, u)
