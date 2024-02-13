# ruff: noqa: F722
from functools import partial

import torch
from torch import nn, Tensor
from jaxtyping import Float, Integer
import lightning
from lightning.pytorch.loggers import WandbLogger
import wandb

from node_homotopy.dynamics import AbstractDynamics
from node_homotopy.synchronization import CouplingTerm, SynchronizedDynamics
from node_homotopy.odesolve import odesolve, AdjointOption
from node_homotopy.schedules import Schedule
from node_homotopy.datasets import DynamicsDatasetBatch


class VanillaTraining(lightning.LightningModule):
    """A class to perform vanilla training (i.e. standard gradient descent training) of NeuralODEs.

    Uses the mean squared loss as the loss function and AdamW as the optimizer.

    Attributes:
        model: AbstractDynamics (e.g. NeuralODE model) to train.
        lr: Learning rate for the optimizer.
        odesolve_kwargs: A dictionary containing the arguments to be passed to node_homotopy.odesolve.odesolve function.
        odesolve: A partial function of node_homotopy.odesolve.odesolve with self.odesolve_kwargs as the fixed argument values.
        loss_function: A callable that computes the loss value.
    """

    def __init__(
        self,
        model: AbstractDynamics,
        lr: float,
        method: str = "dopri5",
        rtol: float = 1e-7,
        atol: float = 1e-9,
        adjoint: AdjointOption = "backprop",
    ):
        """Initializes the class.

        Args:
            model: AbstractDynamics (e.g. NeuralODE model) to train.
            lr: A float value specifying the learning rate for the gradient descent.
            method: String indicating the ODE solver algorithm to be used to integrate the dynamics.
                Must be supported by the torchdiffeq library. Defaults to dopri5.
            rtol: Relative tolerance value to be used by the adaptive ODE solver.
                Defaults to 1e-7.
            atol: Absolute tolerance value to be used by the adaptive ODE solver.
                Defaults to 1e-9.
            adjoint: One of the literal values in AdjointOption specifying how the gradient will be calculated.
                Defaults to backprop, which is backpropagation through the solver internals.
                For more information, see the documentation for node_homotopy.odesolve.select_odeint_alg.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.odesolve_kwargs = {
            "method": method,
            "rtol": rtol,
            "atol": atol,
            "adjoint": adjoint,
        }
        self.odesolve = partial(odesolve, **self.odesolve_kwargs)
        self.log = partial(super().log, on_step=False, on_epoch=True)
        self.loss_function = nn.MSELoss(reduction="mean")

    def forward(
        self, u0: Float[Tensor, "batch {self.model.dof}"], t: Float[Tensor, " time"]
    ) -> Float[Tensor, "batch {self.model.dof} time"]:
        """Defines the forward pass of the training method during model training.

        For VanillaTraining, this is identical to solving the dynamics of self.model.

        Args:
            u0: A Tensor of initial conditions to solve the dynamics with.
                Has shape (batch size, degree of freedom).
            t: A Tensor of time points to obtain the solutions at.
                Has shape (time points,).

        Returns:
            u: A Tensor containing the solved dynamics.
                Has shape (batch size, degree of freedom, time points).
        """
        return self.odesolve(self.model, u0, t)

    def unpack_batch(
        self, batch: DynamicsDatasetBatch
    ) -> tuple[
        Float[Tensor, " time"],
        Float[Tensor, "batch {self.model.dof}"],
        Float[Tensor, "batch {self.model.dof} time"],
    ]:
        return batch["t"][0], batch["u0"], batch["u"]

    def training_step(
        self, batch: DynamicsDatasetBatch, batch_idx: int
    ) -> Float[Tensor, ""]:
        """Defines the computation graph of the forward pass during model training.

        As a side effect, logs some quantities of interest; in this case, the train loss under the name "train_loss".

        Args:
            batch: A batch of training data from torch.utils.data.DataLoader.
            batch_idx: Integer denoting the index of the batch.
                This argument is part of PyTorchLightning's API.

        Returns:
            loss: A scalar Tensor containing the train loss.
        """
        del batch_idx
        t, u0, u_true = self.unpack_batch(batch)
        u_pred = self.forward(u0, t)
        loss = self.loss_function(u_true, u_pred)
        self.log("train_loss", loss)
        self.log(
            "mse", loss, prog_bar=True
        )  # An alias for train_loss for consistency with the other training methods.
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer used during training.

        The function is written to use the AdamW optimizer with learning rate of self.lr.
        This function is also a part of PyTorchLightning's API.

        Returns:
            optimizer: A torch.optim.Optimizer object used for training.
        """
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )

    def on_train_start(self) -> None:
        """Defines the train_loss and mse metrics in wandb, so that the minimum values across the training run is monitored.

        Does not have an effect if wandb is not used to log training.
        """
        if isinstance(self.logger, WandbLogger):
            wandb.define_metric("train_loss", summary="min")
            wandb.define_metric("mse", summary="min")


class HomotopyTraining(VanillaTraining):
    """A class to perform homotopy-based training of NeuralODEs.

    Uses the mean squared loss as the loss function and AdamW as the optimizer.

    Attributes:
        model: AbstractDynamics (e.g. NeuralODE model) to train.
        lr: Learning rate for the optimizer.
        odesolve_kwargs: A dictionary containing the arguments to be passed to node_homotopy.odesolve.odesolve function.
        odesolve: A partial function of node_homotopy.odesolve.odesolve with self.odesolve_kwargs as the fixed argument values.
        loss_function: A callable that computes the loss value.
        model_sync: A SynchronizedDynamics representing the model to be trained, synchronized by a given coupling term.
        schedule: A Schedule specifying how to tune the homotopy parameter during training.
    """

    def __init__(
        self,
        model: AbstractDynamics,
        lr: float,
        coupling_term: CouplingTerm,
        homotopy_schedule: Schedule,
        method: str = "dopri5",
        rtol: float = 1e-7,
        atol: float = 1e-9,
        adjoint: AdjointOption = "backprop",
    ):
        """Initializes the class.

        Args:
            model: AbstractDynamics (e.g. NeuralODE model) to train.
            lr: A float value specifying the learning rate for the gradient descent.
            coupling_term: A CouplingTerm to link the model to some desired data trajectory.
            homotopy_schedule: A Schedule specifying how to decrease the homotopy parameter as a function of training epochs.
            method: String indicating the ODE solver algorithm to be used to integrate the dynamics.
                Must be supported by the torchdiffeq library. Defaults to dopri5.
            rtol: Relative tolerance value to be used by the adaptive ODE solver.
                Defaults to 1e-7.
            atol: Absolute tolerance value to be used by the adaptive ODE solver.
                Defaults to 1e-9.
            adjoint: One of the literal values in AdjointOption specifying how the gradient will be calculated.
                Defaults to backprop, which is backpropagation through the solver internals.
                For more information, see the documentation for node_homotopy.odesolve.select_odeint_alg.
        """
        super().__init__(model, lr, method, rtol, atol, adjoint)
        self.save_hyperparameters(ignore=["model", "coupling_term"])
        self.model_sync = SynchronizedDynamics(self.model, coupling_term)
        self.model_sync.k.requires_grad = False
        self.schedule = homotopy_schedule

    @property
    def lambda_(self) -> float:
        """Returns the value of the homotopy parameter, according to self.schedule.

        Uses the read-only self.current_epoch property, which is part of the lightning.LightningModule API.

        Returns:
            lambda_: The homotopy parameter at the current epoch
        """
        return self.schedule(self.current_epoch)

    def forward(
        self, u0: Float[Tensor, "batch {self.model.dof}"], t: Float[Tensor, " time"]
    ) -> Float[Tensor, "batch {self.model.dof} time"]:
        """Defines the forward pass of the training method during model training.

        For HomotopyTraining, this solves the data-synchronized dynamics of self.model_sync.
        Also note that the output is dependent on the training epoch due to the use of self.lambda_ inside,
        which in turn is determined by self.current_epoch, which is part of PyTorchLightning's LightningModule API.

        Args:
            u0: A Tensor of initial conditions to solve the dynamics with.
                Has shape (batch size, degree of freedom).
            t: A Tensor of time points to obtain the solutions at.
                Has shape (time points,).

        Returns:
            u: A Tensor containing the solved dynamics.
                Has shape (batch size, degree of freedom, time points).
        """
        self.model_sync.scale = self.lambda_
        return self.odesolve(self.model_sync, u0, t)

    def training_step(
        self, batch: DynamicsDatasetBatch, batch_idx: int
    ) -> Float[Tensor, ""]:
        """Defines the computation graph of the forward pass during model training.

        As a side effect, logs some quantities of interest; in this case, the train loss under the name "train_loss",
        the un-synchronized mean-squared error under the name "mse", and the homotopy parameter under the name "lambda".

        Args:
            batch: A batch of training data from torch.utils.data.DataLoader.
            batch_idx: Integer denoting the index of the batch.
                This argument is part of PyTorchLightning's API.

        Returns:
            loss: A scalar Tensor containing the train loss.
        """
        del batch_idx
        t, u0, u_true = self.unpack_batch(batch)
        u_pred = self.forward(u0, t)
        loss = self.loss_function(u_true, u_pred)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("lambda", self.lambda_, on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            u_vanilla = self.odesolve(self.model, u0, t)
            mse = self.loss_function(u_vanilla, u_true)
            self.log("mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        return loss


class MultipleShootingTraining(VanillaTraining):
    """A class to perform multple-shooting training of NeuralODEs.

    Uses the mean squared loss as the loss function and AdamW as the optimizer.

    Attributes:
        model: AbstractDynamics (e.g. NeuralODE model) to train.
        lr: Learning rate for the optimizer.
        odesolve_kwargs: A dictionary containing the arguments to be passed to node_homotopy.odesolve.odesolve function.
        odesolve: A partial function of node_homotopy.odesolve.odesolve with self.odesolve_kwargs as the fixed argument values.
        loss_function: A callable that computes the loss value.
        n_segments: Integer denoting the number of multiple-shooting segments to be used for the training.
        init_cond_corrections: Trainable parameters intended to compensate for the error caused while estimating the initial conditions for
            the multiple-shooting segments. See self.create_segments() for more info.
        continuity_weight: A float denoting how severely the discontinuity between the multiple-shooting segments must be penalized.
    """

    def __init__(
        self,
        model: AbstractDynamics,
        lr: float,
        n_segments: int,
        continuity_weight: float,
        method: str = "dopri5",
        rtol: float = 1e-7,
        atol: float = 1e-9,
        adjoint: AdjointOption = "backprop",
    ):
        """Initializes the class.

        Args:
            model: AbstractDynamics (e.g. NeuralODE model) to train.
            lr: A float value specifying the learning rate for the gradient descent.
            n_segments: Integer denoting the number of multiple-shooting segments to be used for the training.
            continuity_weight: A float denoting how severely the discontinuity between the multiple-shooting segments must be penalized.
            method: String indicating the ODE solver algorithm to be used to integrate the dynamics.
                Must be supported by the torchdiffeq library. Defaults to dopri5.
            rtol: Relative tolerance value to be used by the adaptive ODE solver.
                Defaults to 1e-7.
            atol: Absolute tolerance value to be used by the adaptive ODE solver.
                Defaults to 1e-9.
            adjoint: One of the literal values in AdjointOption specifying how the gradient will be calculated.
                Defaults to backprop, which is backpropagation through the solver internals.
                For more information, see the documentation for node_homotopy.odesolve.select_odeint_alg.
        """
        super().__init__(model, lr, method, rtol, atol, adjoint)
        self.save_hyperparameters(ignore=["model"])
        self.n_segments = n_segments
        self.init_cond_corrections = nn.Parameter(
            torch.zeros(self.model.dof, self.n_segments - 1)
        )  # Assumes that the batch size is 1. Otherwise, the parameter must have shape (batch, dof, n_segments-1)
        self.continuity_weight = continuity_weight

    def get_split_indices(
        self, t: Float[Tensor, " time"]
    ) -> Integer[Tensor, " {self.n_segments}+1"]:
        """Computes the indices that splits the 1D Tensor t into self.n_segments as
        torch.tensor_split would.

        Note the that the implementation is largely inspired from that of np.array_split.

        Args:
            t: Tensor containing the time points to solve the NeuralODE at.
                Has shape (time points, ).

        Returns:
            split_indices: 1D Tensor containing the indices to split the Tensor t at.
                That is, the i-th segment of t is given by the index range [split_indices[i], split_indices[i+1]].
                Has shape (self.n_segments+1, )
        """
        len_per_segment, extras = divmod(len(t), self.n_segments)
        segment_sizes = (
            [0]
            + extras * [len_per_segment + 1]
            + (self.n_segments - extras) * [len_per_segment]
        )  # Trick borrowed from the implementation of numpy.array_split
        split_indices = torch.cumsum(
            torch.tensor(segment_sizes, device=t.device, dtype=torch.long), dim=0
        )
        return split_indices

    def create_segments(
        self,
        t: Float[Tensor, " time"],
        u0: Float[Tensor, "batch {self.model.dof}"],
        u: Float[Tensor, "batch {self.model.dof} time"],
    ) -> tuple[
        list[Float[Tensor, "batch {self.model.dof}"]], list[Float[Tensor, " _time"]]
    ]:
        """Creates the segmented initial conditions and time points for the multiple-shooting forward pass.

        The multiple-shooting initial condition has to be guessed because except for the initial condition of the first segment
        (which is identical to u0, the initial condition of the total trajectory), the initial conditions for the other segments are not known
        prefectly due to noise.
        Therefore, the initial condition for the other trajectories are created by adding a trainable parameter self.init_cond_corrections (initialized to zero, and designed to compensate for the noise)
        to the value of the state u corresponding to the start of each trajectory.

        Args:
            t: Tensor containing the time points to solve the NeuralODE at.
                Has shape (time points, ).
            u0: Tensor of the initial condition to solve the NeuralODE with.
                Has shape (batch size, degree of freedom).
            u: Tensor of the measured states at each time point of t.
                Has shape (batch size, degree of freedom, time points).

        Returns:
            u0_segments: A list of Tensor of initial conditions to solve the dynamics of each segment with.
                Each Tensor in the list has shape (batch size, degree of freedom).
            t_segments: A list of Tensor of time points representing the multiple shooting segments.
                Each Tensor in the list has shape (time points per segment,).
        """
        split_inds = self.get_split_indices(t)
        t_segments = [
            t[split_inds[i] : split_inds[i + 1] + 1] for i in range(self.n_segments)
        ]
        u0_segment_guess: Float[Tensor, "{self.model.dof} {self.n_segments}-1"] = (
            u[..., split_inds[1:-1]] + self.init_cond_corrections
        )

        u0_segments = [u0] + [
            u0_segment_guess[..., i - 1] for i in range(1, self.n_segments)
        ]
        return u0_segments, t_segments

    def join_segments(
        self, u_pred_segments: list[Float[Tensor, "batch {self.model.dof} _time"]]
    ) -> Float[Tensor, "batch {self.model.dof} time"]:
        """Concatenates the segmented multiple shooting trajectory predictions into a single connected trajectory.

        Args:
            u_pred_segments: A list of tensors containing the predicted trajectory for each segmented time interval.
                Each tensor in the list has shape (batch, degree of freedom, length of time interval the segment belongs to).

        Returns:
            u_pred: A Tensor representing the concatenated trajectory prediction.
                Has shape (batch, degree of freedom, time points).
        """
        u_pred = torch.cat(
            [u_pred_i[..., :-1] for u_pred_i in u_pred_segments[:-1]], dim=-1
        )
        u_pred = torch.cat([u_pred, u_pred_segments[-1]], dim=-1)
        return u_pred

    def forward(
        self,
        u0: list[Float[Tensor, "batch {self.model.dof}"]],
        t: list[Float[Tensor, " _time"]],
    ) -> list[Float[Tensor, "batch {self.model.dof} _time"]]:
        """Defines the forward pass of the training method during model training.

        For MultpleShootingTraining, this means solving the NeuralODE given by self.model for each time segments
        (defined by individual elements of list t) via the corresponding initial conditions provided in the list u0.
        One gotcha with this method is that it has a different type signature than that of VanillaTraining or HomotopyTraining,
        as it accepts/returns (a) list(s) of Tensors instead of Tensors themselves.

        Args:
            u0: A list of Tensor of initial conditions to solve the dynamics of each segment with.
                Each Tensor in the list has shape (batch size, degree of freedom).
            t: A list of Tensor of time points representing the multiple shooting segments.
                Each Tensor in the list has shape (time points per segment,).

        Returns:
            u: A list of Tensors containing the solved dynamics per segment.
                Each Tensor in the list Has shape (batch size, degree of freedom, time points per segment).
        """
        u_pred = [self.odesolve(self.model, u0_i, t_i) for u0_i, t_i in zip(u0, t)]
        return u_pred

    def continuity_penalty(
        self,
        u0_segments: list[Float[Tensor, "batch {self.model.dof}"]],
        u_pred_segments: list[Float[Tensor, "batch {self.model.dof} _time"]],
    ) -> Float[Tensor, ""]:
        """Calculates the continuity penalty term for the multiple shooting prediction.

        This is given as the average of the mean squared losses between the last point of the i-1 th segment
        and the first point of the ith trajectory.

        Args:
            u0_segments: A list of Tensors corresponding to the initial values used to generate each multiple shooting segments.
                Each Tensor has the shape (batch, degree of freedom).
            u_pred_segments: A list of tensors containing the predicted trajectory for each segmented time interval.
                Each tensor in the list has shape (batch, degree of freedom, length of time interval the segment belongs to).

        Returns:
            continuity_penalty: A scalar Tensor containing the computed value of the penalty.
        """
        u0_stacked: Float[
            Tensor, "batch {self.model.dof} {self.n_segments}-1"
        ] = torch.stack(u0_segments[1:], dim=-1)
        u_end_stacked: Float[
            Tensor, "batch {self.model.dof} {self.n_segments}-1"
        ] = torch.stack([u_i[..., -1] for u_i in u_pred_segments[:-1]], dim=-1)
        return self.loss_function(u0_stacked, u_end_stacked)

    def training_step(
        self, batch: DynamicsDatasetBatch, batch_idx: int
    ) -> Float[Tensor, ""]:
        """Defines the computation graph of the forward pass during model training.

        As a side effect, logs some quantities of interest; in this case, the train loss under the name "train_loss",
        the non-multple-shooting mean-squared error under the name "mse", the data loss part of the multple-shooting training loss under the name "train_error",
        and the continuity penalty part of the multiple-shooting train loss under the name "continuity_penalty".

        Args:
            batch: A batch of training data from torch.utils.data.DataLoader.
            batch_idx: Integer denoting the index of the batch.
                This argument is part of PyTorchLightning's API.

        Returns:
            loss: A scalar Tensor containing the train loss.
        """
        del batch_idx
        t, u0, u_true = self.unpack_batch(batch)
        u0_segments, t_segments = self.create_segments(t, u0, u_true)
        u_pred_segments = self.forward(u0_segments, t_segments)
        u_pred = self.join_segments(u_pred_segments)

        contin_penalty = self.continuity_penalty(u0_segments, u_pred_segments)
        train_error = self.loss_function(u_true, u_pred)
        loss = train_error + self.continuity_weight * contin_penalty

        self.log("train_error", train_error, on_step=False, on_epoch=True)
        self.log("continuity_penalty", contin_penalty, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        with torch.no_grad():
            u_vanilla = self.odesolve(self.model, u0, t)
            mse = self.loss_function(u_vanilla, u_true)
            self.log("mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        return loss


NeuralODETraining = VanillaTraining | HomotopyTraining | MultipleShootingTraining
