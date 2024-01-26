import torch
from torch import nn
from pytorch_lightning import LightningModule
import wandb

from .dynamics import AbstractDynamics
from .controllers import Controller
from .odesolve import odesolve
from .utils import StepwiseScheduler


class BasicLearner(LightningModule):
    def __init__(
        self,
        model: AbstractDynamics,
        lr: float = 1e-2,
        odesolve_backend: str = "backprop",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.lr = lr
        self.odesolve_backend = odesolve_backend
        self.loss_ftn = nn.MSELoss(reduction="mean")

    def forward(self, t, u):
        return self.model(t, u)

    def solve(self, t, u0):  #
        u_pred = odesolve(
            self,
            u0,
            t,
            enable_grad=True,
            backend=self.odesolve_backend,
            method="dopri5",
            rtol=1e-7,
            atol=1e-9,
        )
        return u_pred  # (batch, dof, time)

    def training_step(self, batch, batch_idx):  #
        t, u0, u_true = self.unpack_batch(batch, batch_idx)
        u_pred = self.solve(t, u0)
        loss = self.loss_ftn(u_true, u_pred)
        self.log("train_loss", loss)
        return loss

    def unpack_batch(self, batch, batch_idx):
        t = batch["t"][0].float()
        u0 = batch["u0"].float()
        u_true = batch["u"].float()
        return t, u0, u_true

    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
        # return torch.optim.Adam(self.parameters(), lr = self.lr)


class MultipleShootingLearner(BasicLearner):
    def __init__(
        self,
        model: AbstractDynamics,
        lr: float,
        n_intervals: int,
        dof: int,
        continuity_weight: float,
        odesolve_backend: str = "backprop",
    ):
        super().__init__(model=model, lr=lr, odesolve_backend=odesolve_backend)
        self.save_hyperparameters(ignore=["model"])
        self.n_intervals = n_intervals
        self.dof = dof
        self.eps_i = nn.Parameter(torch.zeros(self.n_intervals - 1, self.dof))
        self.continuity_weight = continuity_weight

    def calculate_split_lengths(self, t):
        lens = [len(t_i) for t_i in torch.tensor_split(t, self.n_intervals)]
        return lens

    def on_train_start(self) -> None:
        wandb.define_metric("train_loss", summary="min")
        wandb.define_metric("mse", summary="min")

    def training_step(self, batch, batch_idx):  #
        t, u0, u_true = self.unpack_batch(batch, batch_idx)
        len_split = self.calculate_split_lengths(t)
        u_pred = []
        end_penalty = []
        i_start = 0
        for i, len_ in enumerate(len_split):
            if i == 0:
                u_init = u0

            else:
                u_init = u_true[..., i_start] + self.eps_i[i - 1]

            if i != self.n_intervals - 1:
                i_end = i_start + len_ + 1
                u_interval = self.solve(t[i_start:i_end], u_init)
                u_pred.append(u_interval[..., :-1])
                end_penalty.append(
                    torch.mean(
                        (
                            u_interval[..., -1]
                            - u_true[..., i_end]
                            - self.eps_i[i]
                        )
                        ** 2
                    )
                )
            else:
                i_end = i_start + len_
                u_interval = self.solve(t[i_start:i_end], u_init)
                u_pred.append(u_interval)

            i_start += len_

        u_pred = torch.cat(u_pred, dim=-1)
        train_error = self.loss_ftn(u_true, u_pred)
        end_penalty = torch.mean(torch.stack(end_penalty))
        loss = train_error + self.continuity_weight * end_penalty
        self.log("train_error", train_error, on_step=False, on_epoch=True)
        self.log(
            "continuity_penalty", end_penalty, on_step=False, on_epoch=True
        )
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True
        )

        with torch.no_grad():
            u_pred_noctrl = odesolve(
                self.model,
                u0,
                t,
                enable_grad=False,
                backend=self.odesolve_backend,
                method="dopri5",
                rtol=1e-5,
                atol=1e-5,
            )
            mse = self.loss_ftn(u_true, u_pred_noctrl)
            self.log("mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        return loss


class SteerLearner(BasicLearner):
    def __init__(
        self,
        model: AbstractDynamics,
        lr: float = 1e-2,
        b: float = 0.1,
        odesolve_backend: str = "symplectic_adjoint",
    ):
        super().__init__(model=model, lr=lr, odesolve_backend=odesolve_backend)
        self.save_hyperparameters(ignore=["model"])
        self.b = b

    def perturb_t(self, t):
        dt = t[1:] - t[0:-1]
        rand = torch.rand_like(dt) - 0.5
        t_perturb = 2 * self.b * rand * dt
        t_new = t.clone()
        t_new[1:] += t_perturb
        return t_new

    def training_step(self, batch, batch_idx):  #
        t, u0, u_true = self.unpack_batch(batch, batch_idx)
        t_new = self.perturb_t(t)
        u_pred = self.solve(t_new, u0)
        loss = self.loss_ftn(u_true, u_pred)
        self.log("train_loss", loss)
        return loss

    def unpack_batch(self, batch, batch_idx):
        t = batch["t"][0].float()
        u0 = batch["u0"].float()
        u_true = batch["u"].float()
        return t, u0, u_true

    def configure_optimizers(self):
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )


class SynchronizedLearner(BasicLearner):
    def __init__(
        self,
        model: AbstractDynamics,
        controller: Controller,
        lr: float = 5e-3,
        odesolve_backend: str = "backprop",
    ):
        super().__init__(model=model, lr=lr, odesolve_backend=odesolve_backend)
        self.save_hyperparameters(ignore=["model", "controller"])
        self.controller = controller

    def forward(self, t, u):
        return self.model(t, u) + self.controller(t, u)

    def training_step(self, batch, batch_idx):
        t, u0, u_true = self.unpack_batch(batch, batch_idx)
        u_pred = self.solve(t, u0)
        mse_ctrl = self.loss_ftn(u_true, u_pred)
        k = torch.mean(torch.relu(self.controller.k))
        loss = mse_ctrl + k
        self.log("mse_ctrl", mse_ctrl)
        self.log("k", k)
        self.log("train_loss", loss)

        with torch.no_grad():
            u_pred_noctrl = odesolve(
                self.model,
                u0,
                t,
                enable_grad=False,
                backend=self.odesolve_backend,
                method="dopri5",
                rtol=1e-7,
                atol=1e-9,
            )
            mse = self.loss_ftn(u_true, u_pred_noctrl)
            self.log("mse", mse)

        return loss


class HomotopyLearner(SynchronizedLearner):
    def __init__(
        self,
        model: AbstractDynamics,
        controller: Controller,
        homotopy_schedule: StepwiseScheduler,
        lr: float = 5e-3,
        odesolve_backend: str = "backprop",
    ):
        super().__init__(model, controller, lr, odesolve_backend)
        self.save_hyperparameters(ignore=["model", "controller"])
        self.controller.k.requires_grad = False
        self.scheduler = homotopy_schedule

    @property
    def lambda_(self):
        return self.scheduler(self.current_epoch)

    def forward(self, t, u):
        return self.model(t, u) + self.lambda_ * self.controller(t, u)

    def on_train_start(self) -> None:
        wandb.define_metric("train_loss", summary="min")
        wandb.define_metric("mse", summary="min")

    def training_step(self, batch, batch_idx):
        t, u0, u_true = self.unpack_batch(batch, batch_idx)
        u_pred = self.solve(t, u0)
        loss = self.loss_ftn(u_true, u_pred)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("lambda", self.lambda_, on_step=False, on_epoch=True)

        with torch.no_grad():
            u_pred_noctrl = odesolve(
                self.model,
                u0,
                t,
                enable_grad=False,
                backend=self.odesolve_backend,
                method="dopri5",
                rtol=1e-7,
                atol=1e-9,
            )
            mse = self.loss_ftn(u_true, u_pred_noctrl)
            self.log("mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        return loss
