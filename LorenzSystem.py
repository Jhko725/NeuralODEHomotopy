# %%
from typing import Callable

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from node_homotopy.dynamics import LorenzSystem
from node_homotopy.datasets import IVPDataset, make_dynamics_dataset
from node_homotopy.models import FullyConnectedNetwork, Blackbox
from node_homotopy.learners import (
    HomotopyLearner,
    BasicLearner,
    MultipleShootingLearner,
)
from node_homotopy.controllers import make_controller
from node_homotopy.utils import (
    WandbContext,
    LogStepwiseScheduler,
    get_activation_name,
)
from node_homotopy.train import train_homotopy

RANDOM_SEED = 10
ODESOLVE_KWARGS = {"method": "dopri5", "rtol": 1e-7, "atol": 1e-9}
DYNAMICS_PARAMS = {"s": 10.0, "r": 28.0, "b": 8.0 / 3.0}

ACTIVATION = torch.nn.functional.gelu


def make_lorenz_dataset(
    t_end: float,
    dt=0.1,
    noise_strength: float = 0.25,
    relative_to: str | None = None,
) -> IVPDataset:
    dynamics = LorenzSystem(**DYNAMICS_PARAMS)
    dataset = make_dynamics_dataset(
        dynamics,
        u0=torch.as_tensor([1.2, 2.1, 1.7]).view(1, -1),
        tspan=(0, t_end),
        dt=dt,
        **ODESOLVE_KWARGS,
    )
    dataset.add_noise(strength=noise_strength, relative_to=relative_to)
    return dataset


def make_lorenz_batch(
    t_end: float,
    dt: float,
    noise_strength: float,
    relative_to: str | None = None,
    batch_size: int = 10,
) -> IVPDataset:
    dynamics = LorenzSystem(**DYNAMICS_PARAMS)
    torch.manual_seed(10)
    dataset = make_dynamics_dataset(
        dynamics,
        # u0=torch.rand((batch_size, 3)) * 3.0,
        # u0=torch.rand((batch_size, 3)) * torch.tensor([10.0, 10.0, 10.0])
        # -torch.tensor([5.0, 5.0, 0.0]),
        u0=torch.rand((batch_size, 3)) * torch.tensor([20.0, 20.0, 20.0])
        - torch.tensor([10.0, 10.0, 0.0]),
        tspan=(0, t_end),
        dt=dt,
        **ODESOLVE_KWARGS,
    )
    dataset.add_noise(strength=noise_strength, relative_to=relative_to)
    return dataset


def setup_lorenz_experiment(
    t_end: float,
    dt: float = 0.1,
    nodes_per_layer: int = 50,
    noise_strength: float = 0.25,
    lr: float = 0.02,
    activation: Callable = torch.nn.functional.gelu,
    k: float = 0.0,
    alpha: float = 1e-4,
    schedule: LogStepwiseScheduler | None = None,
    experiment_seed: int | None = None,
) -> tuple[IVPDataset, HomotopyLearner]:
    dataset = make_lorenz_dataset(t_end, dt, noise_strength, relative_to=None)

    pl.seed_everything(experiment_seed)
    k = k * torch.ones(dataset.dof)
    controller = make_controller(
        dataset,
        k,
        interp_alg="cubic",
        alpha=alpha,
    )

    surrogate = FullyConnectedNetwork(
        num_nodes=[dataset.dof, nodes_per_layer, nodes_per_layer, dataset.dof],
        activation=activation,
    )
    model = Blackbox(surrogate, true_params=DYNAMICS_PARAMS)
    learner = HomotopyLearner(
        model, controller, homotopy_schedule=schedule, lr=lr
    )
    return dataset, learner


def setup_lorenz_vanilla(
    t_end: float,
    noise_strength: float = 0.25,
    lr: float = 0.03,
    activation: Callable = torch.tanh,
    experiment_seed: int | None = None,
) -> tuple[IVPDataset, BasicLearner]:
    dataset = make_lorenz_dataset(t_end, noise_strength, relative_to=None)

    pl.seed_everything(experiment_seed)

    surrogate = FullyConnectedNetwork(
        num_nodes=[dataset.dof, 50, 50, dataset.dof], activation=activation
    )
    model = Blackbox(surrogate, true_params=DYNAMICS_PARAMS)
    learner = BasicLearner(model, lr=lr)
    return dataset, learner


def setup_lorenz_multishoot(
    t_end: float,
    noise_strength: float = 0.25,
    lr: float = 0.02,
    activation: Callable = torch.nn.functional.gelu,
    n_intervals: float = 5,
    continuity_weight: float = 0.005,
    experiment_seed: int | None = None,
) -> tuple[IVPDataset, HomotopyLearner]:
    dataset = make_lorenz_dataset(t_end, noise_strength, relative_to=None)

    pl.seed_everything(experiment_seed)

    surrogate = FullyConnectedNetwork(
        num_nodes=[dataset.dof, 50, 50, dataset.dof], activation=activation
    )
    model = Blackbox(surrogate, true_params=DYNAMICS_PARAMS)
    learner = MultipleShootingLearner(
        model,
        lr=lr,
        n_intervals=n_intervals,
        dof=dataset.dof,
        continuity_weight=continuity_weight,
    )
    return dataset, learner


# %%
RANDOM_SEED = 10
TRAIN_HPARAMS = {
    "k": 6,
    "lr": 0.02,
    "dl_ratio": 0.6,
    "n_steps": 7,
    "f_ep": 300,
    "t_end": 3.1,
    "dt": 0.1,
    "noise_strength": 0.25,
    "nodes_per_layer": 50,
}
ACTIVATION = torch.nn.functional.gelu
if __name__ == "__main__":
    with WandbContext(
        project="LZ_homotopy_sweep", entity="jhelab", config=TRAIN_HPARAMS
    ) as wc:
        homotopy_schedule = LogStepwiseScheduler(
            wc.config.n_steps, wc.config.dl_ratio, wc.config.f_ep
        )

        dataset, learner = setup_lorenz_experiment(
            t_end=wc.config.t_end,
            dt=wc.config.dt,
            nodes_per_layer=wc.config.nodes_per_layer,
            noise_strength=wc.config.noise_strength,
            k=wc.config.k,
            alpha=0.0001,
            lr=wc.config.lr,
            activation=ACTIVATION,
            schedule=homotopy_schedule,
            experiment_seed=RANDOM_SEED,
        )

        # learner.odesolve_backend = "backward"
        wc.config.update(
            {
                "autodiff_alg": learner.odesolve_backend,
                "random_seed": RANDOM_SEED,
                "activation": get_activation_name(ACTIVATION),
            }
        )

        train_dataloader = DataLoader(
            dataset, batch_size=1, num_workers=8, pin_memory=True
        )

        trainer = train_homotopy(
            learner,
            train_dataloader,
        )
# %%
from pytorch_lightning.callbacks import ModelCheckpoint

RANDOM_SEED = 30  # 10, 20, 30
TRAIN_HPARAMS = {
    "lr": 0.02,
    "n_intervals": 6,
    "continuity_weight": 0.01,
}
if __name__ == "__main__":
    activation = torch.nn.functional.gelu  # torch.relu rbf tanh

    with WandbContext(
        project="LZ_baseline_sweep", entity="jhelab", config=TRAIN_HPARAMS
    ) as wc:
        dataset, learner = setup_lorenz_vanilla(
            t_end=3.1,
            lr=wc.config.lr,
            activation=activation,
            n_intervals=wc.config.n_intervals,
            continuity_weight=wc.config.continuity_weight,
            experiment_seed=RANDOM_SEED,
        )

        wc.config.update(
            {
                "autodiff_alg": learner.odesolve_backend,
                "random_seed": RANDOM_SEED,
                "activation": get_activation_name(activation),
            }
        )

        train_dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=8,
            pin_memory=True,
        )
        checkpoint_every_n_epochs = 1
        trainer = pl.Trainer(
            max_epochs=4000,
            callbacks=[
                ModelCheckpoint(
                    monitor="mse",
                    mode="min",
                    save_top_k=1,
                    filename="best_{epoch}-{step}",
                ),
                ModelCheckpoint(
                    monitor="mse",
                    every_n_epochs=checkpoint_every_n_epochs,
                    save_top_k=-1,
                    filename="{epoch}-{step}",
                ),
            ],
            log_every_n_steps=1,
            auto_select_gpus=True,
            deterministic="warn",
            accelerator="gpu",
            devices=1,
        )
        trainer.fit(learner, train_dataloader)
# %%
from pytorch_lightning.callbacks import ModelCheckpoint

RANDOM_SEED = 30  # 10, 20, 30
TRAIN_HPARAMS = {
    "lr": 0.01,
    "n_intervals": 5,
    "continuity_weight": 0.5,
}
if __name__ == "__main__":
    activation = torch.nn.functional.gelu  # torch.relu rbf tanh

    with WandbContext(
        project="LZMultishoot", entity="jhelab", config=TRAIN_HPARAMS
    ) as wc:
        dataset, learner = setup_lorenz_multishoot(
            t_end=3.1,
            lr=wc.config.lr,
            activation=activation,
            n_intervals=wc.config.n_intervals,
            continuity_weight=wc.config.continuity_weight,
            experiment_seed=RANDOM_SEED,
        )

        wc.config.update(
            {
                "autodiff_alg": learner.odesolve_backend,
                "random_seed": RANDOM_SEED,
                "activation": get_activation_name(activation),
            }
        )

        train_dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=8,
            pin_memory=True,
        )
        checkpoint_every_n_epochs = 200
        trainer = pl.Trainer(
            max_epochs=4000,
            callbacks=[
                ModelCheckpoint(
                    monitor="mse",
                    mode="min",
                    save_top_k=1,
                    filename="best_{epoch}-{step}",
                ),
                ModelCheckpoint(
                    monitor="mse",
                    every_n_epochs=checkpoint_every_n_epochs,
                    save_top_k=-1,
                    filename="{epoch}-{step}",
                ),
            ],
            log_every_n_steps=1,
            auto_select_gpus=True,
            deterministic="warn",
            accelerator="gpu",
            devices=1,
        )
        trainer.fit(learner, train_dataloader)
# %%
