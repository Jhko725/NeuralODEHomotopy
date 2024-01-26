# %%
from typing import Callable

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from node_homotopy.datasets import IVPDataset, SchmidtDataset
from node_homotopy.models import (
    FullyConnectedNetwork,
    Blackbox,
    SecondOrderBlackbox,
)
from node_homotopy.learners import (
    HomotopyLearner,
    BasicLearner,
    MultipleShootingLearner,
)
from node_homotopy.controllers import make_controller
from node_homotopy.train import train_homotopy
from node_homotopy.utils import (
    WandbContext,
    LogStepwiseScheduler,
    get_activation_name,
)

SCHMIDT_FILEPATH = "./data/invar_datasets/real_double_pend_h_1.txt"
ACTIVATION = torch.nn.functional.gelu


def make_doublependulum_dataset(filepath, time_length):
    schmidt_dataset = SchmidtDataset(filepath)
    dataset = schmidt_dataset.subset_by_batch(
        batch_idx=0
    )  # dataset only containing the 0-th trajectory
    dataset = dataset.subset_by_time(time_idx=slice(0, time_length))
    return dataset


def setup_double_pend_experiment(
    time_length: int,
    lr: float = 0.01,
    model_type: str = "blackbox",
    nodes_per_layer: int = 50,
    activation: Callable = torch.nn.functional.gelu,
    k: float = 0.0,
    alpha: float = 0.0,
    schedule: LogStepwiseScheduler | None = None,
    experiment_seed: int | None = None,
) -> tuple[IVPDataset, HomotopyLearner]:
    dataset = make_doublependulum_dataset(SCHMIDT_FILEPATH, time_length)
    pl.seed_everything(experiment_seed)
    k = k * torch.ones(dataset.dof)
    controller = make_controller(dataset, k, interp_alg="cubic", alpha=alpha)

    match model_type:
        case "blackbox":
            surrogate = FullyConnectedNetwork(
                num_nodes=[
                    dataset.dof,
                    nodes_per_layer,
                    nodes_per_layer,
                    dataset.dof,
                ],
                activation=activation,
            )
            model = Blackbox(surrogate, {"filepath": SCHMIDT_FILEPATH})
        case "secondorder":
            surrogate = FullyConnectedNetwork(
                num_nodes=[
                    dataset.dof,
                    nodes_per_layer,
                    nodes_per_layer,
                    dataset.dof // 2,
                ],
                activation=activation,
            )
            model = SecondOrderBlackbox(
                surrogate, {"filepath": SCHMIDT_FILEPATH}, dataset.dof
            )

    learner = HomotopyLearner(model, controller, schedule, lr=lr)
    return dataset, learner


def setup_double_pend_vanilla(
    time_length: int,
    lr: float = 0.01,
    model_type: str = "blackbox",
    nodes_per_layer: int = 50,
    activation: Callable = torch.nn.functional.gelu,
    experiment_seed: int | None = None,
) -> tuple[IVPDataset, BasicLearner]:
    dataset = make_doublependulum_dataset(SCHMIDT_FILEPATH, time_length)

    pl.seed_everything(
        experiment_seed
    )  # For reproducibility; dataset is seeded separately

    match model_type:
        case "blackbox":
            surrogate = FullyConnectedNetwork(
                num_nodes=[
                    dataset.dof,
                    nodes_per_layer,
                    nodes_per_layer,
                    dataset.dof,
                ],
                activation=activation,
            )
            model = Blackbox(surrogate, {"filepath": SCHMIDT_FILEPATH})
        case "secondorder":
            surrogate = FullyConnectedNetwork(
                num_nodes=[
                    dataset.dof,
                    nodes_per_layer,
                    nodes_per_layer,
                    dataset.dof // 2,
                ],
                activation=activation,
            )
            model = SecondOrderBlackbox(
                surrogate, {"filepath": SCHMIDT_FILEPATH}, dataset.dof
            )

    learner = BasicLearner(model, lr=lr)
    return dataset, learner


def setup_double_pend_multishoot(
    time_length: int,
    nodes_per_layer: int = 50,
    lr: float = 0.01,
    model_type: str = "blackbox",
    activation: Callable = torch.nn.functional.gelu,
    n_intervals: float = 5,
    continuity_weight: float = 100.0,
    experiment_seed: int | None = None,
) -> tuple[IVPDataset, BasicLearner]:
    dataset = make_doublependulum_dataset(SCHMIDT_FILEPATH, time_length)
    pl.seed_everything(experiment_seed)

    match model_type:
        case "blackbox":
            surrogate = FullyConnectedNetwork(
                num_nodes=[
                    dataset.dof,
                    nodes_per_layer,
                    nodes_per_layer,
                    dataset.dof,
                ],
                activation=activation,
            )
            model = Blackbox(surrogate, {"filepath": SCHMIDT_FILEPATH})
        case "secondorder":
            surrogate = FullyConnectedNetwork(
                num_nodes=[
                    dataset.dof,
                    nodes_per_layer,
                    nodes_per_layer,
                    dataset.dof // 2,
                ],
                activation=activation,
            )
            model = SecondOrderBlackbox(
                surrogate, {"filepath": SCHMIDT_FILEPATH}, dataset.dof
            )

    learner = MultipleShootingLearner(
        model,
        lr=lr,
        n_intervals=n_intervals,
        dof=4,
        continuity_weight=continuity_weight,
    )
    return dataset, learner


# %%
RANDOM_SEED = 30  # 10, 20, 30
TRAIN_HPARAMS = {
    "k": 10,
    "lr": 0.05,
    "dl_ratio": 0.6,
    "n_steps": 5,
    "f_ep": 300,
    "model_type": "secondorder",
}

if __name__ == "__main__":
    with WandbContext(
        project="DP_homotopy_second_sweep",
        entity="jhelab",
        config=TRAIN_HPARAMS,
    ) as wc:
        homotopy_schedule = LogStepwiseScheduler(
            wc.config.n_steps, wc.config.dl_ratio, wc.config.f_ep
        )

        dataset, learner = setup_double_pend_experiment(
            time_length=100,
            k=wc.config.k,
            alpha=0.0,
            model_type=wc.config.model_type,
            lr=wc.config.lr,
            activation=ACTIVATION,
            schedule=homotopy_schedule,
            experiment_seed=RANDOM_SEED,
        )

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
RANDOM_SEED = 10  # 10, 20, 30
TRAIN_HPARAMS = {
    "k": 10,
    "lr": 0.02,
    "dl_ratio": 0.6,
    "n_steps": 6,
    "f_ep": 300,
    "model_type": "blackbox",
}

if __name__ == "__main__":
    with WandbContext(
        project="DP_homotopy_blackbox_sweep",
        entity="jhelab",
        config=TRAIN_HPARAMS,
    ) as wc:
        homotopy_schedule = LogStepwiseScheduler(
            wc.config.n_steps, wc.config.dl_ratio, wc.config.f_ep
        )

        dataset, learner = setup_double_pend_experiment(
            time_length=100,
            k=wc.config.k,
            alpha=0.0,
            model_type=wc.config.model_type,
            lr=wc.config.lr,
            activation=ACTIVATION,
            schedule=homotopy_schedule,
            experiment_seed=RANDOM_SEED,
        )

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
RANDOM_SEED = 20  # 10, 20, 30
TRAIN_HPARAMS = {
    "k": 10,
    "lr": 0.02,
    "dl_ratio": 0.6,
    "n_steps": 6,
    "f_ep": 300,
    "model_type": "blackbox",
}

if __name__ == "__main__":
    with WandbContext(
        project="DP_homotopy_blackbox_sweep",
        entity="jhelab",
        config=TRAIN_HPARAMS,
    ) as wc:
        homotopy_schedule = LogStepwiseScheduler(
            wc.config.n_steps, wc.config.dl_ratio, wc.config.f_ep
        )

        dataset, learner = setup_double_pend_experiment(
            time_length=100,
            k=wc.config.k,
            alpha=0.0,
            model_type=wc.config.model_type,
            lr=wc.config.lr,
            activation=ACTIVATION,
            schedule=homotopy_schedule,
            experiment_seed=RANDOM_SEED,
        )

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
RANDOM_SEED = 30  # 10, 20, 30
TRAIN_HPARAMS = {
    "k": 10,
    "lr": 0.02,
    "dl_ratio": 0.6,
    "n_steps": 6,
    "f_ep": 300,
    "model_type": "blackbox",
}

if __name__ == "__main__":
    with WandbContext(
        project="DP_homotopy_blackbox_sweep",
        entity="jhelab",
        config=TRAIN_HPARAMS,
    ) as wc:
        homotopy_schedule = LogStepwiseScheduler(
            wc.config.n_steps, wc.config.dl_ratio, wc.config.f_ep
        )

        dataset, learner = setup_double_pend_experiment(
            time_length=100,
            k=wc.config.k,
            alpha=0.0,
            model_type=wc.config.model_type,
            lr=wc.config.lr,
            activation=ACTIVATION,
            schedule=homotopy_schedule,
            experiment_seed=RANDOM_SEED,
        )

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

RANDOM_SEED = 20  # 10, 20, 30
CONTROL_HPARAMS = {"lr": 0.02, "model_type": "secondorder"}

if __name__ == "__main__":
    activation = torch.nn.functional.gelu  # torch.relu rbf

    with WandbContext(
        project="DP_loss_function", entity="jhelab", config=CONTROL_HPARAMS
    ) as wc:
        dataset, learner = setup_double_pend_vanilla(
            time_length=100,
            lr=wc.config.lr,
            model_type=wc.config.model_type,
            activation=activation,
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
                    monitor="train_loss",
                    mode="min",
                    save_top_k=1,
                    filename="best_{epoch}-{step}",
                ),
                ModelCheckpoint(
                    monitor="train_loss",
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

RANDOM_SEED = 10  # 10, 20, 30
CONTROL_HPARAMS = {
    "num_nodes": 50,
    "lr": 0.02,
    "model_type": "blackbox",
    "n_intervals": 5,
    "continuity_weight": 0.001,
}

if __name__ == "__main__":
    activation = torch.nn.functional.gelu  # torch.relu rbf tanh silu gelu

    with WandbContext(
        project="DP_multi_black_sweep", entity="jhelab", config=CONTROL_HPARAMS
    ) as wc:
        dataset, learner = setup_double_pend_multishoot(
            time_length=100,
            nodes_per_layer=wc.config.num_nodes,
            model_type=wc.config.model_type,
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
                    save_top_k=-1,
                    filename="best_{epoch}-{step}",
                ),
            ],
            log_every_n_steps=1,
            deterministic="warn",
            accelerator="gpu",
            devices=1,
        )
        trainer.fit(learner, train_dataloader)

# %%
