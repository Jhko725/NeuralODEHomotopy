# %%
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from node_homotopy.dynamics import LotkaVolterra
from node_homotopy.datasets import IVPDataset, make_dynamics_dataset
from node_homotopy.models import (
    FullyConnectedNetwork,
    LotkaVolterraUDE,
    Blackbox,
)
from node_homotopy.learners import (
    BasicLearner,
    HomotopyLearner,
    SteerLearner,
    MultipleShootingLearner,
)
from node_homotopy.controllers import make_controller
from node_homotopy.utils import (
    WandbContext,
    get_activation_name,
    LogStepwiseScheduler,
)
from node_homotopy.train import train_homotopy

ODESOLVE_KWARGS = {"method": "dopri5", "rtol": 1e-7, "atol": 1e-9}
DYNAMICS_PARAMS = {"a": 1.3, "b": 0.9, "c": 0.8, "d": 1.8}


def make_lotka_dataset(
    t_end: float,
    noise_strength: float,
    dt: float = 0.1,
    relative_to: str | None = "mean",
) -> IVPDataset:
    dynamics = LotkaVolterra(**DYNAMICS_PARAMS)
    dataset = make_dynamics_dataset(
        dynamics,
        u0=torch.tensor([0.44249296, 4.6280594]).view(1, -1),
        tspan=(0, t_end),
        dt=dt,
        **ODESOLVE_KWARGS,
    )
    dataset.add_noise(strength=noise_strength, relative_to=relative_to)
    return dataset


def setup_lotka_experiment(
    t_end: float = 6.1,
    dt: float = 0.1,
    noise_strength: float = 0.05,
    model_type: str = "ude",
    nodes_per_layer: int = 32,
    lr: float = 0.01,
    activation: Callable = torch.nn.functional.gelu,
    k: float = 0.0,
    alpha: float = 0.001,
    schedule: LogStepwiseScheduler | None = None,
    experiment_seed: int | None = None,
) -> tuple[IVPDataset, HomotopyLearner]:
    dataset = make_lotka_dataset(t_end, dt=dt, noise_strength=noise_strength)

    pl.seed_everything(
        experiment_seed
    )  # For reproducibility; dataset is seeded separately

    k = k * torch.ones(dataset.dof)
    controller = make_controller(dataset, k, interp_alg="cubic", alpha=alpha)

    match model_type:
        case "ude":
            surrogate = FullyConnectedNetwork(
                num_nodes=[
                    dataset.dof,
                    nodes_per_layer,
                    nodes_per_layer,
                    dataset.dof,
                ],
                activation=activation,
            )
            model = LotkaVolterraUDE(
                {key: DYNAMICS_PARAMS[key] for key in ("a", "d")}, surrogate
            )
        case "blackbox":
            surrogate = FullyConnectedNetwork(
                num_nodes=[dataset.dof, nodes_per_layer, dataset.dof],
                activation=activation,
            )
            model = Blackbox(surrogate, true_params=DYNAMICS_PARAMS)
        case _:
            raise ValueError(
                """Unsupported model type! Must be one of "ude" or "blackbox" """
            )
    model = torch.jit.script(model)
    learner = HomotopyLearner(model, controller, schedule, lr=lr)
    return dataset, learner


def setup_lotka_vanilla(
    t_end: float,
    dt: float = 0.1,
    noise_strength: float = 0.05,
    model_type: str = "ude",
    nodes_per_layer: int = 32,
    lr: float = 0.01,
    activation: Callable = torch.nn.functional.gelu,
    experiment_seed: int | None = None,
) -> tuple[IVPDataset, BasicLearner]:
    dataset = make_lotka_dataset(t_end, noise_strength, dt=dt)

    pl.seed_everything(
        experiment_seed
    )  # For reproducibility; dataset is seeded separately

    match model_type:
        case "ude":
            surrogate = FullyConnectedNetwork(
                num_nodes=[
                    dataset.dof,
                    nodes_per_layer,
                    nodes_per_layer,
                    dataset.dof,
                ],
                activation=activation,
            )
            model = LotkaVolterraUDE(
                {key: DYNAMICS_PARAMS[key] for key in ("a", "d")}, surrogate
            )
        case "blackbox":
            surrogate = FullyConnectedNetwork(
                num_nodes=[dataset.dof, nodes_per_layer, dataset.dof],
                activation=activation,
            )
            model = Blackbox(surrogate, true_params=DYNAMICS_PARAMS)
        case _:
            raise ValueError(
                """Unsupported model type! Must be one of "ude" or "blackbox" """
            )

    learner = BasicLearner(model, lr=lr)
    return dataset, learner


def setup_lotka_steer(
    t_end: float,
    dt: float = 0.1,
    noise_strength: float = 0.05,
    model_type: str = "ude",
    nodes_per_layer: int = 5,
    lr: float = 0.01,
    activation: Callable = torch.tanh,
    b: float = 0.1,
    experiment_seed: int | None = None,
) -> tuple[IVPDataset, BasicLearner]:
    dataset = make_lotka_dataset(t_end, dt=dt, noise_strength=noise_strength)

    pl.seed_everything(
        experiment_seed
    )  # For reproducibility; dataset is seeded separately

    match model_type:
        case "ude":
            surrogate = FullyConnectedNetwork(
                num_nodes=[
                    dataset.dof,
                    nodes_per_layer,
                    nodes_per_layer,
                    dataset.dof,
                ],
                activation=activation,
            )
            model = LotkaVolterraUDE(
                {key: DYNAMICS_PARAMS[key] for key in ("a", "d")}, surrogate
            )
        case "blackbox":
            surrogate = FullyConnectedNetwork(
                num_nodes=[dataset.dof, nodes_per_layer, dataset.dof],
                activation=activation,
            )
            model = Blackbox(surrogate, true_params=DYNAMICS_PARAMS)
        case _:
            raise ValueError(
                """Unsupported model type! Must be one of "ude" or "blackbox" """
            )

    learner = SteerLearner(model, lr=lr, b=b)
    return dataset, learner


def setup_lotka_multishoot(
    t_end: float,
    dt: float = 0.1,
    noise_strength: float = 0.05,
    model_type: str = "ude",
    nodes_per_layer: int = 32,
    lr: float = 0.01,
    activation: Callable = torch.nn.functional.gelu,
    n_intervals: float = 5,
    continuity_weight: float = 100.0,
    experiment_seed: int | None = None,
) -> tuple[IVPDataset, BasicLearner]:
    dataset = make_lotka_dataset(t_end, dt=dt, noise_strength=noise_strength)

    pl.seed_everything(
        experiment_seed
    )  # For reproducibility; dataset is seeded separately

    match model_type:
        case "ude":
            surrogate = FullyConnectedNetwork(
                num_nodes=[
                    dataset.dof,
                    nodes_per_layer,
                    nodes_per_layer,
                    dataset.dof,
                ],
                activation=activation,
            )
            model = LotkaVolterraUDE(
                {key: DYNAMICS_PARAMS[key] for key in ("a", "d")}, surrogate
            )
        case "blackbox":
            surrogate = FullyConnectedNetwork(
                num_nodes=[dataset.dof, nodes_per_layer, dataset.dof],
                activation=activation,
            )
            model = Blackbox(surrogate, true_params=DYNAMICS_PARAMS)
        case _:
            raise ValueError(
                """Unsupported model type! Must be one of "ude" or "blackbox" """
            )

    learner = MultipleShootingLearner(
        model,
        lr=lr,
        n_intervals=n_intervals,
        dof=2,
        continuity_weight=continuity_weight,
    )
    return dataset, learner


def load_model(
    filepath: str | Path,
    experiment_type: str,
    model_type: str,
    t_end: float = 6.1,
):
    nodes_config = {"ude": 20, "blackbox": 32}
    common_kwargs = {
        "model_type": model_type,
        "nodes_per_layer": nodes_config[model_type],
        "activation": torch.nn.functional.gelu,
    }
    match experiment_type:
        case "vanilla":
            dataset, learner = setup_lotka_vanilla(
                t_end,
                **common_kwargs,
            )
            learner.load_from_checkpoint(filepath, model=learner.model)
        case "homotopy":
            dataset, learner = setup_lotka_experiment(
                t_end,
                **common_kwargs,
            )
            learner.load_from_checkpoint(
                filepath, model=learner.model, controller=learner.controller
            )
        case "multishoot":
            dataset, learner = setup_lotka_multishoot(
                t_end,
                **common_kwargs,
            )
            learner.load_from_checkpoint(filepath, model=learner.model)
    return dataset, learner


# %%
def load_model2(
    filepath: str | Path,
    experiment_type: str,
    model_type: str,
    t_end: float = 6.1,
):
    nodes_config = {"ude": 20, "blackbox": 32}
    common_kwargs = {
        "model_type": model_type,
        "nodes_per_layer": nodes_config[model_type],
        "activation": torch.nn.functional.gelu,
    }
    trainer = pl.Trainer()
    match experiment_type:
        case "vanilla":
            dataset, learner = setup_lotka_vanilla(
                t_end,
                **common_kwargs,
            )
            trainer.fit(learner, filepath)
        case "homotopy":
            dataset, learner = setup_lotka_experiment(
                t_end,
                **common_kwargs,
            )
            trainer.fit(learner, filepath)
        case "multishoot":
            dataset, learner = setup_lotka_multishoot(
                t_end,
                **common_kwargs,
            )
            trainer.fit(learner, filepath)
    return dataset, learner


# %%
RANDOM_SEED = 10  # 20, 30
TRAIN_HPARAMS = {
    "k": 6,
    "lr": 0.05,
    "dl_ratio": 0.6,
    "n_steps": 5,
    "f_ep": 100,
    "num_nodes": 8,
    "model_type": "blackbox",
    "noise_strength": 0.05,
    "t_end": 6.1,
    "dt": 0.1,
}


if __name__ == "__main__":
    activation = torch.nn.functional.gelu  # torch.relu rbf

    with WandbContext(
        project="LV_homotopy_experiment", entity="jhelab", config=TRAIN_HPARAMS
    ) as wc:
        homotopy_schedule = LogStepwiseScheduler(
            wc.config.n_steps, wc.config.dl_ratio, wc.config.f_ep
        )

        dataset, learner = setup_lotka_experiment(
            t_end=wc.config.t_end,
            dt=wc.config.dt,
            noise_strength=wc.config.noise_strength,
            nodes_per_layer=wc.config.num_nodes,
            model_type=wc.config.model_type,
            lr=wc.config.lr,
            activation=activation,
            k=wc.config.k,
            schedule=homotopy_schedule,
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

        trainer = train_homotopy(
            learner,
            train_dataloader,
        )
# %%
from pytorch_lightning.callbacks import ModelCheckpoint

RANDOM_SEED = 10  # 10, 20, 30
CONTROL_HPARAMS = {
    "num_nodes": 16,
    "lr": 0.02,
    "model_type": "blackbox",
    "noise_strength": 0.05,
    "t_end": 6.1,
    "dt": 0.1,
}

if __name__ == "__main__":
    activation = torch.nn.functional.gelu  # torch.relu rbf tanh silu gelu

    with WandbContext(
        project="LV_baseline_experiment",
        entity="jhelab",
        config=CONTROL_HPARAMS,
    ) as wc:
        dataset, learner = setup_lotka_vanilla(
            t_end=wc.config.t_end,
            dt=wc.config.dt,
            noise_strength=wc.config.noise_strength,
            nodes_per_layer=wc.config.num_nodes,
            model_type=wc.config.model_type,
            lr=wc.config.lr,
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

        trainer = pl.Trainer(
            max_epochs=4000,
            callbacks=[
                ModelCheckpoint(
                    monitor="train_loss",
                    mode="min",
                    save_top_k=-1,
                    save_on_train_epoch_end=True,
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
from pytorch_lightning.callbacks import ModelCheckpoint

RANDOM_SEED = 20  # 10, 20, 30
CONTROL_HPARAMS = {
    "num_nodes": 8,
    "lr": 0.02,
    "model_type": "blackbox",
    "noise_strength": 0.05,
    "t_end": 6.1,
    "dt": 0.1,
}

if __name__ == "__main__":
    activation = torch.nn.functional.gelu  # torch.relu rbf tanh silu gelu

    with WandbContext(
        project="LV_baseline_experiment",
        entity="jhelab",
        config=CONTROL_HPARAMS,
    ) as wc:
        dataset, learner = setup_lotka_vanilla(
            t_end=wc.config.t_end,
            dt=wc.config.dt,
            noise_strength=wc.config.noise_strength,
            nodes_per_layer=wc.config.num_nodes,
            model_type=wc.config.model_type,
            lr=wc.config.lr,
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

        trainer = pl.Trainer(
            max_epochs=4000,
            callbacks=[
                ModelCheckpoint(
                    monitor="train_loss",
                    mode="min",
                    save_top_k=-1,
                    save_on_train_epoch_end=True,
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
from pytorch_lightning.callbacks import ModelCheckpoint

RANDOM_SEED = 30  # 10, 20, 30
CONTROL_HPARAMS = {
    "num_nodes": 8,
    "lr": 0.02,
    "model_type": "blackbox",
    "noise_strength": 0.05,
    "t_end": 6.1,
    "dt": 0.1,
}

if __name__ == "__main__":
    activation = torch.nn.functional.gelu  # torch.relu rbf tanh silu gelu

    with WandbContext(
        project="LV_baseline_experiment",
        entity="jhelab",
        config=CONTROL_HPARAMS,
    ) as wc:
        dataset, learner = setup_lotka_vanilla(
            t_end=wc.config.t_end,
            dt=wc.config.dt,
            noise_strength=wc.config.noise_strength,
            nodes_per_layer=wc.config.num_nodes,
            model_type=wc.config.model_type,
            lr=wc.config.lr,
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

        trainer = pl.Trainer(
            max_epochs=4000,
            callbacks=[
                ModelCheckpoint(
                    monitor="train_loss",
                    mode="min",
                    save_top_k=-1,
                    save_on_train_epoch_end=True,
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
from pytorch_lightning.callbacks import ModelCheckpoint

RANDOM_SEED = 10  # 10, 20, 30
CONTROL_HPARAMS = {
    "num_nodes": 20,
    "lr": 0.01,
    "model_type": "ude",
    "noise_strength": 0.05,
    "t_end": 6.1,
    "dt": 0.1,
    "b": 0.025,
}

if __name__ == "__main__":
    activation = torch.nn.functional.gelu  # torch.relu rbf tanh silu gelu

    with WandbContext(
        project="hydra_test", entity="jhelab", config=CONTROL_HPARAMS
    ) as wc:
        dataset, learner = setup_lotka_steer(
            t_end=wc.config.t_end,
            dt=wc.config.dt,
            noise_strength=wc.config.noise_strength,
            nodes_per_layer=wc.config.num_nodes,
            model_type=wc.config.model_type,
            lr=wc.config.lr,
            activation=activation,
            b=wc.config.b,
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
RANDOM_SEED = 20  # 10, 20, 30
CONTROL_HPARAMS = {
    "num_nodes": 32,
    "lr": 0.05,
    "model_type": "blackbox",
    "noise_strength": 0.05,
    "t_end": 6.1,
    "dt": 0.1,
    "n_intervals": 5,
    "continuity_weight": 0.01,
}

if __name__ == "__main__":
    activation = torch.nn.functional.gelu  # torch.relu rbf tanh silu gelu

    with WandbContext(
        project="LV_multi_sweep",
        entity="jhelab",
        # group="test",
        config=CONTROL_HPARAMS,
    ) as wc:
        dataset, learner = setup_lotka_multishoot(
            t_end=wc.config.t_end,
            dt=wc.config.dt,
            noise_strength=wc.config.noise_strength,
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

        trainer = pl.Trainer(
            max_epochs=4000,
            callbacks=[
                ModelCheckpoint(
                    monitor="mse",
                    mode="min",
                    save_top_k=-1,
                    save_on_train_epoch_end=True,
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
