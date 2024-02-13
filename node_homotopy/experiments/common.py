from typing import Literal

from torch.utils.data import DataLoader
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf, DictConfig
import hydra

from node_homotopy.datasets import DynamicsDataset
from node_homotopy.dynamics import AbstractDynamics
from node_homotopy.odesolve import ODESolveKwargs
from node_homotopy.synchronization import CouplingTerm
from node_homotopy.training import (
    NeuralODETraining,
    VanillaTraining,
    MultipleShootingTraining,
    HomotopyTraining,
)


def setup_training(
    model: AbstractDynamics,
    dataset: DynamicsDataset,
    odesolve_kwargs: ODESolveKwargs,
    training_config: DictConfig,
) -> NeuralODETraining:
    training_type: Literal[
        "vanilla", "multishoot", "homotopy"
    ] = training_config.training_type
    lr = training_config.lr

    match training_type:
        case "vanilla":
            training = VanillaTraining(model, lr, **odesolve_kwargs)
        case "multishoot":
            n_segments = training_config.n_segments
            continuity_weight = training_config.continuity_weight
            training = MultipleShootingTraining(
                model, lr, n_segments, continuity_weight, **odesolve_kwargs
            )
        case "homotopy":
            schedule = hydra.utils.instantiate(training_config.schedule)
            coupling = CouplingTerm.from_dataset(dataset, training_config.k)
            training = HomotopyTraining(
                model, lr, coupling, schedule, **odesolve_kwargs
            )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)

    return training, dataloader


def make_trainer(
    max_epochs: int, checkpoint_every_epoch: bool = False, **trainer_kwargs
) -> lightning.Trainer:
    callbacks = [
        ModelCheckpoint(
            monitor="mse",
            mode="min",
            save_top_k=1,
            save_on_train_epoch_end=True,
            filename="best_{epoch}-{step}",
        )
    ]
    if checkpoint_every_epoch:
        callbacks.append(
            ModelCheckpoint(
                monitor="mse",
                every_n_epochs=1,
                save_top_k=-1,
                filename="{epoch}-{step}",
            ),
        )
    trainer = lightning.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        deterministic="warn",
        accelerator="gpu",
        devices=1,
        **trainer_kwargs,
    )
    return trainer


def wandb_config_from_hydra_config(hydra_config):
    wandb_config = dict(random_seed=hydra_config.random_seed)

    training_kwargs = OmegaConf.to_container(hydra_config.training)
    for key in ("training_type", "lr"):
        wandb_config[key] = training_kwargs.pop(key)
    wandb_config["training"] = training_kwargs

    wandb_config["odesolve"] = OmegaConf.to_container(hydra_config.odesolve)

    model_kwargs = OmegaConf.to_container(hydra_config.model)
    del model_kwargs["_target_"]
    wandb_config["model"] = model_kwargs

    dataset_kwargs = OmegaConf.to_container(hydra_config.dataset)
    del dataset_kwargs["_target_"]
    wandb_config["dataset"] = dataset_kwargs

    wandb_config["max_epochs"] = hydra_config.trainer.max_epochs
    return wandb_config
