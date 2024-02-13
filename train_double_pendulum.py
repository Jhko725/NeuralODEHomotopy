import lightning
from lightning.pytorch.loggers import WandbLogger
import hydra
from omegaconf import DictConfig

from node_homotopy.datasets import DynamicsDataset
from node_homotopy.models import AbstractDynamics
from node_homotopy.training import HomotopyTraining
from node_homotopy.experiments.common import (
    setup_training,
    make_trainer,
    wandb_config_from_hydra_config,
)


@hydra.main(
    version_base=None, config_path="configs", config_name="double_pendulum_config"
)
def main(cfg: DictConfig):
    dataset: DynamicsDataset = hydra.utils.call(cfg.dataset)
    lightning.seed_everything(cfg.random_seed)
    model: AbstractDynamics = hydra.utils.call(cfg.model)

    training, dataloader = setup_training(
        model=model,
        dataset=dataset,
        odesolve_kwargs=cfg.odesolve,
        training_config=cfg.training,
    )

    logger = hydra.utils.instantiate(cfg.logger)
    if isinstance(logger, WandbLogger):
        wandb_config = wandb_config_from_hydra_config(cfg)
        wandb_config["dataset"] = "double_pendulum"
        logger.experiment.config.update(wandb_config)

    if isinstance(training, HomotopyTraining):
        max_epochs = training.schedule.max_epochs
    else:
        max_epochs = cfg.trainer.max_epochs
    trainer = make_trainer(max_epochs, logger=logger)
    trainer.fit(training, dataloader)


if __name__ == "__main__":
    main()
