from typing import Sequence, Callable
from pathlib import Path
from functools import partialmethod
from shutil import rmtree
import re

import numpy as np
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from .typealiases import Pathlike


# From python 3.10, could use match statement
def cast_to_nparray(x: Sequence) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        # Copy then turn into np.array to avoid complications with device and gradients
        return x.clone().detach().numpy()
    else:
        return np.array(x)


# TODO: Verify error handling is working as intended, delete checkpoint files when a run errors
class WandbContext:
    def __init__(
        self,
        project: str,
        entity: str,
        # id: str,
        save_dir: Pathlike = "./lightning",
        **wandb_init_kwargs
    ):
        wandb.finish()  # Clean up any previous wandb runs remaining
        self.run = wandb.init(project=project, entity=entity, **wandb_init_kwargs)
        self.logger = WandbLogger()
        self.save_dir = (
            Path(save_dir) / project
        )  # Rigorously, need to check if path is valid

    def __enter__(self):
        self._set_pl_trainer_logger()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.run.finish()
        self._reset_pl_trainer()
        if exc_type is not None:
            print(exc_type)
            print(exc_value)
            print(traceback)
            # self._delete_run_api()
            # self._delete_run_local()

    @property
    def config(self):
        return self.logger.experiment.config

    def _set_pl_trainer_logger(self):
        self._original_init = pl.Trainer.__init__  # Monkeypatch pl.Trainer()
        pl.Trainer.__init__ = partialmethod(
            self._original_init,
            logger=self.logger,
            default_root_dir=str(self.save_dir),
        )

    def _reset_pl_trainer(self):
        pl.Trainer.__init__ = self._original_init

    def _delete_run_api(self):
        api = wandb.Api()
        run_api = api.run(self.run.path)
        run_api.delete()

    def _delete_run_local(self):  # Broken at the moment
        wandb_run_dir = self.run.dir
        rmtree(wandb_run_dir)
        # Maybe need code to delete pytorch lightning checkpoints as well?


class StepwiseScheduler:
    def __init__(self, decrement: float, epochs_per_step: int):
        self.decrement = decrement
        self.epochs_per_step = epochs_per_step
        self.values = self._make_values()
        self.epochs = torch.arange(len(self.values)) * epochs_per_step

    def _make_values(self):
        values = torch.arange(1.0, -self.decrement, -self.decrement)
        values = torch.clamp(values, min=0.0, max=1.0)
        return values

    def __call__(self, epoch):
        ind = torch.bucketize(epoch, self.epochs, right=True) - 1
        return self.values[ind]

    @property
    def max_epochs(self) -> int:
        return int(self.epochs[-1] + self.epochs_per_step)


class LogStepwiseScheduler:
    def __init__(self, n_steps: int, step_decay_rate: float, epochs_per_step: int):
        self.n_steps = n_steps
        self.step_decay_rate = step_decay_rate
        self.epochs_per_step = epochs_per_step
        self.values = self._make_values()
        self.epochs = torch.arange(len(self.values)) * epochs_per_step

    def _make_values(self):
        decrements = self.step_decay_rate ** torch.arange(self.n_steps)
        decrements = decrements / torch.sum(decrements)

        values = torch.ones(len(decrements) + 1)
        for i in range(len(values) - 1):
            values[i + 1] = values[i] - decrements[i]
        values[-1] = 0.0
        return values

    def __call__(self, epoch):
        ind = torch.bucketize(epoch, self.epochs, right=True) - 1
        return self.values[ind]

    @property
    def max_epochs(self) -> int:
        return int(self.epochs[-1] + self.epochs_per_step)


def calc_mean_and_stderr(
    array_list: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    array_stack = np.stack(array_list, axis=0)
    mean = np.mean(array_stack, axis=0)
    stderr = np.std(array_stack, axis=0) / np.sqrt(array_stack.shape[0])
    return mean, stderr


def sort_by_epoch(folder, ckpt_regex=r"epoch=(\d+)-step=\d+"):
    ckpt_regex = re.compile(ckpt_regex)

    def get_epoch(filepath: Path) -> int:
        match_result = ckpt_regex.search(filepath.stem)
        return int(match_result.groups()[0])

    return sorted(folder.iterdir(), key=get_epoch)
