from typing import Sequence

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .learners import SynchronizedLearner, HomotopyLearner


def train_until_target_metric(
    model: SynchronizedLearner,
    dataloader: DataLoader,
    metric_name: str,
    target_value: float,
    max_epochs: int,
    callbacks: Sequence[pl.Callback] | None = None,
    **trainer_kwargs,
):
    earlystop_callback = EarlyStopping(
        monitor=metric_name,
        mode="min",
        patience=max_epochs,
        stopping_threshold=target_value,
        check_on_train_epoch_end=True,
    )

    if callbacks is None:
        callbacks = [earlystop_callback]
    else:
        callbacks = [earlystop_callback, *callbacks]

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        # check_val_every_n_epoch=max_epochs,  # added
        **trainer_kwargs,
    )

    trainer.fit(model, dataloader)
    return trainer


def train_homotopy(
    learner: HomotopyLearner,
    train_dataloader: DataLoader,
    **trainer_kwargs,
):
    trainer = pl.Trainer(
        max_epochs=learner.scheduler.max_epochs,
        callbacks=[
            ModelCheckpoint(
                monitor="mse",
                mode="min",
                save_top_k=-1,
                save_on_train_epoch_end=True,
                filename="best_{lambda:.2f}-{epoch}-{step}",
            ),
        ],
        log_every_n_steps=1,
        accelerator="gpu",
        devices=1,
        deterministic="warn",
        **trainer_kwargs,
    )

    trainer.fit(learner, train_dataloader)

    return trainer
