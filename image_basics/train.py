"""Trainer routines and classes.

we have to integrate: data, model, loss, optimizer, scheduler
"""

import json
from datetime import datetime
from functools import wraps
from pathlib import Path

import safetensors.torch as safetensors
import torch
import torchmetrics.classification as tmc
from loguru import logger
from torchmetrics import MetricCollection

import image_basics as ib


def create_loss(loss_name: str = "cross_entropy", **kwargs):
    if loss_name == "cross_entropy":
        return torch.nn.CrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"{loss_name} is not a valid loss str")


def create_optimizer(model: torch.nn.Module, optimizer_name: str = "adamw", **kwargs):
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), **kwargs)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    else:
        raise ValueError(f"{optimizer_name} is not a valid optimizer str")


def dispatch(f):
    """Method decorator that runs associated methods on callbacks."""

    @wraps(f)
    def wrapper(trainer, *args, **kwargs):
        result = None
        for cb in trainer.callbacks:
            if hasattr(cb, f.__name__):
                r = getattr(cb, f.__name__)(trainer, *args, **kwargs)
                result = r if r is not None else result
        return result

    return wrapper


class MetricsCallback:
    def __init__(self, num_classes):
        # loss used during training
        self.epoch = -1
        # dummy placeholder, gets reinitialized per epoch
        self.train_loss = torch.tensor(0.0)
        self.train_results = {}
        # dummy placeholder, gets reinitialized per epoch
        self.val_loss = torch.tensor(0.0)
        self.val_results = {}

        # torchmetrics collection
        self.train_metrics = MetricCollection(
            {
                "accuracy": tmc.Accuracy(task="multiclass", num_classes=num_classes),
                "f1": tmc.F1Score(task="multiclass", num_classes=num_classes),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")

    def on_train_start(self, trainer, epoch_idx):
        # keeping running loss on device
        self.train_loss = torch.tensor(0.0, device=trainer.device)
        self.train_metrics.to(trainer.device)

    def on_train_batch(self, trainer, batch_loss, y_pred, y, batch_idx, epoch_idx):
        self.train_loss += batch_loss * trainer.Bt
        self.train_metrics.update(y_pred, y)

    def on_train_end(self, trainer, epoch_idx):
        self.train_loss /= trainer.Nt
        # updates and stores latest metric results (cpu, native python)
        self.train_results = {
            k: v.cpu().numpy().tolist() for k, v in self.train_metrics.compute().items()
        }
        self.epoch = epoch_idx

        # reset internal metric state
        self.train_metrics.reset()

    def on_val_start(self, trainer, epoch_idx):
        # keeping running loss on device
        self.val_loss = torch.tensor(0.0, device=trainer.device)
        self.val_metrics.to(trainer.device)

    def on_val_batch(self, trainer, batch_loss, y_pred, y, batch_idx, epoch_idx):
        self.val_loss += batch_loss * trainer.Bv
        self.val_metrics.update(y_pred, y)

    def on_val_end(self, trainer, epoch_idx):
        self.val_loss /= trainer.Nv
        # updates and stores latest metric results (cpu, native python)
        self.val_results = {
            k: v.cpu().numpy().tolist() for k, v in self.val_metrics.compute().items()
        }
        self.epoch = epoch_idx

        # reset internal metric state
        self.val_metrics.reset()
        # return current metrics including validation
        return self.all_metrics

    def is_new_model_better(self, trainer, new_metrics, old_metrics):
        return old_metrics is None or new_metrics["val_loss"] < old_metrics["val_loss"]

    @property
    def all_metrics(self):
        """Provide dictionary of all current metrics."""
        return dict(
            epoch=self.epoch,
            train_loss=self.train_loss.item(),
            val_loss=self.val_loss.item(),
            **self.train_results,
            **self.val_results,
        )


class ConsoleCallback:
    def __init__(self, metric_callback):
        # relies on metric callback calculations
        self.metric_callback = metric_callback

    def on_train_batch(self, trainer, batch_loss, y_pred, y, batch_idx, epoch_idx):
        logger.info(
            f"processed {(batch_idx + 1)*trainer.Bt}/{trainer.Nt} batch loss {batch_loss.item()}"
        )

    def on_train_end(self, trainer, epoch_idx):
        logger.info(
            f"epoch {epoch_idx}/{trainer.num_epochs} "
            f"train loss {self.metric_callback.all_metrics['train_loss']}"
        )

    def on_val_end(self, trainer, epoch_idx):
        logger.info(
            f"epoch {epoch_idx}/{trainer.num_epochs} "
            f"val loss {self.metric_callback.all_metrics['val_loss']}\n"
            f"{self.metric_callback.all_metrics}"
        )

    def save_best_checkpoint(self, trainer, epoch_idx):
        logger.info(f"epoch {epoch_idx} found best model")


class CheckpointCallback:
    def __init__(self, metric_callback, checkpoint_dir=None):
        # relies on metric callback calculations
        self.metric_callback = metric_callback
        # storage defaults to <cache>/_experiments/
        self.checkpoint_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else ib.utils.CACHE_DIR
            / f"_experiments/exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    def _save_json(self, data_dict: dict, json_file: Path):
        """Save dictionary to json (helper)."""
        json_file.parent.mkdir(parents=True, exist_ok=True)
        with json_file.open("w") as f:
            json.dump(data_dict, f, indent=4)

    def _save_model(self, model: torch.nn.Module, model_file: Path):
        safetensors.save_model(model, model_file)

    def _load_model(self, model: torch.nn.Module, model_file: Path):
        safetensors.load_model(model, model_file)

    def on_val_end(self, trainer, epoch_idx):
        # store metrics after validation
        metrics_file = self.checkpoint_dir / f"metrics/epoch_{epoch_idx:04d}.json"
        self._save_json(self.metric_callback.all_metrics, metrics_file)

    def save_best_checkpoint(self, trainer, epoch_idx):
        metrics_file = self.checkpoint_dir / "best_metrics.json"
        model_file = self.checkpoint_dir / "best_model.safetensors"
        # checkpoint_file = self.checkpoint_dir / "best_checkpoint.tar"

        # save best metrics
        self._save_json(self.metric_callback.all_metrics, metrics_file)
        # save best model
        self._save_model(trainer.model, model_file)
        # save other checkpoint info

    def load_best_checkpoint(self, trainer):
        model_file = self.checkpoint_dir / "best_model.safetensors"

        # load best model
        self._load_model(trainer.model, model_file)


def default_callbacks(num_classes):
    mc = MetricsCallback(num_classes)
    return [mc, ConsoleCallback(mc), CheckpointCallback(mc, None)]


class SimpleTrainer:
    """Simple trainer implementation.

    orchestrates the training process assuming we've been given
    the 5 key modeling component objects:
     - model, train dataloader, val dataloader, loss, optimizer
    """

    def __init__(
        self,
        model,
        train_dl,
        val_dl,
        loss,
        optimizer,
        num_epochs=3,
        epoch_interval=1,
        device=None,
        callbacks=None,
        non_blocking=False,
    ):
        # params reflect training recipe, does not change trainer state
        # core elements
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.loss = loss
        self.optimizer = optimizer
        # training hyperparams
        self.num_epochs = num_epochs
        self.epoch_interval = epoch_interval
        self.device = device
        self.callbacks = callbacks
        self.non_blocking = non_blocking

    def _init_trainer(self):
        # prepare trainer, may change trainer state
        # convert device name to device
        self.device = ib.utils.get_device(self.device)

        # default callbacks
        self.callbacks = (
            self.callbacks
            if self.callbacks is not None
            else default_callbacks(self.model.num_classes)
        )

        # data total and batch sizes, assuming drop_last=True
        self.Bt = self.train_dl.batch_size
        self.Nt = self.Bt * len(self.train_dl)
        self.Bv = self.val_dl.batch_size
        self.Nv = self.Bv * len(self.val_dl)

    @dispatch
    def on_train_start(self, epoch_idx):
        pass

    @dispatch
    def on_train_batch(self, batch_loss, y_pred, y, batch_idx, epoch_idx):
        pass

    @dispatch
    def on_train_end(self, epoch_idx):
        pass

    @dispatch
    def on_val_start(self, epoch_idx):
        pass

    @dispatch
    def on_val_batch(self, batch_loss, y_pred, y, batch_idx, epoch_idx):
        pass

    @dispatch
    def on_val_end(self, epoch_idx):
        pass

    @dispatch
    def is_new_model_better(self, new_metrics, old_metrics):
        pass

    @dispatch
    def save_best_checkpoint(self, epoch_idx):
        pass

    @dispatch
    def load_best_checkpoint(self, epoch_idx):
        pass

    def eval_model(self, val_dl, epoch):
        self.model.eval()
        self.on_val_start(epoch)
        with torch.no_grad():
            for i, batch in enumerate(self.val_dl):
                X, y = batch["image"], batch["target"]
                X, y = X.to(self.device), y.to(self.device)

                # calculate loss
                y_pred = self.model(X)
                loss = self.loss(y_pred, y)

                self.on_val_batch(loss, y_pred, y, i, epoch)
        eval_metrics = self.on_val_end(epoch)
        # assume a callback will provide a dictionary of metrics
        return eval_metrics

    def train_epoch(self, epoch):
        """Perform one epoch of training over training data."""
        self.model.train()
        self.on_train_start(epoch)
        for i, batch in enumerate(self.train_dl):
            X, y = batch["image"], batch["target"]
            X = X.to(self.device, non_blocking=self.non_blocking)
            y = y.to(self.device, non_blocking=self.non_blocking)

            # forward pass
            y_pred = self.model(X)
            # calculate loss
            loss = self.loss(y_pred, y)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.on_train_batch(loss, y_pred, y, i, epoch)
        self.on_train_end(epoch)

    def train(self):
        self._init_trainer()

        self.model.to(self.device)
        self.best_metrics = None
        self.best_epoch = -1
        for epoch in range(1, self.num_epochs + 1):
            # train
            self.train_epoch(epoch)

            if epoch % self.epoch_interval == 0:
                # eval
                eval_metrics = self.eval_model(self.val_dl, epoch)

                # update results and checkpoint
                if self.is_new_model_better(eval_metrics, self.best_metrics):
                    self.save_best_checkpoint(epoch)
                    self.best_metrics = eval_metrics
                    self.best_epoch = epoch

        # load best model after training is complete
        self.load_best_checkpoint()


def create_trainer(
    model_params=None,
    data_params=None,
    loss_params=None,
    optimizer_params=None,
    num_epochs=1,
    epoch_interval=1,
    device=None,
    callbacks=None,
    non_blocking=False,
):
    """Recipe-style entry point that creates a training task.

    Takes task hyperparams, uses them to generate component objects,
    then pass them to trainer.
    Returns trainer ojbect prepared to train.
    """
    # create model
    model_params = model_params if model_params is not None else {}
    model = ib.model.create_model(**model_params)

    # create dataloaders
    data_params = data_params if data_params is not None else {}
    train_dl, val_dl = ib.data.create_dataloaders(**data_params)

    # create loss
    loss_params = loss_params if loss_params is not None else {}
    loss = create_loss(**loss_params)

    # create optimizer
    optimizer_params = optimizer_params if optimizer_params is not None else {}
    optimizer = create_optimizer(model, **optimizer_params)

    return SimpleTrainer(
        model,
        train_dl,
        val_dl,
        loss,
        optimizer,
        num_epochs=num_epochs,
        epoch_interval=epoch_interval,
        device=device,
        callbacks=callbacks,
        non_blocking=non_blocking,
    )
