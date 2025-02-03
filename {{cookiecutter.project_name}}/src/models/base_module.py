from typing import Any, Dict
from lightning.pytorch import LightningModule
import torch
from torch import Tensor
from src.models.metrics.base import BaseMetricContainer
from src.models.losses.base import BaseLossContainer


class BaseModule(LightningModule):
    def __init__(self, architecture: torch.nn.Module, 
                 loss: BaseLossContainer, 
                 metrics_train: BaseMetricContainer, 
                 metrics_val:   BaseMetricContainer, 
                 metrics_test:  BaseMetricContainer,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()

        # create model
        self.model: torch.nn.Module = architecture

        # create loss function
        self.loss_fn_train: BaseLossContainer = loss(stage="train")
        self.loss_fn_val: BaseLossContainer = loss(stage="val")

        # create metrics
        self.metrics_train: BaseMetricContainer = metrics_train(stage="train")
        self.metrics_val: BaseMetricContainer = metrics_val(stage="val")
        self.metrics_test: BaseMetricContainer = metrics_test(stage="test")

        # store partial optimizer
        self.optimizer: torch.optim.Optimizer = optimizer

        # store partial scheduler
        self.scheduler: torch.optim.lr_scheduler = scheduler


    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        return self.model(x)

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        """Training step."""
        model_output = self.model(batch["image"])
        loss_dict = self.loss_fn_train(model_output, batch)
        self.metrics_train.update(model_output, batch)

        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss_dict["train/loss"]

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        metric_dict = self.metrics_train.compute()
        self.metrics_train.reset()

        self.log_dict(metric_dict, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Tensor]:
        """Validation step."""
        model_output = self.model(batch["image"])
        loss_dict = self.loss_fn_val(model_output, batch)
        self.metrics_val.update(model_output, batch)

        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=True)
        return loss_dict["val/loss"]

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        metric_dict = self.metrics_val.compute()
        self.metrics_val.reset()

        self.log_dict(metric_dict, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Test step."""
        model_output = self.model(batch["image"])
        self.metrics_test.update(model_output, batch)

    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        metric_dict = self.metrics_test.compute()
        self.metrics_test.reset()

        self.log_dict(metric_dict, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}