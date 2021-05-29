from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, StepLR

from src.utils import load_class


class TrainingContainer(LightningModule):
    def __init__(self, model: nn.Module, config: DictConfig):
        """Model Container for Training

        Args:
            model (nn.Module): model for train
            config (DictConfig): configuration with Omegaconf.DictConfig format for dataset/model/runner
        """
        super().__init__()
        self.model = model
        self.save_hyperparameters(config)
        self.config = config
        self.lr = config.optimizer.params.lr
        self.scheduler_gamma = config.scheduler.params.gamma
        self.image_sizes = (config.model.params.width, config.model.params.height)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt_args = dict(self.config.optimizer.params)
        opt_args.update({"params": self.model.parameters(), "lr": self.lr})
        opt = load_class(module=optim, name=self.config.optimizer.type, args=opt_args)

        scheduler_args = dict(self.config.scheduler.params)
        scheduler_args.update({"optimizer": opt, "gamma": self.scheduler_gamma})
        scheduler = load_class(
            module=optim.lr_scheduler,
            name=self.config.scheduler.type,
            args=scheduler_args,
        )

        result = {"optimizer": opt, "lr_scheduler": scheduler}
        if self.config.scheduler.params == "ReduceLROnPlateau":
            result.update({"monitor": self.config.scheduler.monitor})

        return result

    def shared_step(self, input, target):
        pred_tensor = self(input)
        loss_dict = self.model.loss(pred_tensor=pred_tensor, target_tensor=target, image_sizes=self.image_sizes)

        return pred_tensor, loss_dict

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss_dict = self.shared_step(x, y)

        total_loss = loss_dict["total_loss"]
        self.logging_loss(loss_dict, prefix="train")

        return {
            "loss": loss_dict["total_loss"],
            "train_loss": total_loss,
        }

    def training_epoch_end(self, training_step_outputs):
        loss = 0
        num_of_outputs = len(training_step_outputs)

        for log_dict in training_step_outputs:
            loss += log_dict["loss"]

        loss /= num_of_outputs

        self.log(
            name="train/loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            name="train_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss_dict = self.shared_step(x, y)
        total_loss = loss_dict["total_loss"]
        self.logging_loss(loss_dict, prefix="valid")

        return {
            "valid/loss": total_loss,
            "valid_loss": total_loss,
        }

    def validation_epoch_end(self, validation_step_outputs):
        loss = 0
        num_of_outputs = len(validation_step_outputs)

        for log_dict in validation_step_outputs:
            loss += log_dict["valid/loss"]

        loss = loss / num_of_outputs

        self.log(
            name="valid/loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            name="valid_loss",
            value=loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

    def logging_loss(self, loss, prefix: str = "train"):
        total_loss = loss["total_loss"]
        self.log(
            name=f"{prefix}/loss",
            value=total_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        self.log(
            name=f"{prefix}_loss",
            value=total_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        box1_confidence_loss = loss["obj_loss"]["confidence1_loss"]
        self.log(
            name=f"{prefix}/box1_confidence_loss",
            value=box1_confidence_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        self.log(
            name=f"{prefix}_box1_confidence_loss",
            value=box1_confidence_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        box1_cx_loss = loss["obj_loss"]["box1_cx_loss"]
        self.log(
            name=f"{prefix}/box1_cx_loss",
            value=box1_cx_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        self.log(
            name=f"{prefix}_box1_cx_loss",
            value=box1_cx_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        box1_cy_loss = loss["obj_loss"]["box1_cy_loss"]
        self.log(
            name=f"{prefix}/box1_cy_loss",
            value=box1_cy_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        self.log(
            name=f"{prefix}_box1_cy_loss",
            value=box1_cy_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        box1_width_loss = loss["obj_loss"]["box1_width_loss"]
        self.log(
            name=f"{prefix}/box1_width_loss",
            value=box1_width_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        self.log(
            name=f"{prefix}_box1_width_loss",
            value=box1_width_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        box1_height_loss = loss["obj_loss"]["box1_height_loss"]
        self.log(
            name=f"{prefix}/box1_height_loss",
            value=box1_height_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        self.log(
            name=f"{prefix}_box1_height_loss",
            value=box1_height_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        box2_confidence_loss = loss["obj_loss"]["confidence2_loss"]
        self.log(
            name=f"{prefix}/box2_confidence_loss",
            value=box2_confidence_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        self.log(
            name=f"{prefix}_box2_confidence_loss",
            value=box2_confidence_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        box2_cx_loss = loss["obj_loss"]["box2_cx_loss"]
        self.log(
            name=f"{prefix}/box2_cx_loss",
            value=box2_cx_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        self.log(
            name=f"{prefix}_box2_cx_loss",
            value=box2_cx_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        box2_cy_loss = loss["obj_loss"]["box2_cy_loss"]
        self.log(
            name=f"{prefix}/box2_cy_loss",
            value=box2_cy_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        self.log(
            name=f"{prefix}_box2_cy_loss",
            value=box2_cy_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        box2_width_loss = loss["obj_loss"]["box2_width_loss"]
        self.log(
            name=f"{prefix}/box2_width_loss",
            value=box2_width_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        self.log(
            name=f"{prefix}_box2_width_loss",
            value=box2_width_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        box2_height_loss = loss["obj_loss"]["box2_height_loss"]
        self.log(
            name=f"{prefix}/box2_height_loss",
            value=box2_height_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        self.log(
            name=f"{prefix}_box2_height_loss",
            value=box2_height_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )

        classes_loss = loss["obj_loss"]["classes_loss"]
        self.log(
            name=f"{prefix}/classes_loss",
            value=classes_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
        self.log(
            name=f"{prefix}_classes_loss",
            value=classes_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=False,
        )
