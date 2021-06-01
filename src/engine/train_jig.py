import gc
import os
from typing import Tuple

import psutil
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
        self.image_sizes = (config.model.params.width, config.model.params.height)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        opt_args = dict(self.config.optimizer.params)
        opt_args.update({"params": self.model.parameters(), "lr": self.lr})
        opt = load_class(module=optim, name=self.config.optimizer.type, args=opt_args)

        scheduler_args = dict(self.config.scheduler.params)
        scheduler_args.update({"optimizer": opt})
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
        box1_confidence_loss = loss_dict["obj_loss"].pop("confidence1_loss").cpu()
        box1_cx_loss = loss_dict["obj_loss"].pop("box1_cx_loss").cpu()
        box1_cy_loss = loss_dict["obj_loss"].pop("box1_cy_loss").cpu()
        box1_width_loss = loss_dict["obj_loss"].pop("box1_width_loss").cpu()
        box1_height_loss = loss_dict["obj_loss"].pop("box1_height_loss").cpu()

        box2_confidence_loss = loss_dict["obj_loss"].pop("confidence2_loss").cpu()
        box2_cx_loss = loss_dict["obj_loss"].pop("box2_cx_loss").cpu()
        box2_cy_loss = loss_dict["obj_loss"].pop("box2_cy_loss").cpu()
        box2_width_loss = loss_dict["obj_loss"].pop("box2_width_loss").cpu()
        box2_height_loss = loss_dict["obj_loss"].pop("box2_height_loss").cpu()

        classes_loss = loss_dict["obj_loss"].pop("classes_loss").cpu()

        pid = os.getpid()
        current_process = psutil.Process(pid)
        current_process_memory_usage_as_MB = round(current_process.memory_info()[0] / 2.0 ** 20)

        self.log("memory", current_process_memory_usage_as_MB, on_step=True, prog_bar=True)

        self.log("train_loss", total_loss, on_step=True)

        self.log("train_box1_confidence_loss", box1_confidence_loss, on_step=True)
        self.log("train_box1_cx_loss", box1_cx_loss, on_step=True)
        self.log("train_box1_cy_loss", box1_cy_loss, on_step=True)
        self.log("train_box1_width_loss", box1_width_loss, on_step=True)
        self.log("train_box1_height_loss", box1_height_loss, on_step=True)

        self.log("train_box2_confidence_loss", box2_confidence_loss, on_step=True)
        self.log("train_box2_cx_loss", box2_cx_loss, on_step=True)
        self.log("train_box2_cy_loss", box2_cy_loss, on_step=True)
        self.log("train_box2_width_loss", box2_width_loss, on_step=True)
        self.log("train_box2_height_loss", box2_height_loss, on_step=True)

        self.log("train_classes_loss", classes_loss, on_step=True)

        return {
            "loss": total_loss,
            "train_loss": total_loss,
            "train_box1_confidence_loss": box1_confidence_loss,
            "train_box1_cx_loss": box1_cx_loss,
            "train_box1_cy_loss": box1_cy_loss,
            "train_box1_width_loss": box1_width_loss,
            "train_box1_height_loss": box1_height_loss,
            "train_box2_confidence_loss": box2_confidence_loss,
            "train_box2_cx_loss": box2_cx_loss,
            "train_box2_cy_loss": box2_cy_loss,
            "train_box2_width_loss": box2_width_loss,
            "train_box2_height_loss": box2_height_loss,
            "train_classes_loss": classes_loss,
        }

    def training_epoch_end(self, training_step_outputs):
        loss = 0
        box1_confidence_loss = 0
        box1_cx_loss, box1_cy_loss = 0, 0
        box1_width_loss, box1_height_loss = 0, 0

        box2_confidence_loss = 0
        box2_cx_loss, box2_cy_loss = 0, 0
        box2_width_loss, box2_height_loss = 0, 0

        classes_loss = 0

        num_of_outputs = len(training_step_outputs)

        for log_dict in training_step_outputs:
            loss += log_dict["loss"]

            box1_confidence_loss += log_dict["train_box1_confidence_loss"]
            box1_cx_loss += log_dict["train_box1_cx_loss"]
            box1_cy_loss += log_dict["train_box1_cy_loss"]
            box1_width_loss += log_dict["train_box1_width_loss"]
            box1_height_loss += log_dict["train_box1_height_loss"]

            box2_confidence_loss += log_dict["train_box2_confidence_loss"]
            box2_cx_loss += log_dict["train_box2_cx_loss"]
            box2_cy_loss += log_dict["train_box2_cy_loss"]
            box2_width_loss += log_dict["train_box2_width_loss"]
            box2_height_loss += log_dict["train_box2_height_loss"]

            classes_loss += log_dict["train_classes_loss"]

        loss /= num_of_outputs
        box1_confidence_loss /= num_of_outputs
        box1_cx_loss /= num_of_outputs
        box1_cy_loss /= num_of_outputs
        box1_width_loss /= num_of_outputs
        box1_height_loss /= num_of_outputs

        box2_confidence_loss /= num_of_outputs
        box2_cx_loss /= num_of_outputs
        box2_cy_loss /= num_of_outputs
        box2_width_loss /= num_of_outputs
        box2_height_loss /= num_of_outputs

        classes_loss /= num_of_outputs

        self.log("train_loss", loss, on_epoch=True)

        self.log("train_box1_confidence_loss", box1_confidence_loss, on_epoch=True)
        self.log("train_box1_cx_loss", box1_cx_loss, on_epoch=True)
        self.log("train_box1_cy_loss", box1_cy_loss, on_epoch=True)
        self.log("train_box1_width_loss", box1_width_loss, on_epoch=True)
        self.log("train_box1_height_loss", box1_height_loss, on_epoch=True)

        self.log("train_box2_confidence_loss", box2_confidence_loss, on_epoch=True)
        self.log("train_box2_cx_loss", box2_cx_loss, on_epoch=True)
        self.log("train_box2_cy_loss", box2_cy_loss, on_epoch=True)
        self.log("train_box2_width_loss", box2_width_loss, on_epoch=True)
        self.log("train_box2_height_loss", box2_height_loss, on_epoch=True)

        self.log("train_classes_loss", classes_loss, on_epoch=True)

        for obj in gc.get_objects():
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                del obj
        torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss_dict = self.shared_step(x, y)

        total_loss = loss_dict["total_loss"]
        box1_confidence_loss = loss_dict["obj_loss"].pop("confidence1_loss")
        box1_cx_loss = loss_dict["obj_loss"].pop("box1_cx_loss")
        box1_cy_loss = loss_dict["obj_loss"].pop("box1_cy_loss")
        box1_width_loss = loss_dict["obj_loss"].pop("box1_width_loss")
        box1_height_loss = loss_dict["obj_loss"].pop("box1_height_loss")

        box2_confidence_loss = loss_dict["obj_loss"].pop("confidence2_loss")
        box2_cx_loss = loss_dict["obj_loss"].pop("box2_cx_loss")
        box2_cy_loss = loss_dict["obj_loss"].pop("box2_cy_loss")
        box2_width_loss = loss_dict["obj_loss"].pop("box2_width_loss")
        box2_height_loss = loss_dict["obj_loss"].pop("box2_height_loss")

        classes_loss = loss_dict["obj_loss"].pop("classes_loss")

        self.log("valid_loss", total_loss, on_step=True, logger=True)

        self.log("valid_box1_confidence_loss", box1_confidence_loss, on_step=True, logger=True)
        self.log("valid_box1_cx_loss", box1_cx_loss, on_step=True, logger=True)
        self.log("valid_box1_cy_loss", box1_cy_loss, on_step=True, logger=True)
        self.log("valid_box1_width_loss", box1_width_loss, on_step=True, logger=True)
        self.log("valid_box1_height_loss", box1_height_loss, on_step=True, logger=True)

        self.log("valid_box2_confidence_loss", box2_confidence_loss, on_step=True, logger=True)
        self.log("valid_box2_cx_loss", box2_cx_loss, on_step=True, logger=True)
        self.log("valid_box2_cy_loss", box2_cy_loss, on_step=True, logger=True)
        self.log("valid_box2_width_loss", box2_width_loss, on_step=True, logger=True)
        self.log("valid_box2_height_loss", box2_height_loss, on_step=True, logger=True)

        self.log("valid_classes_loss", classes_loss, on_step=True, logger=True)

        return {
            "loss": total_loss,
            "valid_loss": total_loss,
            "valid_box1_confidence_loss": box1_confidence_loss,
            "valid_box1_cx_loss": box1_cx_loss,
            "valid_box1_cy_loss": box1_cy_loss,
            "valid_box1_width_loss": box1_width_loss,
            "valid_box1_height_loss": box1_height_loss,
            "valid_box2_confidence_loss": box2_confidence_loss,
            "valid_box2_cx_loss": box2_cx_loss,
            "valid_box2_cy_loss": box2_cy_loss,
            "valid_box2_width_loss": box2_width_loss,
            "valid_box2_height_loss": box2_height_loss,
            "valid_classes_loss": classes_loss,
        }

    def validation_epoch_end(self, validation_step_outputs):
        loss = 0

        box1_confidence_loss = 0
        box1_cx_loss, box1_cy_loss = 0, 0
        box1_width_loss, box1_height_loss = 0, 0

        box2_confidence_loss = 0
        box2_cx_loss, box2_cy_loss = 0, 0
        box2_width_loss, box2_height_loss = 0, 0

        classes_loss = 0

        num_of_outputs = len(validation_step_outputs)

        for log_dict in validation_step_outputs:
            loss += log_dict["loss"]

            box1_confidence_loss += log_dict["valid_box1_confidence_loss"]
            box1_cx_loss += log_dict["valid_box1_cx_loss"]
            box1_cy_loss += log_dict["valid_box1_cy_loss"]
            box1_width_loss += log_dict["valid_box1_width_loss"]
            box1_height_loss += log_dict["valid_box1_height_loss"]

            box2_confidence_loss += log_dict["valid_box2_confidence_loss"]
            box2_cx_loss += log_dict["valid_box2_cx_loss"]
            box2_cy_loss += log_dict["valid_box2_cy_loss"]
            box2_width_loss += log_dict["valid_box2_width_loss"]
            box2_height_loss += log_dict["valid_box2_height_loss"]

            classes_loss += log_dict["valid_classes_loss"]

        loss /= num_of_outputs
        box1_confidence_loss /= num_of_outputs
        box1_cx_loss /= num_of_outputs
        box1_cy_loss /= num_of_outputs
        box1_width_loss /= num_of_outputs
        box1_height_loss /= num_of_outputs

        box2_confidence_loss /= num_of_outputs
        box2_cx_loss /= num_of_outputs
        box2_cy_loss /= num_of_outputs
        box2_width_loss /= num_of_outputs
        box2_height_loss /= num_of_outputs

        classes_loss /= num_of_outputs

        self.log("valid_loss", loss, on_epoch=True, logger=True)

        self.log("valid_box1_confidence_loss", box1_confidence_loss, on_epoch=True, logger=True)
        self.log("valid_box1_cx_loss", box1_cx_loss, on_epoch=True, logger=True)
        self.log("valid_box1_cy_loss", box1_cy_loss, on_epoch=True, logger=True)
        self.log("valid_box1_width_loss", box1_width_loss, on_epoch=True, logger=True)
        self.log("valid_box1_height_loss", box1_height_loss, on_epoch=True, logger=True)

        self.log("valid_box2_confidence_loss", box2_confidence_loss, on_epoch=True, logger=True)
        self.log("valid_box2_cx_loss", box2_cx_loss, on_epoch=True, logger=True)
        self.log("valid_box2_cy_loss", box2_cy_loss, on_epoch=True, logger=True)
        self.log("valid_box2_width_loss", box2_width_loss, on_epoch=True, logger=True)
        self.log("valid_box2_height_loss", box2_height_loss, on_epoch=True, logger=True)

        self.log("valid_classes_loss", classes_loss, on_epoch=True, logger=True)
