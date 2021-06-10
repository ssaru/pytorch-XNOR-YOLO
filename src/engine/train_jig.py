import functools
import gc
import math
import os
from collections import defaultdict
from typing import Tuple

import numpy as np
import psutil
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, ReduceLROnPlateau, StepLR

from src.engine.predictor import Predictor
from src.utils import load_class, compute_mAP


class TrainingContainer(LightningModule):
    def __init__(self, model: nn.Module, config: DictConfig, len_dataloader: int):
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
        self.len_dataloader = len_dataloader

        self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.conf_thresh = 0.001
        self.prob_thresh = 0.001
        self.nms_thresh = 0.5

        self.mAP_targets = defaultdict(list)
        self.mAP_preds = defaultdict(list)

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
        loss = self.model.loss(pred_tensor=pred_tensor, target_tensor=target)

        return pred_tensor, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self.shared_step(x, y)
        pid = os.getpid()
        current_process = psutil.Process(pid)
        current_process_memory_usage_as_MB = round(current_process.memory_info()[0] / 2.0 ** 20)

        self.log("memory", current_process_memory_usage_as_MB, on_step=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True)

        return {"loss": loss}

    def training_epoch_end(self, training_step_outputs):
        loss = 0
        num_of_outputs = len(training_step_outputs)
        for log_dict in training_step_outputs:
            loss += log_dict["loss"]

        loss /= num_of_outputs
        self.log("train_loss", loss, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self.shared_step(x, y)
        self.log("valid_loss", loss, on_step=True)

        return {"valid_loss", loss}

    def validation_epoch_end(self, validation_step_outputs):
        loss = 0
        num_of_outputs = len(validation_step_outputs)
        for log_dict in validation_step_outputs:
            loss += log_dict["valid_loss"]

        loss /= num_of_outputs
        self.log("valid_loss", loss, on_epoch=True)

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     target = OmegaConf.create(y)
    #     filename = str(target.annotation.filename)

    #     # Preparing ground-truth data...
    #     for object_info in target.annotation.object:
    #         classes = object_info.name

    #         xmin = object_info.bndbox.xmin
    #         ymin = object_info.bndbox.ymin
    #         xmax = object_info.bndbox.xmax
    #         ymax = object_info.bndbox.ymax

    #         self.mAP_targets[(filename, classes)].append([xmin, ymin, xmax, ymax])

    #     # Predicting...
    #     x = transforms.ToPILImage()(x.squeeze())
    #     image = x.resize(self.image_sizes)
    #     image = np.array(image)
    #     image = (image - self.mean) / 255.0
    #     image = torchvision.transforms.ToTensor()(image)
    #     image = image.unsqueeze(0)
    #     device = self.model.device
    #     image = image.to(device)
    #     boxes_detected, class_names_detected, probs_detected = self.model.inference(
    #         image,
    #         image_size=self.image_sizes,
    #         conf_thresh=self.conf_thresh,
    #         prob_thresh=self.prob_thresh,
    #         nms_thresh=self.nms_thresh,
    #     )
    #     for box, class_name, prob in zip(boxes_detected, class_names_detected, probs_detected):
    #         xmin, ymin, xmax, ymax = box
    #         self.mAP_preds[class_name].append([filename, prob, xmin, ymin, xmax, ymax])

    # def validation_epoch_end(self, validation_step_outputs):
    #     voc_class_names = self.model.class_name_list
    #     mAP, aps = compute_mAP(self.mAP_preds, self.mAP_targets, class_names=voc_class_names)
    #     self.log("mAP", mAP, on_epoch=True, logger=True)

    #     for key, value in aps.items():
    #         self.log(key, value, on_epoch=True, logger=True)

    #     self.mAP_targets = defaultdict(list)
    #     self.mAP_preds = defaultdict(list)
