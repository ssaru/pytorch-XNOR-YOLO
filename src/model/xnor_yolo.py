from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchsummary import summary as torch_summary

from src.data.pascal_voc import VOC2012
from src.model.detection_loss import yolo_loss
from src.model.utils import BinarizedConvBlock, BinarizedLinearBlock, LinearBlock
from src.utils import make_logger

logger = make_logger(name=str(__name__))


def _build_conv_layers(conv_layers_config):
    return nn.ModuleList([BinarizedConvBlock(**params) for params in conv_layers_config])


def _build_linear_layers(linear_layers_config):
    return nn.ModuleList([LinearBlock(**params) for params in linear_layers_config])


class XnorNetYolo(nn.Module):
    def __init__(self, model_config: DictConfig) -> None:
        super(XnorNetYolo, self).__init__()
        logger.info(f"config: {model_config}, type: {type(model_config)}")

        self.class_map = VOC2012()

        self._width: int = model_config.params.width
        self._height: int = model_config.params.height
        self._channels: int = model_config.params.channels

        self.input_shape: tuple = (self._channels, self._height, self._width)
        self.in_channels: int = self._channels

        logger.info(f"build conv layers")
        if model_config.params.feature_layers.conv:
            self.conv_layers: nn.ModuleList = _build_conv_layers(
                conv_layers_config=model_config.params.feature_layers.conv
            )

        logger.info(f"build linear layers")
        if model_config.params.feature_layers.linear:
            self.linear_layers: nn.ModuleList = _build_linear_layers(
                linear_layers_config=model_config.params.feature_layers.linear
            )

        logger.info(f"build loss layers")
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = yolo_loss

    def forward(self, x):
        if hasattr(self, "conv_layers"):
            if self.conv_layers:
                for conv_layer in self.conv_layers:
                    x = conv_layer(x)

        x = x.view(x.size()[0], -1)

        if hasattr(self, "linear_layers"):
            for linear_layer in self.linear_layers:
                x = linear_layer(x)

        x = x.view(-1, 7, 7, 30)

        return x

    def loss(self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor, image_sizes: Tuple = (448, 448)):
        return self.loss_fn(pred_tensor=pred_tensor, target_tensor=target_tensor, image_sizes=image_sizes)

    def inference(self, x: torch.Tensor):
        outputs = self.batch_inference(x)
        # TODO. batch processing을 염두에 두어야함
        detection_boxes = self.post_processing(x=outputs)

        return detection_boxes

    def post_processing(self, x: torch.Tensor):
        raise NotImplementedError()

    def summary(self):
        # torchsummary only supported [cuda, cpu]. not cuda:0
        device = str(self.device).split(":")[0]
        torch_summary(
            self,
            input_size=(self._channels, self._height, self._width),
            device=device,
        )

    @property
    def device(self):
        devices = {param.device for param in self.parameters()} | {buf.device for buf in self.buffers()}
        if len(devices) != 1:
            raise RuntimeError("Cannot determine device: {} different devices found".format(len(devices)))
        return next(iter(devices))
