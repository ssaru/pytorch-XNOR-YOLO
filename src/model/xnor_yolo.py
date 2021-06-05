from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchsummary import summary as torch_summary

from src.data.pascal_voc import VOC2012
from src.model.detection_loss import yolo_loss, yolotensor_to_xyxyabs
from src.model.utils import (
    BinarizedConvBlock,
    BinarizedLinearBlock,
    LinearBlock,
    get_boxes,
)
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

        self._confidence = model_config.params.confidence
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

        x = x.view(-1, 7, 7, 31)

        x[:, :, :, 0] = torch.sigmoid(x[:, :, :, 0])
        x[:, :, :, 5] = torch.sigmoid(x[:, :, :, 0])
        x[:, :, :, 10:] = torch.sigmoid(x[:, :, :, 10:])

        return x

    def loss(self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor, image_sizes: Tuple = (448, 448)):
        return self.loss_fn(pred_tensor=pred_tensor, target_tensor=target_tensor, image_sizes=image_sizes)

    def inference(self, x: torch.Tensor, image_size: Tuple):
        # single inference        
        pred_tensor = self(x)

        # width, height power of 2
        pred_tensor[:, :, :, 3:5] = torch.pow(pred_tensor[:, :, :, 3:5], 2)
        pred_tensor[:, :, :, 9:11] = torch.pow(pred_tensor[:, :, :, 9:11], 2)

        pred_boxes = yolotensor_to_xyxyabs(yolo_coord_output=pred_tensor, image_sizes=image_size)        
        for boxes_info in pred_boxes:
            box1_idx, box1, box2 = boxes_info
            b, y, x = box1_idx
            pred_tensor[b, y, x, 1:5] = box1
            pred_tensor[b, y, x, 7:11] = box2
        prediction = get_boxes(pred_tensor=pred_tensor, confidence_score=self._confidence)

        return prediction

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
