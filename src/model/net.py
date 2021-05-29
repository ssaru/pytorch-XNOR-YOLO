import copy
from logging import config
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchsummary import summary as torch_summary

from src.data.pascal_voc import VOC2012
from src.model.detection_loss import yolo_loss
from src.nn.binarized_conv2d import BinarizedConv2d
from src.nn.binarized_linear import BinarizedLinear
from src.utils import load_class, make_logger

logger = make_logger(name=str(__name__))


class BinarizedLinearBlock(nn.Module):
    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        bias: bool = False,
        batch_norm: bool = False,
        activation: Optional[Dict] = None,
        mode: str = "stochastic",
    ) -> None:
        super(BinarizedLinearBlock, self).__init__()

        self.binarized_linear = BinarizedLinear(in_features=in_feature, out_features=out_feature, bias=bias, mode=mode)

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_features=in_feature)

        self.activation = activation
        if self.activation:
            self.activation = getattr(nn, activation["type"])(**activation["args"])

    def forward(self, x):
        if self.batch_norm:
            x = self.batch_norm(x)

        x = self.binarized_linear(x)

        if self.activation:
            x = self.activation(x)

        return x


class BinarizedConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        batch_norm: bool = False,
        activation: Optional[Dict] = None,
        pool: Optional[Dict] = None,
        mode: str = "stochastic",
    ) -> None:
        super(BinarizedConvBlock, self).__init__()

        self.binarized_conv = BinarizedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            mode=mode,
        )

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm2d(num_features=in_channels)

        self.activation = activation
        if self.activation:
            self.activation = getattr(nn, activation["type"])(**activation["args"])

        self.pool = pool
        if self.pool:
            # yaml not supported tuple. omegaconf too
            pool_dict = dict(pool)

            kernel_size = tuple(list(pool.args.kernel_size))

            old_args = pool_dict.pop("args", None)
            new_args = {}
            for key in old_args.keys():
                if key == "kernel_size":
                    continue
                new_args.update({key: old_args[key]})
            new_args.update({"kernel_size": kernel_size})
            pool_dict.update({"args": new_args})

            self.pool = getattr(nn, pool_dict["type"])(**pool_dict["args"])

    def forward(self, x):
        logger.warning(f"x shape : {x.shape}")
        if self.batch_norm:
            logger.warning(f"num feature in batchnorm: {self.batch_norm.num_features}")
            x = self.batch_norm(x)

        x = self.binarized_conv(x)

        if self.activation:
            x = self.activation(x)

        if self.pool:
            x = self.pool(x)

        return x


def _build_conv_layers(conv_layers_config):
    return nn.ModuleList([BinarizedConvBlock(**params) for params in conv_layers_config])


def _build_linear_layers(linear_layers_config):
    return nn.ModuleList([BinarizedLinearBlock(**params) for params in linear_layers_config])


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
        if hasattr(self, "linear_layers"):
            if self.conv_layers:
                for conv_layer in self.conv_layers:
                    x = conv_layer(x)

        x = x.view(x.size()[0], -1)

        if hasattr(self, "linear_layers"):
            for linear_layer in self.linear_layers:
                x = linear_layer(x)

        x = x.view(-1, 7, 7, 30)

        return x

    def loss(self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor, image_sizes: torch.Tensor):
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
