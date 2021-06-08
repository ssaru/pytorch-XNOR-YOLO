from typing import Tuple

import torch
import torch.nn as nn
import torchvision
import numpy as np
from omegaconf import DictConfig
from PIL import Image
import numpy as np

from src import model as Net
from src.utils import load_class


def build_model(model_conf: DictConfig):
    return load_class(module=Net, name=model_conf.type, args={"model_config": model_conf})


class Predictor(torch.nn.Module):
    def __init__(self, config: DictConfig, conf_thresh=0.1, prob_thresh=0.1, nms_thresh=0.35) -> None:
        """Model Container for predict

        Args:
            model (nn.Module): model for train
            config (DictConfig): configuration with Omegaconf.DictConfig format for dataset/model/runner
        """
        super().__init__()
        print(f"=======CONFIG=======")
        print(config)
        print(f"====================")
        self.model: nn.Module = build_model(model_conf=config.model)
        self.resize_size = (448,448)
        self.mean = np.array([122.67891434, 116.66876762, 104.00698793], dtype=np.float32)
        self.conf_thresh=conf_thresh
        self.prob_thresh=prob_thresh
        self.nms_thresh=nms_thresh

    def forward(self, x):        
        return self.model.inference(x, image_size=self.image_size, conf_thresh=self.conf_thresh, prob_thresh=self.prob_thresh, nms_thresh=self.nms_thresh)

    def preprocess(self, image: Image):
        self.image_size = image.size
        image = image.resize(self.resize_size)
        image = np.array(image)
        image = (image - self.mean) / 255.0
        # image = Image.fromarray(image)
        image = torchvision.transforms.ToTensor()(image).unsqueeze(0)
        # image = torch.div((image - self.mean), 255)
        return image
