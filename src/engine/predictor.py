from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from omegaconf import DictConfig
from PIL import Image

from src import model as Net
from src.utils import load_class


def build_model(model_conf: DictConfig):
    return load_class(module=Net, name=model_conf.type, args={"model_config": model_conf})


class Predictor(torch.nn.Module):
    def __init__(self, config: DictConfig) -> None:
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

    def forward(self, x):        
        predictions = self.model.inference(x, image_size=self.image_size)
        
        for idx, pred in enumerate(predictions):
            score, bbox, cls_id = pred
            xmin, ymin, xmax, ymax = bbox            
            
            norm_xmin = xmin / self.resize_size[0]
            norm_ymin = ymin / self.resize_size[1]
            norm_xmax = xmax / self.resize_size[0]
            norm_ymax = ymax / self.resize_size[1]

            recon_xmin = norm_xmin * self.image_size[0]
            recon_ymin = norm_ymin * self.image_size[1]
            recon_xmax = norm_xmax * self.image_size[0]
            recon_ymax = norm_ymax * self.image_size[1]

            predictions[idx][1] = [recon_xmin, recon_ymin, recon_xmax, recon_ymax]

        return predictions

    def preprocess(self, image: Image):
        self.image_size = image.size
        image = image.resize(self.resize_size)        
        return torchvision.transforms.ToTensor()(image).unsqueeze(0)
