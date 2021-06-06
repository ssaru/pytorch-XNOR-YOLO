import sys
from typing import Tuple

import torchvision
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.transforms import Yolofy
from src.utils import load_class, make_logger


def get_data_loaders(config: DictConfig) -> Tuple[DataLoader, DataLoader]:
    current_func_name = sys._getframe().f_code.co_name
    logger = make_logger(name=str(current_func_name))

    args = dict(config.data.dataset.params)

    args["image_set"] = "train"

    args["transforms"] = Yolofy(resize_sizes=(config.model.params.width, config.model.params.height))
    train_dataset = load_class(module=torchvision.datasets, name=config.data.dataset.type, args=args)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=config.dataloader.params.batch_size,
        num_workers=config.dataloader.params.num_workers,
        drop_last=True,
        shuffle=True,
    )

    args["image_set"] = "val"
    args["download"] = False
    
    test_dataset = load_class(module=torchvision.datasets, name=config.data.dataset.type, args=args)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.dataloader.params.batch_size,
        num_workers=config.dataloader.params.num_workers,
        drop_last=False,
        shuffle=True,
    )
    return train_dataloader, test_dataloader
