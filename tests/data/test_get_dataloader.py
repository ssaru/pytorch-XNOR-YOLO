import sys

import pytest
import torch
from omegaconf import OmegaConf

from src.data.dataloader import get_data_loaders
from src.utils import make_logger

dataloader_testcase = [
    (OmegaConf.load("conf/data/data.yml"), OmegaConf.load("conf/training/training.yml"))
]


@pytest.mark.parametrize("data_config, training_config", dataloader_testcase)
def test_dataloader(data_config, training_config):
    current_func_name = sys._getframe().f_code.co_name
    logger = make_logger(name=str(current_func_name))

    config = OmegaConf.create()
    config.update(data_config)
    config.update(training_config)
    logger.info(f"fields: {config.keys()}, config : {config}")

    config.dataloader.params.batch_size = 1
    config.dataloader.params.num_worker = 0
    logger.info(
        f"change batch size to {config.dataloader.params.batch_size} & num worker to {config.dataloader.params.num_worker}"
    )

    train_dataloader, test_dataloader = get_data_loaders(config=config)

    train_dataset = train_dataloader.dataset
    test_dataset = test_dataloader.dataset

    logger.info(f"train_dataset : {type(train_dataset)}:{train_dataset}")
    logger.info(f"test_dataset : {type(test_dataset)}:{test_dataset}")

    train_image, train_target = train_dataset.__getitem__(0)
    test_image, test_target = test_dataset.__getitem__(0)

    logger.info(
        f"image : {type(train_image)}:{train_image}, target: {type(train_target)}{train_target}"
    )
    logger.info(
        f"image : {type(test_image)}:{test_image}, target: {type(test_target)}{test_target}"
    )

    assert isinstance(train_image, torch.Tensor)
    assert isinstance(train_target, torch.Tensor)
    assert isinstance(test_image, torch.Tensor)
    assert isinstance(test_target, torch.Tensor)
