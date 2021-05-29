import sys

import pytest
import torch
from omegaconf import OmegaConf

from src.data.dataloader import get_data_loaders
from src.utils import make_logger

dataloader_testcase = [
    (
        OmegaConf.load("conf/data/data.yml"),
        OmegaConf.load("conf/model/model.yml"),
        OmegaConf.load("conf/training/training.yml"),
    )
]


@pytest.mark.parametrize("data_config, model_config, training_config", dataloader_testcase)
def test_dataloader(data_config, model_config, training_config):
    current_func_name = sys._getframe().f_code.co_name
    logger = make_logger(name=str(current_func_name))

    config = OmegaConf.create()
    config.update(data_config)
    config.update(model_config)
    config.update(training_config)
    logger.info(f"fields: {config.keys()}, config : {config}")

    config.dataloader.params.batch_size = 1
    config.dataloader.params.num_worker = 0
    logger.info(
        f"change batch size to {config.dataloader.params.batch_size} & num worker to {config.dataloader.params.num_worker}"
    )

    train_dataloader, test_dataloader = get_data_loaders(config=config)

    for image, target in train_dataloader:
        logger.info(f"shape of image : {type(image)}:{image.shape}, target: {type(target)}{target.shape}")
        assert isinstance(image, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        break

    for image, target in test_dataloader:
        logger.info(f"shape of image : {type(image)}:{image.shape}, target: {type(target)}{target.shape}")
        assert isinstance(image, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        break
