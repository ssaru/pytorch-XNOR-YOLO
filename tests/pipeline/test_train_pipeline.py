import logging
import os
import sys

import pytest
import pytorch_lightning
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from src.data.dataloader import get_data_loaders
from src.engine.train_jig import TrainingContainer
from src.nn.binarized_conv2d import BinarizedConv2d
from src.utils import build_model, get_config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def tearup_conv_config() -> DictConfig:
    config_path = {
        "--data-config": "conf/data/data.yml",
        "--model-config": "conf/model/model.yml",
        "--training-config": "conf/training/training.yml",
    }
    config_list = ["--data-config", "--model-config", "--training-config"]
    config: DictConfig = get_config(hparams=config_path, options=config_list)

    return config


train_test_case = [
    # (config, gpus)
    (tearup_conv_config(), None),
    (tearup_conv_config(), 0),
]


@pytest.mark.parametrize("config, gpus", train_test_case)
def test_train_pipeline(fix_seed, config, gpus):
    config = OmegaConf.create(config)

    train_dataloader, test_dataloader = get_data_loaders(config=config)
    model = build_model(model_conf=config.model)
    logger.debug(f"config.trainer in test_train_pipeline: {config.keys()}")
    training_container = TrainingContainer(model=model, config=config)

    trainer_params = dict(config.runner.trainer.params)
    trainer_params["limit_train_batches"] = 0.1
    trainer_params["limit_val_batches"] = 0.1
    trainer_params["max_epochs"] = 2
    trainer_params["gpus"] = gpus
    if not gpus:
        trainer_params["accelerator"] = None

    trainer = Trainer(**trainer_params)

    trainer.fit(
        model=training_container,
        train_dataloader=train_dataloader,
        val_dataloaders=test_dataloader,
    )
