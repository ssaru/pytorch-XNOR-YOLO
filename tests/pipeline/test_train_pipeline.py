import logging

import pytest
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from src.data.dataloader import get_data_loaders
from src.engine.train_jig import TrainingContainer
from src.utils import build_model, get_config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


train_test_case = [
    # (gpus)
    (None),
    (0),
]


@pytest.mark.parametrize("gpus", train_test_case)
def test_train_xnoryolo_pipeline(fix_seed, gpus):
    config_path = {
        "--data-config": "conf/data/data.yml",
        "--model-config": "conf/model/xnoryolo.yml",
        "--training-config": "conf/training/training.yml",
    }
    config_list = ["--data-config", "--model-config", "--training-config"]
    config: DictConfig = get_config(hparams=config_path, options=config_list)

    train_dataloader, test_dataloader = get_data_loaders(config=config)
    model = build_model(model_conf=config.model)
    logger.debug(f"config.trainer in test_train_pipeline: {config.keys()}")
    training_container = TrainingContainer(model=model, config=config)

    trainer_params = dict(config.runner.trainer.params)
    trainer_params["limit_train_batches"] = 0.01
    trainer_params["limit_val_batches"] = 0.01
    trainer_params["fast_dev_run"] = True
    trainer_params["max_epochs"] = 2
    trainer_params["gpus"] = gpus
    if not gpus:
        trainer_params["accelerator"] = None

    trainer = Trainer(**trainer_params)

    trainer.fit(model=training_container, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)


@pytest.mark.parametrize("gpus", train_test_case)
def test_train_yolo_pipeline(fix_seed, gpus):
    config_path = {
        "--data-config": "conf/data/data.yml",
        "--model-config": "conf/model/yolo.yml",
        "--training-config": "conf/training/training.yml",
    }
    config_list = ["--data-config", "--model-config", "--training-config"]
    config: DictConfig = get_config(hparams=config_path, options=config_list)

    train_dataloader, test_dataloader = get_data_loaders(config=config)
    model = build_model(model_conf=config.model)
    logger.debug(f"config.trainer in test_train_pipeline: {config.keys()}")
    training_container = TrainingContainer(model=model, config=config)

    trainer_params = dict(config.runner.trainer.params)
    trainer_params["limit_train_batches"] = 0.01
    trainer_params["limit_val_batches"] = 0.01
    trainer_params["fast_dev_run"] = True
    trainer_params["max_epochs"] = 2
    trainer_params["gpus"] = gpus
    if not gpus:
        trainer_params["accelerator"] = None

    trainer = Trainer(**trainer_params)

    trainer.fit(model=training_container, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)
