import json
import logging
import logging.config
import operator
import sys
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import albumentations as A
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Dataset


def make_logger(name: Optional[str] = None, config_path: str = "conf/logger/logging.json"):
    with Path(config_path).open("rt") as f:
        config = json.load(f)

    logging.config.dictConfig(config)
    logger = logging.getLogger(name)

    return logger


logger = make_logger(name=__name__)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def build_model(model_conf: DictConfig):
    import src.model as Net

    # logging
    current_func_name = sys._getframe().f_code.co_name
    logger.debug(f"{current_func_name} : {model_conf}")

    return load_class(module=Net, name=model_conf.type, args={"model_config": model_conf})


def get_next_version(root_dir: Path) -> str:
    """generating folder name for managed version

    Args:
        root_dir (Path): saving directory for log, model checkpoint

    Returns:
        str: folder name for saving
    """

    # logging
    current_func_name = sys._getframe().f_code.co_name
    logger.debug(f"{current_func_name} : {root_dir}")

    version_prefix = "v"
    if not root_dir.exists():
        next_version = 0

    else:
        existing_versions = []
        for child_path in root_dir.iterdir():
            if child_path.is_dir() and child_path.name.startswith(version_prefix):
                existing_versions.append(int(child_path.name[len(version_prefix) :]))

        logger.debug(f"existing_versions: {existing_versions}")
        last_version = max(existing_versions) if len(existing_versions) > 0 else -1
        next_version = last_version + 1
        logger.debug(f"last_version: {last_version}")
        logger.debug(f"next_version: {next_version}")

    return f"{version_prefix}{next_version:0>3}"


def get_config(hparams: Dict, options: List) -> DictConfig:
    # logging
    current_func_name = sys._getframe().f_code.co_name
    logger.debug(f"{current_func_name} : hparams->{hparams}, options->{options}")

    config: DictConfig = OmegaConf.create()

    for option in options:
        option_config: DictConfig = OmegaConf.load(hparams.get(option))
        config.update(option_config)

    OmegaConf.set_readonly(config, True)

    return config


def get_log_dir(config: DictConfig) -> Path:
    # logging
    current_func_name = sys._getframe().f_code.co_name
    logger.debug(f"{current_func_name} : config -> {config}")

    root_dir = Path(config.runner.experiments.output_dir) / Path(
        config.runner.experiments.project_name
    )
    next_version = get_next_version(root_dir)
    run_dir = root_dir.joinpath(next_version)

    return run_dir


def get_checkpoint_callback(log_dir: Path, config: DictConfig) -> Union[Callback, List[Callback]]:
    # logging
    current_func_name = sys._getframe().f_code.co_name
    logger.debug(f"{current_func_name} : log_dir->{log_dir}, config->{config}")

    checkpoint_prefix = f"{config.model.type}"
    checkpoint_suffix = "_{epoch:02d}-{train_loss:.2f}-{val_loss:.2f}"

    checkpoint_path = log_dir.joinpath(checkpoint_prefix + checkpoint_suffix)
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        save_top_k=5,
        save_weights_only=False,
        monitor="mAP",
        mode="max",
    )

    return checkpoint_callback


def get_wandb_logger(log_dir: Path, config: DictConfig) -> Tuple[WandbLogger]:
    # logging
    current_func_name = sys._getframe().f_code.co_name
    logger.debug(f"{current_func_name} : log_dir->{log_dir}, config->{config}")

    next_version = str(log_dir.parts[-1])
    ids = log_dir.parts[-1]
    wandb_logger = WandbLogger(
        id=ids,
        name=str(config.runner.experiments.name),
        save_dir=str(log_dir),
        offline=False,
        version=next_version,
        project=str(config.runner.experiments.project_name),
    )

    return wandb_logger


def get_early_stopper(early_stopping_config: DictConfig) -> EarlyStopping:
    # logging
    current_func_name = sys._getframe().f_code.co_name
    logger.debug(f"{current_func_name} : early_stopping_config->{early_stopping_config}")

    return EarlyStopping(
        min_delta=0.00,
        patience=early_stopping_config.patience,
        verbose=early_stopping_config.verbose,
        mode=early_stopping_config.mode,
        monitor=early_stopping_config.monitor,
    )


def load_class(module: Any, name: str, args: Dict):
    # logging
    current_func_name = sys._getframe().f_code.co_name
    logger.debug(f"{current_func_name} : module->{module}, name->{name}, args->{args}")

    return getattr(module, name)(**args)
