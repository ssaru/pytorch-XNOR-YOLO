import sys

from omegaconf import OmegaConf

from src.data.dataloader import get_data_loaders
from src.utils import make_logger


def visualize_gt(
    data_config: str = "conf/data/data.yml",
    training_config: str = "conf/data/training.yml",
):
    current_func_name = sys._getframe().f_code.co_name
    logger = make_logger(name=current_func_name)

    config = OmegaConf.create()
    config.update(OmegaConf.load(data_config))
    config.update(OmegaConf.load(training_config))

    train_dataloader, test_dataloader = get_data_loaders(config=config)
    pass
