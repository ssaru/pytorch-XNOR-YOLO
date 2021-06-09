"""
Usage:
    main.py train [options] [--dataset-config=<dataset config path>] [--model-config=<model config path>] [--runner-config=<runner config path>]
    main.py train (-h | --help)
Options:
    --dataset-config <dataset config path>  Path to YAML file for dataset configuration  [default: conf/data/data.yml] [type: path]
    --model-config <model config path>  Path to YAML file for model configuration  [default: conf/model/xnoryolo.yml] [type: path]
    --runner-config <runner config path>  Path to YAML file for model configuration  [default: conf/training/xnoryolo_training.yml] [type: path]
    --checkpoint-path <checkpoint path>  Path to model weight for resume  [default: None] [type: path]
    -h --help  Show this.
"""

import warnings

warnings.filterwarnings(action="ignore")
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler

from src.data.dataloader import get_data_loaders
from src.engine.train_jig import TrainingContainer
from src.utils import (
    build_model,
    get_checkpoint_callback,
    get_config,
    get_early_stopper,
    get_log_dir,
    get_wandb_logger,
)


# TODO. Augmentation 추가
def train(hparams: dict):
    config_list = ["--dataset-config", "--model-config", "--runner-config"]
    config: DictConfig = get_config(hparams=hparams, options=config_list)
    checkpoint_path = str(hparams.get("--checkpoint-path"))
    checkpoint_path = checkpoint_path if (checkpoint_path != "None") else None

    # TODO. 임시방편
    OmegaConf.set_readonly(config, False)
    for key, value in config.hyperparameter.items():
        config.runner.experiments.name += "-" + str(key) + "_" + str(value)

    log_dir = get_log_dir(config=config)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_dataloader, test_dataloader = get_data_loaders(config=config)
    dataset = test_dataloader

    model: nn.Module = build_model(model_conf=config.model)
    model.summary()

    training_container: pl.LightningModule = TrainingContainer(
        model=model, config=config, len_dataloader=len(train_dataloader)
    )

    checkpoint_callback = get_checkpoint_callback(log_dir=log_dir, config=config)
    wandb_logger = get_wandb_logger(log_dir=log_dir, config=config)
    # wandb_logger.watch(model, log="gradients", log_freq=100)

    lr_logger = LearningRateMonitor()
    # early_stop_callback = get_early_stopper(early_stopping_config=config.runner.earlystopping.params)

    # TODO. SimpleProfiler는 ddp spawn에서 문제가 발생 TextIO Error
    # profiler = SimpleProfiler(output_filename="perf.txt")

    with (log_dir / Path("config.yaml")).open("w") as f:
        OmegaConf.save(config=config, f=f)

    trainer = Trainer(
        log_every_n_steps=1,
        accelerator=config.runner.trainer.params.accelerator,
        fast_dev_run=False,
        gpus=config.runner.trainer.params.gpus,
        amp_level="O2",
        logger=wandb_logger,
        callbacks=[lr_logger],  # early_stop_callback
        checkpoint_callback=checkpoint_callback,
        max_epochs=config.runner.trainer.params.max_epochs,
        weights_summary="top",
        reload_dataloaders_every_epoch=False,
        resume_from_checkpoint=checkpoint_path,
        benchmark=False,
        deterministic=True,
        num_sanity_val_steps=0,
        overfit_batches=0.0,
        precision=32,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        # TODO. SimpleProfiler는 ddp spawn에서 문제가 발생 TextIO Error
        # profiler=profiler,
    )

    trainer.fit(
        model=training_container,
        train_dataloader=train_dataloader,
        val_dataloaders=test_dataloader,
    )
    trainer.save_checkpoint("final.ckpt")
