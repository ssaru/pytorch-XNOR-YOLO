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


def compute_average_precision(recall, precision):
    """Compute AP for one class.
    Args:
        recall: (numpy array) recall values of precision-recall curve.
        precision: (numpy array) precision values of precision-recall curve.
    Returns:
        (float) average precision (AP) for the class.
    """
    # AP (AUC of precision-recall curve) computation using all points interpolation.
    # For mAP computation, you can find a great explaination below.
    # https://github.com/rafaelpadilla/Object-Detection-Metrics

    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))

    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    ap = 0.0  # average precision (AUC of the precision-recall curve).
    for i in range(precision.size - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]

    return ap


def compute_mAP(preds, targets, class_names, threshold=0.5):
    """Compute mAP metric.
    Args:
        preds: (dict) {class_name_1: [[filename, prob, x1, y1, x2, y2], ...], class_name_2: [[], ...], ...}.
        targets: (dict) {(filename, class_name): [[x1, y1, x2, y2], ...], ...}.
        class_names: (list) list of class names.
        threshold: (float) threshold for IoU to separate TP from FP.
    Returns:
        (list of float) list of average precision (AP) for each class.
    """
    # For mAP computation, you can find a great explaination below.
    # https://github.com/rafaelpadilla/Object-Detection-Metrics

    aps = {}  # list of average precisions (APs) for each class.

    for class_name in class_names:
        class_preds = preds[class_name]  # all predicted objects for this class.

        if len(class_preds) == 0:
            ap = 0.0  # if no box detected, assigne 0 for AP of this class.
            # print('---class {} AP {}---'.format(class_name, ap))
            aps[class_name] = ap
            # break

        image_fnames = [pred[0] for pred in class_preds]
        probs = [pred[1] for pred in class_preds]
        boxes = [pred[2:] for pred in class_preds]

        # Sort lists by probs.
        sorted_idxs = np.argsort(probs)[::-1]
        image_fnames = [image_fnames[i] for i in sorted_idxs]
        boxes = [boxes[i] for i in sorted_idxs]

        # Compute total number of ground-truth boxes. This is used to compute precision later.
        num_gt_boxes = 0
        for (filename_gt, class_name_gt) in targets:
            if class_name_gt == class_name:
                num_gt_boxes += len(targets[filename_gt, class_name_gt])

        # Go through sorted lists, classifying each detection into TP or FP.
        num_detections = len(boxes)
        tp = np.zeros(num_detections)  # if detection `i` is TP, tp[i] = 1. Otherwise, tp[i] = 0.
        fp = np.ones(num_detections)  # if detection `i` is FP, fp[i] = 1. Otherwise, fp[i] = 0.

        for det_idx, (filename, box) in enumerate(zip(image_fnames, boxes)):

            if (filename, class_name) in targets:
                boxes_gt = targets[(filename, class_name)]
                for box_gt in boxes_gt:
                    # Compute IoU b/w/ predicted and groud-truth boxes.
                    inter_x1 = max(box_gt[0], box[0])
                    inter_y1 = max(box_gt[1], box[1])
                    inter_x2 = min(box_gt[2], box[2])
                    inter_y2 = min(box_gt[3], box[3])
                    inter_w = max(0.0, inter_x2 - inter_x1 + 1.0)
                    inter_h = max(0.0, inter_y2 - inter_y1 + 1.0)
                    inter = inter_w * inter_h

                    area_det = (box[2] - box[0] + 1.0) * (box[3] - box[1] + 1.0)
                    area_gt = (box_gt[2] - box_gt[0] + 1.0) * (box_gt[3] - box_gt[1] + 1.0)
                    union = area_det + area_gt - inter

                    iou = inter / union
                    if iou >= threshold:
                        tp[det_idx] = 1.0
                        fp[det_idx] = 0.0

                        boxes_gt.remove(
                            box_gt
                        )  # each ground-truth box can be assigned for only one detected box.
                        if len(boxes_gt) == 0:
                            del targets[
                                (filename, class_name)
                            ]  # remove empty element from the dictionary.

                        break

            else:
                pass  # this detection is FP.

        # Compute AP from `tp` and `fp`.
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        eps = np.finfo(np.float64).eps
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, eps)
        recall = tp_cumsum / float(num_gt_boxes)

        ap = compute_average_precision(recall, precision)
        # print('---class {} AP {}---'.format(class_name, ap))
        aps[class_name] = ap

    # Compute mAP by averaging APs for all classes.
    mAP = np.mean(np.array(list(aps.values())))
    # print('---mAP {}---'.format(mAP))

    return mAP, aps


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
        # period=1000,
        save_top_k=5,
        save_weights_only=False,
        monitor="valid_loss",
        mode="min",
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
