import sys

import torch

from src.utils import make_logger

logger = make_logger(name=__name__)


def calc_obj_loss(output: torch.Tensor, target: torch.Tensor) -> dict:
    return {
        "confidence1_loss": torch.sum(
            torch.pow(output[:, :, :, 0] - target[:, :, :, 0], 2)
        ),
        "box1_cx_loss": torch.sum(
            torch.pow(output[:, :, :, 1] - target[:, :, :, 1], 2)
        ),
        "box1_cy_loss": torch.sum(
            torch.pow(output[:, :, :, 2] - target[:, :, :, 2], 2)
        ),
        "box1_width_loss": torch.sum(
            torch.pow(output[:, :, :, 3] - torch.sqrt(target[:, :, :, 3]), 2)
        ),
        "box1_height_loss": torch.sum(
            torch.pow(output[:, :, :, 4] - torch.sqrt(target[:, :, :, 4]), 2)
        ),
        "confidence2_loss": torch.sum(
            torch.pow(output[:, :, :, 5] - target[:, :, :, 0], 2)
        ),
        "box2_cx_loss": torch.sum(
            torch.pow(output[:, :, :, 6] - target[:, :, :, 1], 2)
        ),
        "box2_cy_loss": torch.sum(
            torch.pow(output[:, :, :, 7] - target[:, :, :, 2], 2)
        ),
        "box2_width_loss": torch.sum(
            torch.pow(output[:, :, :, 8] - torch.sqrt(target[:, :, :, 3]), 2)
        ),
        "box2_height_loss": torch.sum(
            torch.pow(output[:, :, :, 9] - torch.sqrt(target[:, :, :, 4]), 2)
        ),
        "classes_loss": torch.sum(
            torch.pow(output[:, :, :, 10:] - target[:, :, :, 10:], 2)
        ),
    }


def calc_nonobj_loss(output: torch.Tensor, target: torch.Tensor) -> dict:
    return {
        "confidence1_loss": torch.sum(
            torch.pow(output[:, :, :, 0] - target[:, :, :, 0], 2)
        ),
        "confidence2_loss": torch.sum(
            torch.pow(output[:, :, :, 5] - target[:, :, :, 0], 2)
        ),
        "classes_loss": torch.sum(
            torch.pow(output[:, :, :, 10:] - target[:, :, :, 10:], 2)
        ),
    }


def yolo_loss(output: torch.Tensor, target: torch.Tensor) -> dict:
    """
    output: (n1, 7, 7, 30)
    label: (n2, 7, 7, 30), where n1==n2

    iou^{truth}_{pred} * Pr(Object)값이 왔다고 가정한다.
    pred_width, pred_height값이 음수일 때, sqrt가 에러가 날 수 있으므로,
    sqrt된 target값에 맞춰 학습하고, 추론할 때 pred-width와 pred-height를 제곱한다.
    """
    logger.info(f"shape of args: output-> {output.shape}, target -> {target.shape}")
    lambda_obj = 5.0
    lambda_noobj = 0.5

    with torch.no_grad():
        obj_mask = torch.stack([target[:, :, :, 0] for _ in range(30)], dim=3)
        noobj_mask = torch.neg(obj_mask - 1)

    logger.info(
        f"shape of obj_mask : {obj_mask.shape}, shape of noobj_mask: {noobj_mask.shape}"
    )
    obj_output_block = obj_mask * output
    obj_target_block = obj_mask * target
    nonobj_output_block = noobj_mask * output
    nonobj_target_block = noobj_mask * target

    logger.info(f"calculate each loss, obj and nonobj")
    obj_loss_dict = calc_obj_loss(output=obj_output_block, target=obj_target_block)
    nonobj_loss_dict = calc_nonobj_loss(
        output=nonobj_output_block, target=nonobj_target_block
    )

    logger.info(f"obj_loss_dict : {obj_loss_dict}")
    logger.info(f"nonobj_loss_dict: {nonobj_loss_dict}")

    return {
        "lambda_obj": lambda_obj,
        "lambda_noobj": lambda_noobj,
        "obj_loss": obj_loss_dict,
        "nonobj_loss": nonobj_loss_dict,
    }
