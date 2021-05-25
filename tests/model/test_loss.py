import sys

import pytest
import torch
import torch.nn.functional as F

from src.model.detection_loss import yolo_loss
from src.utils import make_logger

logger = make_logger(name=__name__)

test_yololoss_testcase = [
    (
        (
            [1, 3],
            [
                1,
                0.01,
                0.01,
                0.10,
                0.10,
                *F.one_hot(torch.tensor(2), 20).tolist(),
            ],
        ),
        (
            [1, 3],
            [
                0.3,
                0.06,
                0.06,
                0.20,
                0.20,
                0.5,
                0.02,
                0.02,
                0.08,
                0.08,
                *F.one_hot(torch.tensor(2), 20).tolist(),
            ],
        ),
    ),
]


@pytest.mark.parametrize("target_info, output_info", test_yololoss_testcase)
def test_yolo_loss(target_info, output_info):

    target_dummy = torch.zeros((1, 7, 7, 25), dtype=torch.float32)
    output_dummy = torch.zeros((1, 7, 7, 30), dtype=torch.float32)

    x, y = target_info[0]
    target_obj_block = torch.tensor(target_info[1], dtype=torch.float32)
    target_dummy[0, y, x, :] = target_obj_block
    logger.info(f"shape of target: {target_dummy.shape}")
    logger.info(
        f"target block of specific indexes: {type(target_dummy[0, y, x, :])}{target_dummy[0, y, x, :]}"
    )

    x, y = output_info[0]
    output_obj_block = torch.tensor(output_info[1], dtype=torch.float32)
    output_dummy[0, y, x, :] = output_obj_block
    logger.info(f"shape of output: {output_dummy.shape}")
    logger.info(
        f"target block of specific indexes: {type(output_dummy[0, y, x, :])}{output_dummy[0, y, x, :]}"
    )

    loss_dict = yolo_loss(output=output_dummy, target=target_dummy)
    logger.info(f"field of loss_dict : {loss_dict.keys()}")
    obj_loss_val = loss_dict["lambda_obj"] * torch.sum(
        torch.stack([*loss_dict["obj_loss"].values()])
    )
    nonobj_loss_val = loss_dict["lambda_noobj"] * torch.sum(
        torch.stack([*loss_dict["nonobj_loss"].values()])
    )

    logger.info(f"loss dict: {loss_dict}")
    logger.info(f"obj loss value: {obj_loss_val}")
    logger.info(f"nonobj loss value: {nonobj_loss_val}")

    assert torch.allclose(loss_dict["obj_loss"]["confidence1_loss"], torch.tensor(0.49))
    assert torch.allclose(loss_dict["obj_loss"]["box1_cx_loss"], torch.tensor(0.0025))
    assert torch.allclose(loss_dict["obj_loss"]["box1_cy_loss"], torch.tensor(0.0025))
    assert torch.allclose(
        loss_dict["obj_loss"]["box1_width_loss"], torch.tensor(0.01350889)
    )
    assert torch.allclose(
        loss_dict["obj_loss"]["box1_height_loss"], torch.tensor(0.01350889)
    )

    assert torch.allclose(loss_dict["obj_loss"]["confidence2_loss"], torch.tensor(0.25))
    assert torch.allclose(loss_dict["obj_loss"]["box2_cx_loss"], torch.tensor(0.0001))
    assert torch.allclose(loss_dict["obj_loss"]["box2_cy_loss"], torch.tensor(0.0001))
    assert torch.allclose(
        loss_dict["obj_loss"]["box2_width_loss"], torch.tensor(0.05580357)
    )
    assert torch.allclose(
        loss_dict["obj_loss"]["box2_height_loss"], torch.tensor(0.05580357)
    )

    assert torch.allclose(nonobj_loss_val, torch.tensor(0.0))
