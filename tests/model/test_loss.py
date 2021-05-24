import sys

import pytest
import torch
import torch.nn.functional as F

from src.model.detection_loss import yolo_loss
from src.utils import make_logger

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
    current_func_name = sys._getframe().f_code.co_name
    logger = make_logger(name=current_func_name)

    target_dummy = torch.zeros((1, 7, 7, 30), dtype=torch.float32)
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
    logger.info(f"shape of target: {output_dummy.shape}")
    logger.info(
        f"target block of specific indexes: {type(output_dummy[0, y, x, :])}{output_dummy[0, y, x, :]}"
    )

    loss = yolo_loss(output=target_dummy, target=target_dummy)

    assert loss == True
