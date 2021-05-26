import sys

import pytest
import torch
import torch.nn.functional as F

from src.model.output_layers import calc_iou, pairwise_iou, yolotensor_to_xyxyabs
from src.utils import make_logger

logger = make_logger(name=__name__)

test_calc_iou_testcase = [
    (
        torch.tensor([15.427, 101.14, 75.427, 161.14]),
        torch.tensor([28.28557, 113.99857, 58.28557, 143.99857]),
        torch.tensor(0.25),
    ),
    (
        torch.tensor([0, 0, 30, 30]),
        torch.tensor([10, 10, 20, 20]),
        torch.tensor(0.11111111),
    ),
]


@pytest.mark.parametrize("pred_box, target_box, expeceted", test_calc_iou_testcase)
def test_calc_iou(pred_box, target_box, expeceted):
    iou = calc_iou(pred_box=pred_box, target_box=target_box)
    logger.info(f"pred iou: {iou}, expected iou: {expeceted}")
    assert torch.allclose(iou, expeceted)


test_pairwise_iou_testcase = [
    (
        [
            (
                (torch.tensor(0), torch.tensor(3), torch.tensor(1)),
                torch.tensor([15.427, 101.14, 75.427, 161.14]),
                torch.tensor([15.427, 115.997, 32.714, 173.997]),
            )
        ],
        [
            (
                (torch.tensor(0), torch.tensor(1), torch.tensor(3)),
                torch.tensor([28.28557, 113.99857, 58.28557, 143.99857]),
            )
        ],
        (
            (torch.tensor(0), torch.tensor(3), torch.tensor(1)),
            torch.tensor(0.25),
            torch.tensor(0.06971),
        ),
    )
]


@pytest.mark.parametrize("pred_boxes, target_boxes, expected", test_pairwise_iou_testcase)
def test_pairwise_iou(pred_boxes, target_boxes, expected):

    ious = pairwise_iou(pred_boxes=pred_boxes, target_boxes=target_boxes)
    logger.info(f"ious: {ious}")
    expected_indices = expected[0]
    expected_box1_iou = expected[1]
    expected_box2_iou = expected[2]

    ious_indices, box1_iou, box2_iou = ious[0]
    assert ious_indices == expected_indices
    assert torch.allclose(box1_iou, expected_box1_iou, rtol=1e-01, atol=1e-01)
    assert torch.allclose(box2_iou, expected_box2_iou, rtol=1e-01, atol=1e-01)


test_yolotensor_to_xyxyabs_testcase = [
    (
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
            torch.tensor([[300, 300]]),
        ),
        (
            (torch.tensor(0), torch.tensor(3), torch.tensor(1)),
            torch.tensor([0.3, 15.427, 101.14, 75.427, 161.14]),
            torch.tensor([0.5, 31.714, 119.997, 55.714, 143.997]),
        ),
    )
]


@pytest.mark.parametrize("output_info, expected_info", test_yolotensor_to_xyxyabs_testcase)
def test_yolotensor_to_xyxyabs(output_info, expected_info):
    x, y = output_info[0]
    expected_box1 = torch.tensor(expected_info[1][1:])
    expected_box2 = torch.tensor(expected_info[2][1:])

    image_sizes = output_info[2]

    output_dummy = torch.zeros((1, 7, 7, 30), dtype=torch.float32)
    output_obj_block = torch.tensor(output_info[1], dtype=torch.float32)
    output_dummy[0, y, x, :] = output_obj_block
    logger.info(f"shape of output: {output_dummy.shape}")

    logger.info(f"output block of specific indexes: {type(output_dummy[0, y, x, :])}{output_dummy[0, y, x, :]}")

    boxes = yolotensor_to_xyxyabs(yolo_coord_output=output_dummy, image_sizes=image_sizes)
    logger.info(f"output of specific indexes: {boxes}")

    expected_idx, expected_ys, expected_xs = expected_info[0]
    idx, ys, xs = boxes[0][0]
    box1 = boxes[0][1]
    box2 = boxes[0][2]

    logger.info(f"expected_indexes: {expected_info[0]}")
    logger.info(f"output indexes: {boxes[0][0]}")

    logger.info(f"expeceted box1: {expected_box1}")
    logger.info(f"output box1: {box1}")

    logger.info(f"expeceted box2: {expected_box2}")
    logger.info(f"output box2: {box2}")

    assert torch.allclose(expected_idx, idx, rtol=1e-01, atol=1e-01)
    assert torch.allclose(expected_ys, ys, rtol=1e-01, atol=1e-01)
    assert torch.allclose(expected_xs, xs, rtol=1e-01, atol=1e-01)

    assert torch.allclose(box1, expected_box1, rtol=1e-01, atol=1e-01)
    assert torch.allclose(box2, expected_box2, rtol=1e-01, atol=1e-01)


test_output_layer_testcase = [
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


@pytest.mark.parametrize("target_info, output_info", test_output_layer_testcase)
def test_yolo_loss(target_info, output_info):
    pass
