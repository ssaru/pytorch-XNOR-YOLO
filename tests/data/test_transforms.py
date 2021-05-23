import sys
from typing import Dict, List

import pytest
import torch
from PIL import Image
from typing_extensions import runtime

from src.data.transforms import Yolofy, build_label_tensor, xyxyabs_to_xywhrel
from src.utils import make_logger


def test_transforms(ready_images):
    current_func_name = sys._getframe().f_code.co_name
    logger = make_logger(name=current_func_name)

    filepath, target = ready_images
    logger.info(f"filepath: {filepath}, target: {target}")

    image = Image.open(str(filepath))
    yolofy = Yolofy()
    image, target = yolofy(image, target)

    logger.info(f"image: {image}, target: {target}")

    assert isinstance(image, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert target.shape == (7, 7, 30)


xyxyabs_testcase = [
    ([10, 10, 50, 100], [300, 300], torch.tensor([0.1, 0.1833, 0.1333, 0.3]))
]


@pytest.mark.parametrize("xyxyabs_boxes, image_sizes, expected_boxes", xyxyabs_testcase)
def test_xyxyabs_to_xywhrel(xyxyabs_boxes, image_sizes, expected_boxes):
    current_func_name = sys._getframe().f_code.co_name
    logger = make_logger(name=current_func_name)

    logger.info(
        f"args: xyxyabs_boxes->{xyxyabs_boxes}, image_sizes->{image_sizes}, expected_boxes->{expected_boxes}"
    )
    ans_boxes = xyxyabs_to_xywhrel(boxes=xyxyabs_boxes, image_sizes=image_sizes)
    logger.info(f"ans_boxes->{ans_boxes}, expected_boxes->{expected_boxes}")

    assert torch.allclose(ans_boxes, expected_boxes, rtol=1e-04, atol=1e-04)


build_label_testcase = [
    (
        torch.tensor(
            [
                [
                    0.1,
                    0.1833,
                    0.1333,
                    0.3,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ]
        )
    )
]


@pytest.mark.parametrize("xywhrel_boxes", build_label_testcase)
def test_build_label_tensor(xywhrel_boxes):
    current_func_name = sys._getframe().f_code.co_name
    logger = make_logger(name=current_func_name)

    expected_label_tensor = torch.zeros((7, 7, 30))
    dx, dy = 1 / 7, 1 / 7

    objectness = 1
    x_shift = xywhrel_boxes[0][0] % dx
    y_shift = xywhrel_boxes[0][1] % dy
    coord_label = [
        objectness,
        x_shift,
        y_shift,
        xywhrel_boxes[0][2],
        xywhrel_boxes[0][3],
    ]
    obj_cell_x, obj_cell_y = (
        int(xywhrel_boxes[0][0] // dx),
        int(xywhrel_boxes[0][1] // dy),
    )

    expected_label_tensor[obj_cell_y][obj_cell_x][0:10] = torch.tensor(
        [*coord_label, *coord_label]
    )
    expected_label_tensor[obj_cell_y][obj_cell_x][10:] = torch.tensor(
        [*xywhrel_boxes[0][4:]]
    )
    logger.info(
        f"expected_label_tensor: {expected_label_tensor.shape}:{expected_label_tensor[obj_cell_y][obj_cell_x]}"
    )

    label_tensor = build_label_tensor(xywhrel_boxes=xywhrel_boxes)

    logger.info(
        f"label tensor: {label_tensor.shape}: {label_tensor[obj_cell_y][obj_cell_x]}"
    )

    assert label_tensor.shape == (7, 7, 30)
    assert torch.allclose(label_tensor, expected_label_tensor, rtol=1e-04, atol=1e-04)
