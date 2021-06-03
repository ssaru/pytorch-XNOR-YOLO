import sys
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from src.data.pascal_voc import VOC2012
from src.utils import make_logger


def xyxyabs_to_xywhrel(
    boxes: Union[torch.Tensor, np.ndarray, list],
    image_sizes: Union[torch.Tensor, np.array, list],
) -> list:
    current_func_name = sys._getframe().f_code.co_name
    logger = make_logger(name=current_func_name)

    is_boxes_numpy, is_image_sizes_numpy = isinstance(boxes, np.ndarray), isinstance(image_sizes, np.ndarray)
    is_boxes_torch, is_image_sizes_torch = isinstance(boxes, torch.Tensor), isinstance(image_sizes, torch.Tensor)
    is_boxes_list, is_image_sizes_list = isinstance(boxes, list), isinstance(image_sizes, list)
    logger.info(f"boxes type: {type(boxes)}, image_sizes type: {type(image_sizes)}")

    if is_boxes_numpy or is_boxes_list:
        arr = torch.from_numpy(np.asarray(boxes, dtype=np.float32)).clone()
    elif is_boxes_torch:
        arr = boxes.clone()
    else:
        logger.warning(f"{type(boxes)} is not supported")
        raise TypeError(f"{type(boxes)} is not supported")

    if is_image_sizes_numpy or is_image_sizes_list:
        sizes = torch.from_numpy(np.asarray(image_sizes, dtype=np.float32)).clone()
    elif is_image_sizes_torch:
        sizes = image_sizes.clone()
    else:
        logger.warning(f"{type(image_sizes)} is not supported")
        raise TypeError(f"{type(image_sizes)} is not supported")

    logger.info(f"arr : {type(arr)}:{arr}")
    logger.info(f"sizes : {type(sizes)}:{sizes}")

    if not (1 <= len(arr.shape) <= 2):
        raise RuntimeError(f"shape of boxes not supported. {arr.shape}")

    if not (len(sizes.shape) == 1):
        raise RuntimeError(f"shape of boxes not supported. {sizes.shape}")

    if len(arr.shape) == 1:
        arr = arr.unsqueeze(0)

    img_width = sizes[0]
    img_height = sizes[1]

    box_xmin = arr[:, 0]
    box_ymin = arr[:, 1]
    box_xmax = arr[:, 2]
    box_ymax = arr[:, 3]

    logger.info(
        f"img_width: {img_width}, img_height:{img_height}, xmin:{box_xmin}, ymin:{box_ymin}, xmax:{box_xmax}, ymax:{box_ymax}"
    )

    box_cx = torch.div((box_xmin + box_xmax) / 2, img_width)
    box_cy = torch.div((box_ymin + box_ymax) / 2, img_height)
    box_width = torch.div((box_xmax - box_xmin), img_width)
    box_height = torch.div((box_ymax - box_ymin), img_height)

    arr[:, 0] = box_cx
    arr[:, 1] = box_cy
    arr[:, 2] = box_width
    arr[:, 3] = box_height

    return arr


def build_label_tensor(xywhrel_boxes: torch.Tensor) -> torch.Tensor:
    current_func_name = sys._getframe().f_code.co_name
    logger = make_logger(name=current_func_name)

    if not 1 <= len(xywhrel_boxes.shape) <= 2:
        raise RuntimeError(
            f"Not mathed xywhrel_boxes shape. length of shape should be 1-2. but {len(xywhrel_boxes.shape)}, {xywhrel_boxes.shape}"
        )

    if len(xywhrel_boxes.shape) == 1:
        xywhrel_boxes = xywhrel_boxes.unsqueeze(0)

    num_boxes = xywhrel_boxes.shape[0]

    dx, dy = 1 / 7, 1 / 7
    label_tensor = torch.zeros((7, 7, 31))

    for i in range(num_boxes):
        objectness = 1
        x_shift = xywhrel_boxes[i][0] % dx
        y_shift = xywhrel_boxes[i][1] % dy
        coord_label = [
            objectness,
            x_shift,
            y_shift,
            xywhrel_boxes[i][2],
            xywhrel_boxes[i][3],
        ]
        obj_cell_x, obj_cell_y = (
            int(xywhrel_boxes[i][0] // dx),
            int(xywhrel_boxes[i][1] // dy),
        )
        logger.info(
            f"obj_cell_x: {obj_cell_x}, obj_cell_y: {obj_cell_y}, x_shift: {x_shift}, y_shift: {y_shift}, coord_label: {coord_label}"
        )

        label_tensor[obj_cell_y][obj_cell_x][0:10] = torch.tensor([*coord_label, *coord_label])
        label_tensor[obj_cell_y][obj_cell_x][10:] = torch.tensor([*xywhrel_boxes[i][4:]])

    logger.info(f"generated label_tensor: {label_tensor.shape}:{label_tensor}")

    return label_tensor


class Yolofy(object):
    def __init__(self, resize_sizes: Tuple = (448, 448)):
        self.logger = make_logger(name=str(__class__))
        self._to_tensor = transforms.ToTensor()
        self._voc2012 = VOC2012()
        self._resize_sizes = resize_sizes

    def __call__(self, image: Image, target: Dict):
        self.logger.info(f"image: {type(image)}:{image}")
        self.logger.info(f"target: {type(target)}:{target}")

        image = image.resize(self._resize_sizes)

        target = OmegaConf.create(target)

        self.logger.info(f"omegaconf target: {type(target)}:{target}")
        image_width = target.annotation.size.width
        image_height = target.annotation.size.height

        self.logger.info(f"object info : {type(target.annotation.object)}:{target.annotation.object}")
        ans_target = []
        for object_info in target.annotation.object:
            cls_idx = torch.tensor(self._voc2012[str(object_info.name)])
            num_cls = len(self._voc2012)
            cls_onehot_vector = F.one_hot(cls_idx, num_cls).tolist()

            xmin = object_info.bndbox.xmin
            ymin = object_info.bndbox.ymin
            xmax = object_info.bndbox.xmax
            ymax = object_info.bndbox.ymax
            ans_target.append([xmin, ymin, xmax, ymax, *cls_onehot_vector])

        xywhrel_boxes = xyxyabs_to_xywhrel(boxes=ans_target, image_sizes=[image_width, image_height])
        ans_target = build_label_tensor(xywhrel_boxes=xywhrel_boxes)
        ans_image = self._to_tensor(image)

        self.logger.info(f"ans image info : {type(ans_image)}, {ans_image.shape}:{ans_image}")
        self.logger.info(f"ans target info : {type(ans_target)}, {ans_target.shape}:{ans_target}")

        return ans_image, ans_target
