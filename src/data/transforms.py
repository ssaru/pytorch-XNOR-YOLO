import sys
from typing import Dict, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from src.data.pascal_voc import VOC2012
from src.model.detection_loss import yolotensor_to_xyxyabs
from src.model.yolo import Yolo
from src.utils import make_logger

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, x_max, y_max = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
    print(f"drawing... xmin, ymin ,xmax, ymax: {x_min, y_min, x_max, y_max}")
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test.png", img)


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

    with torch.no_grad():
        box_width = box_xmax - box_xmin
        box_height = box_ymax - box_ymin
        cx = box_xmin + (box_width / 2)
        cy = box_ymin + (box_height / 2)

        norm_cx = torch.div(cx, img_height)
        norm_cy = torch.div(cy, img_height)
        norm_width = torch.div(box_width, img_width)
        norm_height = torch.div(box_height, img_height)

    arr[:, 0] = norm_cx
    arr[:, 1] = norm_cy
    arr[:, 2] = norm_width
    arr[:, 3] = norm_height

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
        self._resize_width, self._resize_height = resize_sizes
        self.transform = A.Compose(
            [A.Resize(height=self._resize_width, width=self._resize_height, p=1)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
        )

    def __call__(self, image: Image, target: Dict):
        self.logger.info(f"image: {type(image)}:{image}")
        self.logger.info(f"target: {type(target)}:{target}")

        image = np.array(image)
        target = OmegaConf.create(target)

        self.logger.info(f"omegaconf target: {type(target)}:{target}")
        self.logger.info(f"object info : {type(target.annotation.object)}:{target.annotation.object}")

        bboxes, category_ids = [], []
        for object_info in target.annotation.object:
            cls_idx = self._voc2012[str(object_info.name)]
            num_cls = len(self._voc2012)

            xmin = int(object_info.bndbox.xmin)
            ymin = int(object_info.bndbox.ymin)
            xmax = int(object_info.bndbox.xmax)
            ymax = int(object_info.bndbox.ymax)
            bboxes.append([xmin, ymin, xmax, ymax])
            category_ids.append(cls_idx)

        transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
        image = transformed["image"]
        bboxes = transformed["bboxes"]
        category_ids = transformed["category_ids"]

        ans_target = []
        for idx, category_id in enumerate(category_ids):
            cls_onehot_vector = F.one_hot(torch.tensor(category_id), num_cls).tolist()
            box = bboxes[idx]
            ans_target.append([*box, *cls_onehot_vector])

        xywhrel_boxes = xyxyabs_to_xywhrel(boxes=ans_target, image_sizes=[self._resize_width, self._resize_height])
        ans_target = build_label_tensor(xywhrel_boxes=xywhrel_boxes)

        image = Image.fromarray(image)
        ans_image = self._to_tensor(image)

        self.logger.info(f"ans image info : {type(ans_image)}, {ans_image.shape}:{ans_image}")
        self.logger.info(f"ans target info : {type(ans_target)}, {ans_target.shape}:{ans_target}")

        return ans_image, ans_target  # , image, bboxes, category_ids, self._voc2012.label # for visualization


if __name__ == "__main__":
    from torchvision.datasets import VOCDetection

    voc2012 = VOCDetection(root="data", year="2012", image_set="train", download=False)
    yolo = Yolofy()

    for image, target in voc2012:
        ans_image, ans_target, image, bboxes, category_ids, label = yolo(image=image, target=target)
        print(f"output boxes: {bboxes}")
        visualize(np.array(image), bboxes, category_ids, label)
        break
