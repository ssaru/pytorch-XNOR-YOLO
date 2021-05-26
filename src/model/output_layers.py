from typing import List, Tuple

import torch

from src.utils import make_logger

logger = make_logger(name=__name__)


def calc_iou(pred_box: torch.Tensor, target_box: torch.Tensor):
    """
    single pred box and target_box
    """

    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    target_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])

    target_xmin, target_ymin, target_xmax, target_ymax = target_box
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_box

    intersection_x_length = torch.min(target_xmax, pred_xmax) - torch.max(
        target_xmin, pred_xmin
    )
    intersection_y_length = torch.min(target_ymax, pred_ymax) - torch.max(
        target_ymin, pred_ymin
    )

    intersection_area = intersection_x_length * intersection_y_length
    union_area = pred_area + target_area - intersection_area

    if intersection_x_length <= 0 or intersection_y_length <= 0:
        return torch.tensor(0)

    return intersection_area / union_area


def pairwise_iou(pred_boxes: List[Tuple], target_boxes: List[Tuple]):
    """
    pred_boxes: (indices: Tuple[torch.Tensor], box1: torch.Tensor, box2: torch.Tensor ...)
    target_boxes: (indices: Tuple[torch.Tensor], box: torch.Tensor ...)
    """

    ious = []
    for target_info in target_boxes:
        target_box = target_info[1]

        for pred_info in pred_boxes:
            pred_indices = pred_info[0]
            pred_box1 = pred_info[1]
            pred_box2 = pred_info[2]
            iou_box1 = calc_iou(pred_box=pred_box1, target_box=target_box)
            iou_box2 = calc_iou(pred_box=pred_box2, target_box=target_box)
            ious.append((pred_indices, iou_box1, iou_box2))

    return ious


def yolotensor_to_xyxyabs(
    yolo_coord_output: torch.Tensor, image_sizes: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        yolo_coord_output (torch.Tensor): (n,7,7,30) shape of tensor. it represented output for yolo network
        image_sizes (torch.Tensor): (n,w,h) shape of tensor. it describes image sizes

    Returns:
        (torch.Tensor): (n,7,7,30) converted xyxyabs boxes coordinates torch tensor

    YOLO 논문에서 출력은 7x7x30 tensor라고 서술하고있음
    Batch size를 고려한다면, PyTorch로 구현한 YOLO의 출력은 Batchx7x7x30.
    이 때, 1x1x30 tensor block만 고려한다면 각 원소의 값들은 아래와 같음

    ---------------------------------------------------------------------------------------------------------------------------------------------------
    |                                          box1                                                                   | box2 |        classes         |
    ---------------------------------------------------------------------------------------------------------------------------------------------------
    | confidence | pixel x shift from cell | pixel y shift from cell | relative width of box | relative height of box | ...  |   C0   |  ...  |   CN  |
    ---------------------------------------------------------------------------------------------------------------------------------------------------

    IOU를 구하려면 torch tensor output을 {xmin, ymin, xmax, ymax} 형태로 변경해줘야함

    TODO. 알고리즘에 대한 의도를 설명. 의사코드

    """
    logger.info(f"copy coordination output. shape is : {yolo_coord_output.shape}")
    yolo_coord_output = yolo_coord_output.clone()
    batch = yolo_coord_output.shape[0]

    boxes = []
    with torch.no_grad():
        for idx in range(batch):
            cell_indices = torch.where(yolo_coord_output[idx, :, :, 0] > 0)
            image_width, image_height = image_sizes[idx]
            dx, dy = image_width / 7, image_height / 7

            for ys, xs in zip(*cell_indices):
                box1_x_shift = yolo_coord_output[idx, ys, xs, 1]
                box1_y_shift = yolo_coord_output[idx, ys, xs, 2]
                box1_width = yolo_coord_output[idx, ys, xs, 3]
                box1_height = yolo_coord_output[idx, ys, xs, 4]

                box1_cx = dx * xs + box1_x_shift * dx
                box1_cy = dy * ys + box1_y_shift * dy
                box1_width = box1_width * image_width
                box1_height = box1_height * image_height

                box1_xmin = box1_cx - box1_width / 2
                box1_ymin = box1_cy - box1_height / 2
                box1_xmax = box1_cx + box1_width / 2
                box1_ymax = box1_cy + box1_height / 2

                yolo_coord_output[idx, ys, xs, 1] = box1_xmin
                yolo_coord_output[idx, ys, xs, 2] = box1_ymin
                yolo_coord_output[idx, ys, xs, 3] = box1_xmax
                yolo_coord_output[idx, ys, xs, 4] = box1_ymax
                box1 = yolo_coord_output[idx, ys, xs, 1:5]

                box2_x_shift = yolo_coord_output[idx, ys, xs, 6]
                box2_y_shift = yolo_coord_output[idx, ys, xs, 7]
                box2_width = yolo_coord_output[idx, ys, xs, 8]
                box2_height = yolo_coord_output[idx, ys, xs, 9]

                box2_cx = dx * xs + box2_x_shift * dx
                box2_cy = dy * ys + box2_y_shift * dy
                box2_width = image_width * box2_width
                box2_height = image_height * box2_height

                box2_xmin = box2_cx - box2_width / 2
                box2_ymin = box2_cy - box2_height / 2
                box2_xmax = box2_cx + box2_width / 2
                box2_ymax = box2_cy + box2_height / 2

                yolo_coord_output[idx, ys, xs, 6] = box2_xmin
                yolo_coord_output[idx, ys, xs, 7] = box2_ymin
                yolo_coord_output[idx, ys, xs, 8] = box2_xmax
                yolo_coord_output[idx, ys, xs, 9] = box2_ymax

                box2 = yolo_coord_output[idx, ys, xs, 6:10]
                boxes.append(((torch.tensor(idx), ys, xs), box1, box2))

    return boxes


class OutputLayer(object):
    def __init__(self):
        pass

    def __call__(self, output: torch.Tensor):
        pass
