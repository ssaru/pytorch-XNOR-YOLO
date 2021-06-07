import sys
from typing import Dict, List, Tuple

import torch

from src.utils import make_logger

logger = make_logger(name=__name__)


def calc_obj_loss(output: torch.Tensor, target: torch.Tensor) -> dict:

    return {
        "confidence1_loss": torch.sum(torch.pow(output[:, :, :, 0] - target[:, :, :, 0], 2)),
        "box1_cx_loss": torch.sum(torch.pow(output[:, :, :, 1] - target[:, :, :, 1], 2)),
        "box1_cy_loss": torch.sum(torch.pow(output[:, :, :, 2] - target[:, :, :, 2], 2)),
        "box1_width_loss": torch.sum(torch.pow(output[:, :, :, 3] - torch.sqrt(target[:, :, :, 3]), 2)),
        "box1_height_loss": torch.sum(torch.pow(output[:, :, :, 4] - torch.sqrt(target[:, :, :, 4]), 2)),
        "confidence2_loss": torch.sum(torch.pow(output[:, :, :, 5] - target[:, :, :, 0], 2)),
        "box2_cx_loss": torch.sum(torch.pow(output[:, :, :, 6] - target[:, :, :, 1], 2)),
        "box2_cy_loss": torch.sum(torch.pow(output[:, :, :, 7] - target[:, :, :, 2], 2)),
        "box2_width_loss": torch.sum(torch.pow(output[:, :, :, 8] - torch.sqrt(target[:, :, :, 3]), 2)),
        "box2_height_loss": torch.sum(torch.pow(output[:, :, :, 9] - torch.sqrt(target[:, :, :, 4]), 2)),
        "classes_loss": torch.sum(torch.pow(output[:, :, :, 10:] - target[:, :, :, 10:], 2)),
    }


def calc_nonobj_loss(output: torch.Tensor, target: torch.Tensor) -> dict:
    return {
        "confidence1_loss": torch.sum(torch.pow(output[:, :, :, 0] - target[:, :, :, 0], 2)),
        "box1_cx_loss": torch.sum(torch.pow(output[:, :, :, 1] - target[:, :, :, 1], 2)),
        "box1_cy_loss": torch.sum(torch.pow(output[:, :, :, 2] - target[:, :, :, 2], 2)),
        "box1_width_loss": torch.sum(torch.pow(output[:, :, :, 3] - torch.sqrt(target[:, :, :, 3]), 2)),
        "box1_height_loss": torch.sum(torch.pow(output[:, :, :, 4] - torch.sqrt(target[:, :, :, 4]), 2)),
        "confidence2_loss": torch.sum(torch.pow(output[:, :, :, 5] - target[:, :, :, 0], 2)),
        "box2_cx_loss": torch.sum(torch.pow(output[:, :, :, 6] - target[:, :, :, 1], 2)),
        "box2_cy_loss": torch.sum(torch.pow(output[:, :, :, 7] - target[:, :, :, 2], 2)),
        "box2_width_loss": torch.sum(torch.pow(output[:, :, :, 8] - torch.sqrt(target[:, :, :, 3]), 2)),
        "box2_height_loss": torch.sum(torch.pow(output[:, :, :, 9] - torch.sqrt(target[:, :, :, 4]), 2)),
        "classes_loss": torch.sum(torch.pow(output[:, :, :, 10:] - target[:, :, :, 10:], 2)),
    }


def calc_yolo_loss(pred_tensor: torch.Tensor, target_tensor: torch.Tensor, image_sizes: Tuple = (448,448)) -> dict:
    """
    output: (n1, 7, 7, 31)
    label: (n2, 7, 7, 31), where n1==n2

    iou^{truth}_{pred} * Pr(Object)값이 왔다고 가정한다.
    pred_width, pred_height값이 음수일 때, sqrt가 에러가 날 수 있으므로,
    sqrt된 target값에 맞춰 학습하고, 추론할 때 pred-width와 pred-height를 제곱한다.
    """
    logger.info(f"shape of args: output-> {pred_tensor.shape}, target -> {pred_tensor.shape}")
    lambda_obj = 5.0
    lambda_noobj = 0.5

    obj_mask = torch.stack([target_tensor[:, :, :, 0] for _ in range(31)], dim=3)
    noobj_mask = torch.neg(obj_mask - 1)

    logger.info(f"shape of obj_mask : {obj_mask.shape}, shape of noobj_mask: {noobj_mask.shape}")
    obj_output_block = obj_mask * pred_tensor
    obj_target_block = obj_mask * target_tensor
    nonobj_output_block = noobj_mask * pred_tensor
    nonobj_target_block = noobj_mask * target_tensor

    obj_output_block: torch.Tensor = calc_confidence(
            pred_tensor=obj_output_block, target_tensor=target_tensor, image_sizes=image_sizes
        )

    logger.info(f"calculate each loss, obj and nonobj")
    obj_loss_dict = calc_obj_loss(output=obj_output_block, target=obj_target_block)
    nonobj_loss_dict = calc_nonobj_loss(output=nonobj_output_block, target=nonobj_target_block)

    logger.info(f"obj_loss_dict : {obj_loss_dict}")
    logger.info(f"nonobj_loss_dict: {nonobj_loss_dict}")

    obj_loss = (
        obj_loss_dict["box1_cx_loss"]
        + obj_loss_dict["box1_cy_loss"]
        + obj_loss_dict["box1_width_loss"]
        + obj_loss_dict["box1_height_loss"]
        + obj_loss_dict["box2_cx_loss"]
        + obj_loss_dict["box2_cy_loss"]
        + obj_loss_dict["box2_width_loss"]
        + obj_loss_dict["box2_height_loss"]
    )
    obj_loss *= lambda_obj
    
    nonobj_loss = (
        nonobj_loss_dict["box1_cx_loss"]
        + nonobj_loss_dict["box1_cy_loss"]
        + nonobj_loss_dict["box1_width_loss"]
        + nonobj_loss_dict["box1_height_loss"]
        + nonobj_loss_dict["box2_cx_loss"]
        + nonobj_loss_dict["box2_cy_loss"]
        + nonobj_loss_dict["box2_width_loss"]
        + nonobj_loss_dict["box2_height_loss"])
    nonobj_loss *= lambda_noobj

    total_loss = (
        obj_loss
        + nonobj_loss
        + obj_loss_dict["classes_loss"]
        + nonobj_loss_dict["classes_loss"]
        + obj_loss_dict["confidence1_loss"]
        + obj_loss_dict["confidence2_loss"]
        + nonobj_loss_dict["confidence1_loss"]
        + nonobj_loss_dict["confidence2_loss"]
    )

    batch_size = pred_tensor.shape[0]
    total_loss /= batch_size

    return {
        "total_loss": total_loss,
        "lambda_obj": lambda_obj,
        "lambda_noobj": lambda_noobj,
        "obj_loss": obj_loss_dict,
        "nonobj_loss": nonobj_loss_dict,
    }


def calc_iou(pred_box: torch.Tensor, target_box: torch.Tensor):
    """
    single pred box and target_box
    """
    pred_box = pred_box.clone()
    target_box = target_box.clone()

    with torch.no_grad():
        pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        target_area = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])

        target_xmin, target_ymin, target_xmax, target_ymax = target_box
        pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_box

        intersection_x_length = torch.min(target_xmax, pred_xmax) - torch.max(target_xmin, pred_xmin)
        intersection_y_length = torch.min(target_ymax, pred_ymax) - torch.max(target_ymin, pred_ymin)

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
    with torch.no_grad():
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


def yolotensor_to_xyxyabs(yolo_coord_output: torch.Tensor, image_sizes: Tuple = (448, 448)) -> torch.Tensor:
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
    image_width, image_height = image_sizes

    boxes = []

    with torch.no_grad():
        for idx in range(batch):
            cell_indices = torch.where(yolo_coord_output[idx, :, :, 0] > 0)
            dx, dy = image_width / 7, image_height / 7

            # TODO. for loop를 더 줄일 수 있을 것 같은데,,,
            for ys, xs in zip(*cell_indices):
                box1_x_shift, box1_y_shift, box1_width, box1_height = yolo_coord_output[idx, ys, xs, 1:5]

                box1_cx, box1_cy = (dx * xs + box1_x_shift * dx), (dy * ys + box1_y_shift * dy)
                box1_width, box1_height = (box1_width * image_width), (box1_height * image_height)

                box1_xmin, box1_ymin = (box1_cx - box1_width / 2), (box1_cy - box1_height / 2)
                box1_xmax, box1_ymax = (box1_cx + box1_width / 2), (box1_cy + box1_height / 2)
                yolo_coord_output[idx, ys, xs, 1:5] = torch.tensor(
                    [box1_xmin, box1_ymin, box1_xmax, box1_ymax], dtype=torch.float32
                )

                box2_x_shift, box2_y_shift, box2_width, box2_height = yolo_coord_output[idx, ys, xs, 6:10]

                box2_cx, box2_cy = (dx * xs + box2_x_shift * dx), (dy * ys + box2_y_shift * dy)
                box2_width, box2_height = (image_width * box2_width), (image_height * box2_height)

                box2_xmin, box2_ymin = (box2_cx - box2_width / 2), (box2_cy - box2_height / 2)
                box2_xmax, box2_ymax = (box2_cx + box2_width / 2), (box2_cy + box2_height / 2)
                yolo_coord_output[idx, ys, xs, 6:10] = torch.tensor(
                    [box2_xmin, box2_ymin, box2_xmax, box2_ymax], dtype=torch.float32
                )

                box1 = yolo_coord_output[idx, ys, xs, 1:5].clone()
                box2 = yolo_coord_output[idx, ys, xs, 6:10].clone()
                boxes.append(((torch.tensor(idx), ys, xs), box1, box2))

    del yolo_coord_output

    return boxes


def calc_confidence(
    pred_tensor: torch.Tensor, target_tensor: torch.Tensor, image_sizes: Tuple = (448, 448)
) -> torch.Tensor:

    indices = torch.where(target_tensor[:, :, :, 0] > 0)
    pred_boxes: List = yolotensor_to_xyxyabs(yolo_coord_output=pred_tensor, image_sizes=image_sizes)
    target_boxes: List = yolotensor_to_xyxyabs(yolo_coord_output=target_tensor, image_sizes=image_sizes)

    for b, y, x in zip(*indices):
        gt_box = target_boxes[b, y, x, 1:5]
        box1 = pred_boxes[b, y, x, 1:5]
        box2 = pred_boxes[b, y, x, 6:10]

        box1_iou = calc_iou(pred_box=box1, target_box=gt_box)
        box2_iou = calc_iou(pred_box=box2, target_box=gt_box)

        pred_tensor[b,y,x,0] *= box1_iou
        pred_tensor[b,y,x,5] *= box2_iou        

    return pred_tensor


def yolo_loss(pred_tensor: torch.Tensor, target_tensor: torch.Tensor, image_sizes: Tuple = (448, 448)) -> Dict:        
    
    loss_dict = calc_yolo_loss(pred_tensor=pred_tensor, target_tensor=target_tensor)
    return loss_dict
