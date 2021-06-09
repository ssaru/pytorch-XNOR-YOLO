"""
Usage:
    main.py evaluate [options] [--config=<model config path>] [--weights-filepath=<weights file path>] [--image-path=<image path>]
    main.py evaluate (-h | --help)
Options:
    --config <model config path>  Path to YAML file for model configuration  [default: pretrained_model/TINY-YOLO/config.yaml] [type: path]
    --weights-filepath <weights file path>  Path to weights file for model  [default: pretrained_model/TINY-YOLO/Yolo_epoch=90-train_loss=3.86-val_loss=2.25.ckpt] [type: path]    
            
    -h --help  Show this.
"""
import sys
import subprocess
import traceback
from pathlib import Path
from collections import defaultdict

import pytorch_lightning
import numpy as np
import torch
from omegaconf import DictConfig
from torchvision.datasets import VOCDetection
from omegaconf import OmegaConf
from PIL import ImageFile 
ImageFile.LOAD_TRUNCATED_IMAGES = True

from src.engine.predictor import Predictor
from src.data.pascal_voc import VOC2012
from src.data.transforms import Yolofy
from src.utils import get_config

pytorch_lightning.seed_everything(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def compute_average_precision(recall, precision):
    """ Compute AP for one class.
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
        precision[i - 1] = max(precision[i -1], precision[i])

    ap = 0.0 # average precision (AUC of the precision-recall curve).
    for i in range(precision.size - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]

    return ap


def compute_mAP(preds,targets,class_names,threshold=0.5):
    """ Compute mAP metric.
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

    aps = {} # list of average precisions (APs) for each class.

    for class_name in class_names:
        class_preds = preds[class_name] # all predicted objects for this class.

        if len(class_preds) == 0:
            ap = 0.0 # if no box detected, assigne 0 for AP of this class.
            # print('---class {} AP {}---'.format(class_name, ap))
            aps[class_name] = ap
            # break

        image_fnames = [pred[0]  for pred in class_preds]
        probs        = [pred[1]  for pred in class_preds]
        boxes        = [pred[2:] for pred in class_preds]

        # Sort lists by probs.
        sorted_idxs = np.argsort(probs)[::-1]
        image_fnames = [image_fnames[i] for i in sorted_idxs]
        boxes        = [boxes[i]        for i in sorted_idxs]

        # Compute total number of ground-truth boxes. This is used to compute precision later.
        num_gt_boxes = 0
        for (filename_gt, class_name_gt) in targets:
            if class_name_gt == class_name:
                num_gt_boxes += len(targets[filename_gt, class_name_gt])

        # Go through sorted lists, classifying each detection into TP or FP.
        num_detections = len(boxes)
        tp = np.zeros(num_detections) # if detection `i` is TP, tp[i] = 1. Otherwise, tp[i] = 0.
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

                        boxes_gt.remove(box_gt) # each ground-truth box can be assigned for only one detected box.
                        if len(boxes_gt) == 0:
                            del targets[(filename, class_name)] # remove empty element from the dictionary.

                        break

            else:
                pass # this detection is FP.

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

def evaluate(hparams: dict):
    weight_filepath = str(hparams.get("--weights-filepath"))

    config_list = ["--config"]
    config: DictConfig = get_config(hparams=hparams, options=config_list)
    
    predictor = Predictor(config=config, conf_thresh=0.01, prob_thresh=0.01, nms_thresh=0.5)
        
    if weight_filepath:
        predictor.load_state_dict(torch.load(weight_filepath, map_location="cpu")["state_dict"])
            
    predictor.eval()
            
    voc2012 = VOCDetection(root="data", year="2007", image_set="val", download=False)
    
    targets = defaultdict(list)
    preds = defaultdict(list)


    for image, target in voc2012:                
        target = OmegaConf.create(target)        
        filename = str(target.annotation.filename)
        
        # Preparing ground-truth data...
        for object_info in target.annotation.object:
            classes = object_info.name                                

            xmin = float(object_info.bndbox.xmin)
            ymin = float(object_info.bndbox.ymin)
            xmax = float(object_info.bndbox.xmax)
            ymax = float(object_info.bndbox.ymax)

            targets[(filename, classes)].append([xmin, ymin, xmax, ymax])                

        # Predicting...
        with torch.no_grad():
            input_image: torch.Tensor = predictor.preprocess(image)
        
        predictions = predictor(input_image)
        boxes_detected, class_names_detected, probs_detected = predictor(input_image)
        
        
        for box, class_name, prob in zip(boxes_detected, class_names_detected, probs_detected):        
            xmin, ymin, xmax, ymax = box            
            preds[class_name].append([filename, prob, xmin, ymin, xmax, ymax])            
    
    print("Evaluate the detection result...")
    voc_class_names = predictor.model.class_name_list
    mAP, APs = compute_mAP(preds, targets, class_names=voc_class_names)
    print(f"mAP: {mAP}")
    print(f"APs: {APs}")

# def evaluate(hparams: dict):
#     weight_filepath = str(hparams.get("--weights-filepath"))

#     config_list = ["--config"]
#     config: DictConfig = get_config(hparams=hparams, options=config_list)
    
#     predictor = Predictor(config=config, conf_thresh=0.01, prob_thresh=0.01, nms_thresh=0.5)
        
#     if weight_filepath:
#         predictor.load_state_dict(torch.load(weight_filepath, map_location="cpu")["state_dict"])
            
#     predictor.eval()
            
#     voc2012 = VOCDetection(root="data", year="2007", image_set="val", download=False)    

#     root_path = Path("src/measure/input/")
#     pred_path = root_path / Path("detection-results")
#     gt_path = root_path / Path("ground-truth")

    
#     for image, target in voc2012:                
#         target = OmegaConf.create(target)        
#         filename = Path(str(target.annotation.filename)).with_suffix(".txt")        
        
#         gt_filepath = gt_path / filename        
#         gt_filepath.parent.mkdir(parents=True, exist_ok=True)
#         with gt_filepath.open("w") as gt_file:                    
#             for object_info in target.annotation.object:
#                 classes = object_info.name                                

#                 xmin = object_info.bndbox.xmin
#                 ymin = object_info.bndbox.ymin
#                 xmax = object_info.bndbox.xmax
#                 ymax = object_info.bndbox.ymax

#                 write_line = f"{classes} {xmin} {ymin} {xmax} {ymax}\n"
#                 gt_file.write(write_line)                

#         with torch.no_grad():
#             input_image: torch.Tensor = predictor.preprocess(image)
        
#         predictions = predictor(input_image)
#         boxes_detected, class_names_detected, probs_detected = predictor(input_image)

#         pred_filepath = pred_path / filename
#         pred_filepath.parent.mkdir(parents=True, exist_ok=True)
#         with pred_filepath.open("w") as pred_file:
#             for idx, box in enumerate(boxes_detected):        
#                 xmin, ymin, xmax, ymax = box
#                 confidence = probs_detected[idx]
#                 classes = class_names_detected[idx]

#                 write_line = f"{classes} {confidence} {xmin} {ymin} {xmax} {ymax}\n"        
#                 pred_file.write(write_line)                
    
        
#     subprocess.check_call(["python3", "src/measure/main.py"])