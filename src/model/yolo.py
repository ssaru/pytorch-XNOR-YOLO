from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchsummary import summary as torch_summary

from src.data.pascal_voc import VOC2012
from src.model.detection_loss import Loss, yolo_loss, yolotensor_to_xyxyabs
from src.model.utils import Conv2dBlock, LinearBlock, get_boxes
from src.utils import make_logger

logger = make_logger(name=str(__name__))


def _build_conv_layers(conv_layers_config):
    return nn.ModuleList([Conv2dBlock(**params) for params in conv_layers_config])


def _build_linear_layers(linear_layers_config):
    return nn.ModuleList([LinearBlock(**params) for params in linear_layers_config])


class Yolo(nn.Module):
    def __init__(self, model_config: DictConfig) -> None:
        super(Yolo, self).__init__()

        self.class_map = VOC2012()
        self.class_name_list = list(self.class_map.inv_label.keys())

        self._confidence = model_config.params.confidence
        self._width: int = model_config.params.width
        self._height: int = model_config.params.height
        self._channels: int = model_config.params.channels

        self.input_shape: tuple = (self._channels, self._height, self._width)
        self.in_channels: int = self._channels        

        self.S = 7
        self.B = 2
        self.C = 20

        logger.info(f"build conv layers")
        if model_config.params.feature_layers.conv:
            self.conv_layers: nn.ModuleList = _build_conv_layers(
                conv_layers_config=model_config.params.feature_layers.conv
            )

        logger.info(f"build linear layers")
        if model_config.params.feature_layers.linear:
            self.linear_layers: nn.ModuleList = _build_linear_layers(
                linear_layers_config=model_config.params.feature_layers.linear
            )

        logger.info(f"build loss layers")
        self.softmax = nn.Softmax(dim=0)
        self.loss_fn = Loss()

    def forward(self, x):
        if hasattr(self, "conv_layers"):
            if self.conv_layers:
                for conv_layer in self.conv_layers:
                    x = conv_layer(x)

        x = x.view(x.size()[0], -1)

        if hasattr(self, "linear_layers"):
            for linear_layer in self.linear_layers:
                x = linear_layer(x)

        x = x.view(-1, 7, 7, 30)

        x = torch.sigmoid(x)

        return x

    def loss(self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor, image_sizes: Tuple = (448, 448)):
        # return self.loss_fn(pred_tensor=pred_tensor, target_tensor=target_tensor, image_sizes=image_sizes)
        return self.loss_fn(pred_tensor=pred_tensor, target_tensor=target_tensor)

    def inference(self, x: torch.Tensor, image_size: Tuple, conf_thresh: float, prob_thresh: float, nms_thresh: float):
        self.conf_thresh = conf_thresh
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh

        _, _, h, w = x.shape

        # single inference
        pred_tensor = self(x)
        pred_tensor = pred_tensor.cpu().data
        pred_tensor = pred_tensor.squeeze(0) # squeeze batch dimension.        
        
        boxes_normalized_all, class_labels_all, confidences_all, class_scores_all = self.decode(pred_tensor)
        if boxes_normalized_all.size(0) == 0:
            print(f"if no box found, return empty lists.")
            return [], [], [] # if no box found, return empty lists.

        # Apply non maximum supression for boxes of each class.
        boxes_normalized, class_labels, probs = [], [], []

        for class_label in range(len(self.class_name_list)):
            mask = (class_labels_all == class_label)
            if torch.sum(mask) == 0:
                continue # if no box found, skip that class.

            boxes_normalized_masked = boxes_normalized_all[mask]
            class_labels_maked = class_labels_all[mask]
            confidences_masked = confidences_all[mask]
            class_scores_masked = class_scores_all[mask]

            ids = self.nms(boxes_normalized_masked, confidences_masked)            
            
            boxes_normalized.append(boxes_normalized_masked[ids])
            class_labels.append(class_labels_maked[ids])
            probs.append(confidences_masked[ids] * class_scores_masked[ids])
        
        boxes_normalized = torch.cat(boxes_normalized, 0)
        class_labels = torch.cat(class_labels, 0)
        probs = torch.cat(probs, 0)        

        # Postprocess for box, labels, probs.
        boxes_detected, class_names_detected, probs_detected = [], [], []
        for b in range(boxes_normalized.size(0)):
            box_normalized = boxes_normalized[b]
            class_label = class_labels[b]
            prob = probs[b]

            x1, x2 = w * box_normalized[0], w * box_normalized[2] # unnormalize x with image width.
            y1, y2 = h * box_normalized[1], h * box_normalized[3] # unnormalize y with image height.
            boxes_detected.append(torch.tensor([x1, y1, x2, y2]).tolist())

            class_label = int(class_label) # convert from LongTensor to int.
            class_name = self.class_name_list[class_label]
            class_names_detected.append(class_name)

            prob = float(prob) # convert from Tensor to float.
            probs_detected.append(prob)
        
        return boxes_detected, class_names_detected, probs_detected

    def post_processing(self, x: torch.Tensor):
        raise NotImplementedError()

    def summary(self):
        # torchsummary only supported [cuda, cpu]. not cuda:0
        device = str(self.device).split(":")[0]
        torch_summary(
            self,
            input_size=(self._channels, self._height, self._width),
            device=device,
        )

    @property
    def device(self):
        devices = {param.device for param in self.parameters()} | {buf.device for buf in self.buffers()}
        if len(devices) != 1:
            raise RuntimeError("Cannot determine device: {} different devices found".format(len(devices)))
        return next(iter(devices))

    def decode(self, pred_tensor):
        """ Decode tensor into box coordinates, class labels, and probs_detected.
        Args:
            pred_tensor: (tensor) tensor to decode sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
        Returns:
            boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...]. Normalized from 0.0 to 1.0 w.r.t. image width/height, sized [n_boxes, 4].
            labels: (tensor) class labels for each detected boxe, sized [n_boxes,].
            confidences: (tensor) objectness confidences for each detected box, sized [n_boxes,].
            class_scores: (tensor) scores for most likely class for each detected box, sized [n_boxes,].
        """
        import numpy as np

        def softmax(x):
            e_x = torch.exp(x - torch.max(x))
            return e_x / torch.sum(e_x)

        S, B, C = self.S, self.B, self.C
        boxes, labels, confidences, class_scores = [], [], [], []

        cell_size = 1.0 / float(S)

        conf = pred_tensor[:, :, 4].unsqueeze(2) # [S, S, 1]
        for b in range(1, B):
            conf = torch.cat((conf, pred_tensor[:, :, 5*b + 4].unsqueeze(2)), 2)
        conf_mask = conf > self.conf_thresh # [S, S, B]

        # TBM, further optimization may be possible by replacing the following for-loops with tensor operations.
        for i in range(S): # for x-dimension.
            for j in range(S): # for y-dimension.                
                class_tensor = pred_tensor[j, i, 5*B:]
                mean = torch.mean(class_tensor)
                max_val, min_val = torch.max(class_tensor), torch.min(class_tensor)
                class_tensor = (class_tensor - mean) / (max_val - min_val)
                
                #class_score, class_label = torch.max(pred_tensor[j, i, 5*B:], 0)
                class_score, class_label = torch.max(class_tensor, 0)
                test = pred_tensor[j, i, 5*B:]                

                for b in range(B):
                    conf = pred_tensor[j, i, 5*b + 4]
                    prob = conf * class_score

                    if float(prob) < self.prob_thresh:                        
                        continue
                    
                    # Compute box corner (x1, y1, x2, y2) from tensor.
                    box = pred_tensor[j, i, 5*b : 5*b + 4]
                    x0y0_normalized = torch.FloatTensor([i, j]) * cell_size # cell left-top corner. Normalized from 0.0 to 1.0 w.r.t. image width/height.
                    xy_normalized = box[:2] * cell_size + x0y0_normalized   # box center. Normalized from 0.0 to 1.0 w.r.t. image width/height.
                    wh_normalized = box[2:] # Box width and height. Normalized from 0.0 to 1.0 w.r.t. image width/height.
                    box_xyxy = torch.FloatTensor(4) # [4,]
                    box_xyxy[:2] = xy_normalized - 0.5 * wh_normalized # left-top corner (x1, y1).
                    box_xyxy[2:] = xy_normalized + 0.5 * wh_normalized # right-bottom corner (x2, y2).

                    # Append result to the lists.
                    boxes.append(box_xyxy)
                    labels.append(class_label)
                    confidences.append(conf)
                    class_scores.append(class_score)                

        if len(boxes) > 0:
            boxes = torch.stack(boxes, 0) # [n_boxes, 4]
            labels = torch.stack(labels, 0)             # [n_boxes, ]
            confidences = torch.stack(confidences, 0)   # [n_boxes, ]
            class_scores = torch.stack(class_scores, 0) # [n_boxes, ]
        else:
            # If no box found, return empty tensors.
            boxes = torch.FloatTensor(0, 4)
            labels = torch.LongTensor(0)
            confidences = torch.FloatTensor(0)
            class_scores = torch.FloatTensor(0)

        return boxes, labels, confidences, class_scores

    def nms(self, boxes, scores):
        """ Apply non maximum supression.
        Args:
        Returns:
        """
        threshold = self.nms_thresh

        x1 = boxes[:, 0] # [n,]
        y1 = boxes[:, 1] # [n,]
        x2 = boxes[:, 2] # [n,]
        y2 = boxes[:, 3] # [n,]
        areas = (x2 - x1) * (y2 - y1) # [n,]

        _, ids_sorted = scores.sort(0, descending=True) # [n,]
        ids = []
        while ids_sorted.numel() > 0:
            # Assume `ids_sorted` size is [m,] in the beginning of this iter.

            i = ids_sorted.item() if (ids_sorted.numel() == 1) else ids_sorted[0]
            ids.append(i)

            if ids_sorted.numel() == 1:
                break # If only one box is left (i.e., no box to supress), break.

            inter_x1 = x1[ids_sorted[1:]].clamp(min=x1[i]) # [m-1, ]
            inter_y1 = y1[ids_sorted[1:]].clamp(min=y1[i]) # [m-1, ]
            inter_x2 = x2[ids_sorted[1:]].clamp(max=x2[i]) # [m-1, ]
            inter_y2 = y2[ids_sorted[1:]].clamp(max=y2[i]) # [m-1, ]
            inter_w = (inter_x2 - inter_x1).clamp(min=0) # [m-1, ]
            inter_h = (inter_y2 - inter_y1).clamp(min=0) # [m-1, ]

            inters = inter_w * inter_h # intersections b/w/ box `i` and other boxes, sized [m-1, ].
            unions = areas[i] + areas[ids_sorted[1:]] - inters # unions b/w/ box `i` and other boxes, sized [m-1, ].
            ious = inters / unions # [m-1, ]

            # Remove boxes whose IoU is higher than the threshold.
            ids_keep = (ious <= threshold).nonzero().squeeze() # [m-1, ]. Because `nonzero()` adds extra dimension, squeeze it.
            if ids_keep.numel() == 0:
                break # If no box left, break.
            ids_sorted = ids_sorted[ids_keep+1] # `+1` is needed because `ids_sorted[0] = i`.

        return torch.LongTensor(ids)
