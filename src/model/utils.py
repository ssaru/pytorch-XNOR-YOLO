from typing import Dict, Optional

import torch
import torch.nn as nn

from src.nn.binarized_conv2d import BinarizedConv2d
from src.nn.binarized_linear import BinarizedLinear
from src.utils import make_logger

logger = make_logger(name=str(__name__))


class LinearBlock(nn.Module):
    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        bias: bool = False,
        batch_norm: bool = False,
        activation: Optional[Dict] = None,
        dropout: Optional[Dict] = None,
    ) -> None:
        super(LinearBlock, self).__init__()

        self.linear = nn.Linear(in_features=in_feature, out_features=out_feature, bias=bias)
        self.batch_norm = batch_norm
        if self.batch_norm:            
            self.batch_norm = nn.BatchNorm1d(num_features=out_feature)

        self.activation = activation
        if self.activation:
            self.activation = getattr(nn, activation["type"])(**activation["args"])

        self.dropout = dropout
        if self.dropout:
            self.dropout = getattr(nn, dropout["type"])(**dropout["args"])

    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
                    
        x = self.linear(x)
        
        if self.batch_norm:
            x = self.batch_norm(x)

        if self.activation:
            x = self.activation(x)

        return x


class Conv2dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        batch_norm: bool = False,
        activation: Optional[Dict] = None,
        pool: Optional[Dict] = None,
    ) -> None:
        super(Conv2dBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm2d(num_features=out_channels)

        self.activation = activation
        if self.activation:
            self.activation = getattr(nn, activation["type"])(**activation["args"])

        self.pool = pool
        if self.pool:
            # yaml not supported tuple. omegaconf too
            pool_dict = dict(pool)
            kernel_size = tuple(list(pool.args.kernel_size))
            old_args = pool_dict.pop("args", None)
            new_args = {}
            for key in old_args.keys():
                if key == "kernel_size":
                    continue
                new_args.update({key: old_args[key]})
            new_args.update({"kernel_size": kernel_size})
            pool_dict.update({"args": new_args})

            self.pool = getattr(nn, pool_dict["type"])(**pool_dict["args"])

    def forward(self, x):
        logger.info(f"x shape : {x.shape}")
        x = self.conv(x)

        if self.batch_norm:
            logger.info(f"num feature in batchnorm: {self.batch_norm.num_features}")
            x = self.batch_norm(x)

        if self.activation:
            x = self.activation(x)

        if self.pool:
            x = self.pool(x)

        return x


class BinarizedLinearBlock(nn.Module):
    def __init__(
        self,
        in_feature: int,
        out_feature: int,
        bias: bool = False,
        batch_norm: bool = False,
        activation: Optional[Dict] = None,
        mode: str = "stochastic",
    ) -> None:
        super(BinarizedLinearBlock, self).__init__()

        self.binarized_linear = BinarizedLinear(in_features=in_feature, out_features=out_feature, bias=bias, mode=mode)

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_features=in_feature)

        self.activation = activation
        if self.activation:
            self.activation = getattr(nn, activation["type"])(**activation["args"])

    def forward(self, x):
        if self.batch_norm:
            x = self.batch_norm(x)

        x = self.binarized_linear(x)

        if self.activation:
            x = self.activation(x)

        return x


class BinarizedConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        batch_norm: bool = False,
        activation: Optional[Dict] = None,
        pool: Optional[Dict] = None,
        mode: str = "stochastic",
    ) -> None:
        super(BinarizedConvBlock, self).__init__()

        self.binarized_conv = BinarizedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            mode=mode,
        )

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm2d(num_features=in_channels)

        self.activation = activation
        if self.activation:
            self.activation = getattr(nn, activation["type"])(**activation["args"])

        self.pool = pool
        if self.pool:
            # yaml not supported tuple. omegaconf too
            pool_dict = dict(pool)

            kernel_size = tuple(list(pool.args.kernel_size))

            old_args = pool_dict.pop("args", None)
            new_args = {}
            for key in old_args.keys():
                if key == "kernel_size":
                    continue
                new_args.update({key: old_args[key]})
            new_args.update({"kernel_size": kernel_size})
            pool_dict.update({"args": new_args})

            self.pool = getattr(nn, pool_dict["type"])(**pool_dict["args"])

    def forward(self, x):
        logger.info(f"x shape : {x.shape}")
        if self.batch_norm:
            logger.info(f"num feature in batchnorm: {self.batch_norm.num_features}")
            x = self.batch_norm(x)

        x = self.binarized_conv(x)

        if self.activation:
            x = self.activation(x)

        if self.pool:
            x = self.pool(x)

        return x


def get_boxes(pred_tensor: torch.Tensor, confidence_score: float = 0.3):        
    predictions = []

    classes = torch.argmax(pred_tensor[:,:,:,11:], dim=3).float() + 1
    unique = torch.unique(classes)    
    
    for specific_class in unique:
        is_background = (specific_class == 0)
        if is_background:
            continue

        indices = torch.where(classes==specific_class)        
        boxes, scores = [], []        
        for b, y, x in zip(*indices):
            # 이미지를 벗어나는 box를 clipping해야함            
            scores1 = pred_tensor[b,y,x,0]
            boxes1 = pred_tensor[b,y,x,1:5] # shape: (1, 4)            
            boxes.append(boxes1)
            scores.append(scores1)                                                                                 
            
            scores2 = pred_tensor[b,y,x,5]            
            boxes2 = pred_tensor[b,y,x,6:10] # shape: (1, 4)                        
            boxes.append(boxes2)                                    
            scores.append(scores2)            
            
                
        if (len(boxes) == 0) or (len(scores) == 0):
            continue
            
        boxes = torch.stack(boxes)
        scores = torch.stack(scores)        
        
        indices, scores = soft_nms(dets=boxes, box_scores=scores, thresh=confidence_score)

        for idx in indices:        
            if scores[idx] < confidence_score:
                continue

            pred_confidence = float(scores[idx])
            pred_box = boxes[idx].int().tolist()            
            pred_classes = int(specific_class)

            xmin, ymin, xmax, ymax = pred_box
            xmin = xmin if xmin > 0 else 0
            ymin = ymin if ymin > 0 else 0
            xmax = xmax if 448 < xmax else 448
            ymax = ymax if 448 < ymax else 448
            pred_box = [xmin, ymin, xmax, ymax]

            predictions.append([pred_confidence, pred_box, pred_classes])

    return predictions


def soft_nms(dets, box_scores, sigma=0.1, thresh=0.001):
    """
    Build a pytorch implement of Soft NMS algorithm.
    refers: https://github.com/DocF/Soft-NMS/blob/master/softnms_pytorch.py
    
    # Augments
        dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
        box_scores:  box score tensors
        sigma:       variance of Gaussian function
        thresh:      score thresh
        cuda:        CUDA flag
    # Return
        the index of the selected boxes
    """
    dets = dets.clone()
    # Indexes concatenate boxes with the last column
    N = dets.shape[0]

    cuda = True if "cuda" in str(dets.device) else False
    if cuda:
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
    else:
        indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
    dets = torch.cat((dets, indexes), dim=1)

    # The order of boxes coordinate is [y1,x1,y2,x2]
    y1 = dets[:, 0]
    x1 = dets[:, 1]
    y2 = dets[:, 2]
    x2 = dets[:, 3]
    scores = box_scores
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tscore = scores[i].clone()
        pos = i + 1

        if i != N - 1:
            maxscore, maxpos = torch.max(scores[pos:], dim=0)
            if tscore < maxscore:
                dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
                scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

        # IoU calculate
        yy1 = torch.maximum(dets[i, 0], dets[pos:, 0])
        xx1 = torch.maximum(dets[i, 1], dets[pos:, 1])
        yy2 = torch.minimum(dets[i, 2], dets[pos:, 2])
        xx2 = torch.minimum(dets[i, 3], dets[pos:, 3])
        
        w = torch.maximum(torch.tensor(0.0), xx2 - xx1 + 1)
        h = torch.maximum(torch.tensor(0.0), yy2 - yy1 + 1)
        inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
        ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

        # Gaussian decay
        weight = torch.exp(-(ovr * ovr) / sigma)
        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    keep = dets[:, 4][scores > thresh].int()

    return keep, scores
    