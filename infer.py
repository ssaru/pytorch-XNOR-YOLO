"""
Usage:
    main.py predict [options] [--config=<model config path>] [--weights-filepath=<weights file path>] [--image-path=<image path>]
    main.py predict (-h | --help)
Options:
    --config <model config path>  Path to YAML file for model configuration  [default: pretrained_model/20210613/tiny-yolo-adam/config.yaml] [type: path]
    --weights-filepath <weights file path>  Path to weights file for model  [default: pretrained_model/20210613/tiny-yolo-adam/Yolo_epoch=158-train_loss=2.19-val_loss=0.00.ckpt] [type: path]
    --image-path <image path> Path to image filepath for inference  [default: data/VOCdevkit/VOC2012/JPEGImages/2009_002008.jpg]
            
    -h --help  Show this.
"""
# VOC2007
# 000007.jpg
#
# VOC2012
# 2009_002008.jpg, 2012_001533.jpg, 2007_000032.jpg, 2007_000027.jpg
# 2012_003028.jpg, 2012_004273.jpg, 2012_004331.jpg, 2012_004326.jpg
#
#    TINY-YOLO
#    Yolo_epoch=158-train_loss=2.19-val_loss=0.00.ckpt
#
#    TINY-XNOR-YOLO
#    XnorNetYolo_epoch=294-train_loss=2.57-val_loss=0.00.ckpt

import pytorch_lightning
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

from src.engine.predictor import Predictor
from src.utils import get_config

pytorch_lightning.seed_everything(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def infer(hparams: dict):
    weight_filepath = str(hparams.get("--weights-filepath"))
    image_path = str(hparams.get("--image-path"))

    config_list = ["--config"]
    config: DictConfig = get_config(hparams=hparams, options=config_list)
    
    predictor = Predictor(config=config, conf_thresh=0.1, prob_thresh=0.1, nms_thresh=0.3)
        
    if weight_filepath:
        predictor.load_state_dict(torch.load(weight_filepath, map_location="cpu")["state_dict"])
            
    predictor.eval()

    pil_image: Image = Image.open(image_path)
    
    with torch.no_grad():
        image: torch.Tensor = predictor.preprocess(pil_image)
    boxes_detected, class_names_detected, probs_detected = predictor(image)
    print(class_names_detected)
    draw = ImageDraw.Draw(pil_image)    
    for idx, box in enumerate(boxes_detected):
        xmin, ymin, xmax, ymax = box
        confidence = probs_detected[idx]
        classes = class_names_detected[idx]
    
        draw.rectangle(((int(xmin), int(ymin)), (int(xmax), int(ymax))), outline=(0, 0, 255), width=2)
        draw.text((int(xmin), int(ymin)), classes + ":" + f"{confidence:.2f}")

    pil_image.save("prediction.png")
