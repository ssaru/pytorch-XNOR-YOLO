"""
Usage:
    main.py predict [options] [--config=<model config path>] [--weights-filepath=<weights file path>] [--image-path=<image path>]
    main.py predict (-h | --help)
Options:
    --config <model config path>  Path to YAML file for model configuration  [default: pretrained_model/YOLO/config.yaml] [type: path]
    --weights-filepath <weights file path>  Path to weights file for model  [default: pretrained_model/YOLO/Yolo_epoch=17-train_loss=21.46-val_loss=0.00.ckpt] [type: path]
    --image-path <image path> Path to image filepath for inference  [default: data/VOCdevkit/VOC2012/JPEGImages/2012_003028.jpg]
            
    -h --help  Show this.
"""
# 2007_000027.jpg
# 2012_003028.jpg
# 2012_004273.jpg
# 2012_004331.jpg
# 2012_004326.jpg
#
#    YOLO
#    --config <model config path>  Path to YAML file for model configuration  [default: pretrained_model/YOLO/config.yaml] [type: path]
#    --weights-filepath <weights file path>  Path to weights file for model  [default: pretrained_model/YOLO/Yolo_epoch=05-train_loss=7.16-val_loss=0.00.ckpt] [type: path]
#
#    XNOR-YOLO
#    --config <model config path>  Path to YAML file for model configuration  [default: pretrained_model/XNOR-YOLO/config.yaml] [type: path]
#    --weights-filepath <weights file path>  Path to weights file for model  [default: pretrained_model/XNOR-YOLO/XnorNetYolo_epoch=07-train_loss=10.80-val_loss=0.00.ckpt] [type: path]    

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
    
    predictor = Predictor(config=config)
        
    if weight_filepath:
        predictor.load_state_dict(torch.load(weight_filepath, map_location="cpu")["state_dict"])
    
    # predictor.eval()

    pil_image: Image = Image.open(image_path)
    
    with torch.no_grad():
        image: torch.Tensor = predictor.preprocess(pil_image)
    predictions: str = predictor(image)
    print(f"pred: {predictions}")

    draw = ImageDraw.Draw(pil_image)    
    for idx, pred in enumerate(predictions):        
        confidence = pred[0]
        xmin, ymin, xmax, ymax = pred[1]
        classes = predictor.model.class_map[pred[2]]        
    
        draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=(0, 0, 255), width=2)
        draw.text((xmin, ymin), classes)        

    pil_image.save("prediction.png")
