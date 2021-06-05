"""
Usage:
    main.py evaluate [options] [--config=<model config path>] [--weights-filepath=<weights file path>] [--image-path=<image path>]
    main.py evaluate (-h | --help)
Options:
    --config <model config path>  Path to YAML file for model configuration  [default: pretrained_model/XNOR-YOLO/config.yaml] [type: path]
    --weights-filepath <weights file path>  Path to weights file for model  [default: pretrained_model/XNOR-YOLO/XnorNetYolo_epoch=13-train_loss=4.54-val_loss=0.00.ckpt] [type: path]    
            
    -h --help  Show this.
"""
import sys
import subprocess
from pathlib import Path

import pytorch_lightning
import torch
from omegaconf import DictConfig
from torchvision.datasets import VOCDetection
from omegaconf import OmegaConf

from src.engine.predictor import Predictor
from src.data.transforms import Yolofy
from src.utils import get_config

pytorch_lightning.seed_everything(777)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def evaluate(hparams: dict):
    weight_filepath = str(hparams.get("--weights-filepath"))

    config_list = ["--config"]
    config: DictConfig = get_config(hparams=hparams, options=config_list)
    
    predictor = Predictor(config=config)
        
    if weight_filepath:
        predictor.load_state_dict(torch.load(weight_filepath, map_location="cpu")["state_dict"])
        
    if config.model.type == "XnorNetYolo":
        predictor.eval()

    voc2007 = VOCDetection(root="data", year="2007", image_set="val", download=False)

    root_path = Path("src/measure/input/")
    pred_path = root_path / Path("detection-results")
    gt_path = root_path / Path("ground-truth")
    for image, target in voc2007:                
        target = OmegaConf.create(target)
        filename = Path(str(target.annotation.filename)).with_suffix(".txt")
        
        gt_filepath = gt_path / filename        
        gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        with gt_filepath.open("w") as gt_file:                    
            for object_info in target.annotation.object:
                classes = object_info.name                                

                xmin = object_info.bndbox.xmin
                ymin = object_info.bndbox.ymin
                xmax = object_info.bndbox.xmax
                ymax = object_info.bndbox.ymax

                write_line = f"{classes} {xmin} {ymin} {xmax} {ymax}\n"
                gt_file.write(write_line)                

        with torch.no_grad():
            input_image: torch.Tensor = predictor.preprocess(image)
        
        predictions = predictor(input_image)

        pred_filepath = pred_path / filename
        pred_filepath.parent.mkdir(parents=True, exist_ok=True)
        with pred_filepath.open("w") as pred_file:
            for idx, pred in enumerate(predictions):        
                confidence = pred[0]
                xmin, ymin, xmax, ymax = pred[1]
                classes = predictor.model.class_map[pred[2]]

                write_line = f"{classes} {confidence} {xmin} {ymin} {xmax} {ymax}\n"        
                pred_file.write(write_line)                
        
    subprocess.check_call(["python3", "src/measure/main.py"])