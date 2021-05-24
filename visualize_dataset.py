import sys

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torchvision import transforms

from src.data.dataloader import get_data_loaders
from src.data.pascal_voc import VOC2012
from src.utils import make_logger


def draw_grid(image: Image, dx: float = 1 / 7, dy: float = 1 / 7):
    current_func_name = sys._getframe().f_code.co_name
    logger = make_logger(name=current_func_name)

    draw = ImageDraw.Draw(image)
    width, height = image.size
    dx = int(dx * width)
    dy = int(dy * height)

    y_start = 0
    y_end = height
    logger.info(f"width: {width}, dx: {dx}")
    for i in range(0, width, dx):
        line = ((i, y_start), (i, y_end))
        draw.line(line, fill="red")

    x_start = 0
    x_end = width
    logger.info(f"height: {height}, dy: {dy}")
    for i in range(0, height, dy):
        line = ((x_start, i), (x_end, i))
        draw.line(line, fill="red")

    return draw


def visualize_gt(
    data_config: str = "conf/data/data.yml",
    training_config: str = "conf/training/training.yml",
):

    current_func_name = sys._getframe().f_code.co_name
    logger = make_logger(name=current_func_name)

    voc2012 = VOC2012()

    config = OmegaConf.create()
    config.update(OmegaConf.load(data_config))
    config.update(OmegaConf.load(training_config))

    train_dataloader, test_dataloader = get_data_loaders(config=config)
    train_dataset = train_dataloader.dataset
    to_pil = transforms.ToPILImage()

    # generate grid 7 x 7
    dx, dy = 1 / 7, 1 / 7
    for idx, (image, label) in enumerate(train_dataset):
        if idx > 3:
            break

        image = to_pil(image)

        draw = draw_grid(image=image)
        image_width, image_height = image.size

        indices = torch.where(label[:, :, 0] > 0)
        for y, x in zip(*indices):
            object_info = label[y, x, :]

            objectness = object_info[0]
            x_shift = object_info[1]
            y_shift = object_info[2]
            box_relwidth = object_info[3]
            box_relheight = object_info[4]

            cx = (x_shift + (dx * x)) * image_width
            cy = (y_shift + (dy * y)) * image_height
            box_width = box_relwidth * image_width
            box_height = box_relheight * image_height

            xmin = cx - (box_width / 2)
            ymin = cy - (box_height / 2)
            xmax = cx + (box_width / 2)
            ymax = cy + (box_height / 2)

            classes = object_info[10:]
            cls_name = voc2012[int(torch.where(classes > 0)[0].tolist()[0])]

            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline="blue")
            draw.ellipse(((cx - 2, cy - 2), (cx + 2, cy + 2)), fill="blue")
            draw.text((cx, cy), cls_name)

        image.save(f"{idx}.png", "JPEG")


if __name__ == "__main__":
    visualize_gt()
