import logging
import shutil
import sys
import urllib.request
from pathlib import Path

import pytest
import pytorch_lightning
import torch

from src.utils import make_logger


@pytest.fixture(autouse=True)
def capture_wrap():
    """
    NOTE
    If doesn't apply this function, will be raised error as following

    This Error refer to https://github.com/pytest-dev/pytest/issues/5502

    --- Logging error ---
    Traceback (most recent call last):
      File "/usr/lib/python3.6/logging/__init__.py", line 996, in emit
        stream.write(msg)
    ValueError: I/O operation on closed file.
    Call stack:
      File "/home/ubuntu/Documents/dev/Hephaestus-project/pytorch-XNOR-Net/env/lib/python3.6/site-packages/wandb/internal/internal.py", line 137, in handle_exit
        logger.info("Internal process exited")
    Message: 'Internal process exited'
    Arguments: ()
    """

    sys.stderr.close = lambda *args: None
    sys.stdout.close = lambda *args: None
    yield


@pytest.fixture(scope="function")
def fix_seed():
    current_func_name = sys._getframe().f_code.co_name
    logger = make_logger(name=current_func_name)
    logger.debug("Fix SEED")

    pytorch_lightning.seed_everything(777)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture(
    params=[
        (
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/images/01_thumb.jpg",
            "01_thumb.jpg",
            {
                "annotation": {
                    "folder": "VOC2012",
                    "filename": "2008_000008.jpg",
                    "source": {
                        "database": "The VOC2008 Database",
                        "annotation": "PASCAL VOC2008",
                        "image": "flickr",
                    },
                    "size": {"width": "500", "height": "442", "depth": "3"},
                    "segmented": "0",
                    "object": [
                        {
                            "name": "horse",
                            "pose": "Left",
                            "truncated": "0",
                            "occluded": "1",
                            "bndbox": {
                                "xmin": "53",
                                "ymin": "87",
                                "xmax": "471",
                                "ymax": "420",
                            },
                            "difficult": "0",
                        },
                        {
                            "name": "person",
                            "pose": "Unspecified",
                            "truncated": "1",
                            "occluded": "0",
                            "bndbox": {
                                "xmin": "158",
                                "ymin": "44",
                                "xmax": "289",
                                "ymax": "167",
                            },
                            "difficult": "0",
                        },
                    ],
                }
            },
        )
    ]
)
def ready_images(request):
    current_func_name = sys._getframe().f_code.co_name
    logger = make_logger(name=current_func_name)
    logger.info(f"Set up. Download images...: {request.param}")

    image_dir = Path("tests/data/images")

    url, filename, target = request.param
    filepath = image_dir / Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(url, str(filepath))

    yield (filepath, target)

    shutil.rmtree(filepath.parent)
