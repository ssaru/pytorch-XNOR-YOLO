import pytest
import pytorch_lightning
import torch
from omegaconf import OmegaConf

from src.model.xnor_yolo import XnorNetYolo
from src.model.yolo import Yolo


@pytest.fixture(scope="module")
def tearup_xnor_yolo_model_config():
    return OmegaConf.load("conf/model/xnoryolo.yml")


@pytest.fixture(scope="module")
def tearup_yolo_model_config():
    return OmegaConf.load("conf/model/yolo.yml")


yolo_forward_test_case = [
    # (device, test_input)
    ("cpu", torch.randn(((2, 3, 448, 448)))),
    (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        torch.randn(((2, 3, 448, 448))),
    ),
]


@pytest.mark.parametrize(
    "device, test_input",
    yolo_forward_test_case,
)
def test_xnornetyolo_forward(
    fix_seed,
    tearup_xnor_yolo_model_config,
    device,
    test_input,
):

    model = XnorNetYolo(model_config=tearup_xnor_yolo_model_config.model).to(device)

    test_input = test_input.to(device)
    model(test_input)


@pytest.mark.parametrize(
    "device, test_input",
    yolo_forward_test_case,
)
def test_yolo_forward(
    fix_seed,
    tearup_yolo_model_config,
    device,
    test_input,
):

    model = Yolo(model_config=tearup_yolo_model_config.model).to(device)

    test_input = test_input.to(device)
    model(test_input)


summary_test_case = [
    # (device, test_input)
    ("cpu"),
    (torch.device("cuda" if torch.cuda.is_available() else "cpu")),
]


@pytest.mark.parametrize("device", summary_test_case)
def test_xnornetyolo_summary(fix_seed, tearup_xnor_yolo_model_config, device):
    model = XnorNetYolo(model_config=tearup_xnor_yolo_model_config.model).to(device=device)
    model.summary()


@pytest.mark.parametrize("device", summary_test_case)
def test_yolo_summary(fix_seed, tearup_yolo_model_config, device):
    model = Yolo(model_config=tearup_yolo_model_config.model).to(device=device)
    model.summary()
