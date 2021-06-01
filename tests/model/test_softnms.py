import sys

import pytest
import torch

from src.utils import make_logger

from src.model.utils import soft_nms

logger = make_logger(name=__name__)

test_softnms_testcase = [
    #boxes, boxscores, expected
    (torch.tensor([[200, 200, 400, 400],
                   [220, 220, 420, 420],
                   [200, 240, 400, 440],
                   [240, 200, 440, 400],
                   [1, 1, 2, 2]], dtype=torch.float),
                          
     torch.tensor([0.8, 0.7, 0.6, 0.5, 0.9], dtype=torch.float),
     torch.tensor([4, 0, 1, 2, 3], dtype=torch.int)
    )
]

@pytest.mark.parametrize("boxes, boxscores, expeceted", test_softnms_testcase)
def test_sfotnms(boxes, boxscores, expeceted):

    selected_boxes = soft_nms(boxes, boxscores)
    logger.info(selected_boxes)

    assert torch.allclose(selected_boxes, expeceted)
    