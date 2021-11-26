from model import *
import torch


def test_forward():
    model = DqnModel(10, 2, (16, 8))
    assert len(model.layers) == 5

    x = torch.randn((5, 10))
    res = model(x)
    assert res.shape == (5, 2)
