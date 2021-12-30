import numpy as np

from models import *
import torch

BATCH_SIZE = 5


def test_fc_forward():
    model = FC(10, 2, (16, 8))
    assert len(model.layers) == BATCH_SIZE

    x = torch.randn((BATCH_SIZE, 10))
    res = model(x)
    assert res.shape == (BATCH_SIZE, 2)


def test_fc_print():
    model = FC(4, 2, (128, 64))
    print(model)


def test_fcac_forward():
    model = FCAC(10, 2, (16, 8))
    assert len(model.layers) == 4

    x = torch.randn((BATCH_SIZE, 10))
    policy, value = model(x)
    assert policy.shape == (BATCH_SIZE, 2)
    assert value.shape == (BATCH_SIZE, 1)


def test_fcac_print():
    model = FCAC(4, 2, (128, 64))
    print(model)


def test_fcqv_forward():
    model = FCQV(10, 2, (16, 8))
    state = torch.randn((BATCH_SIZE, 10))
    action = torch.randn((BATCH_SIZE, 2))
    res = model(state, action)
    assert res.shape == (BATCH_SIZE, 1)


def test_fcqv_print():
    model = FCQV(4, 2, (128, 64))
    print(model)


def test_convert_to_param():
    res = convert_to_param(1)
    assert isinstance(res, nn.Parameter)
    assert res.data.dtype == torch.float


def test_rescale_forward():
    rescale = RescaleLayer(torch.tensor(-1.), torch.tensor(1.), torch.tensor(0.), torch.tensor(10.))
    x = tensor([-1, -.5, 0, .5, 1], requires_grad=True)
    res = rescale(x)
    assert np.allclose(res.detach().numpy(), [0, 2.5, 5, 7.5, 10])
    # check backprop works
    loss = (res - tensor([0, 2.5, 5, 7.5, 10]) * 2).pow(2).mul(.5).mean()
    loss.backward()
    assert np.allclose(x.grad, [0, -2.5, -5, -7.5, -10])


def test_fcdp_forward():
    action_mins = np.array([0, 0])
    action_maxs = np.array([10, 100])
    model = FCDP(4, (action_mins, action_maxs), (128, 64))
    state = torch.randn((BATCH_SIZE, 4))
    res = model(state)
    assert res.shape == (BATCH_SIZE, 2)
    assert np.logical_and(res[:, 0] >= 0, res[:, 0] <= 10).all()
    assert np.logical_and(res[:, 1] >= 0, res[:, 1] <= 100).all()


def test_fcdp_print():
    action_mins = np.array([0, 0])
    action_maxs = np.array([10, 100])
    model = FCDP(4, (action_mins, action_maxs), (128, 64))
    print(model)
