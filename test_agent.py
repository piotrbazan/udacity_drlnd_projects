from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from agent import DqnAgent
from model import DqnModel
from replay_buffer import ReplayBuffer
from strategy import EpsilonGreedyStrategy
from torch import nn
import pytest

BATCH_SIZE = 5


@pytest.fixture
def agent():
    model = nn.Linear(10, 2)
    memory = ReplayBuffer(10)
    agent = DqnAgent(model, memory, EpsilonGreedyStrategy(1.0))
    yield agent


def create_batch():
    states = np.random.randn(BATCH_SIZE, 10)
    actions = np.random.randint(0, 2, (BATCH_SIZE, 1))
    rewards = np.random.randint(0, 10, (BATCH_SIZE, 1))
    next_states = np.random.randn(BATCH_SIZE, 10)
    dones = np.random.randint(0, 2, (BATCH_SIZE, 1))
    batch = states, actions, rewards, next_states, dones
    return batch


def test_initialize():
    model = DqnModel(10, 2, (5,))
    agent = DqnAgent(model, None, EpsilonGreedyStrategy(1.0))
    agent.initialize(True)


def test_store_load(agent):
    with TemporaryDirectory() as dir:
        path = Path(dir)
        agent.store(path)
        agent.load(path)


def test_get_action(agent):
    for train_mode in [False, True]:
        agent.initialize(train_mode)
        for i in range(5):
            action = agent.get_action(np.random.randn(10))
            assert 0 <= action <= 1


def test_make_tensor(agent):
    arr1 = np.random.randn(3, 5)
    res = agent.make_tensor(arr1)
    assert isinstance(res, torch.Tensor)

    arr2 = np.random.randn(4, 6)
    res1, res2 = agent.make_tensor(arr1, arr2)
    assert isinstance(res1, torch.Tensor)
    assert isinstance(res2, torch.Tensor)


def test_train_model_dqn(agent):
    agent.initialize(True)
    batch = create_batch()
    agent.train_model(batch)


def test_train_model_ddqn(agent):
    agent.ddqn = True
    agent.initialize(True)
    batch = create_batch()
    agent.train_model(batch)


def test_state_dict(agent):
    res = agent.state_dict()
    print(res)
