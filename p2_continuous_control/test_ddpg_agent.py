import numpy as np
import pytest

from ddpg_agent import DDPGAgent
from models import FCDP, FCQV
from replay_buffer import ReplayBuffer
from strategy import NormalNoiseDecayStrategy

BATCH_SIZE = 5


@pytest.fixture
def agent():
    action_mins = np.array([0, 0])
    action_maxs = np.array([10, 100])
    action_bounds = (action_mins, action_maxs)
    actor = FCDP(4, action_bounds, (10,))
    critic = FCQV(4, 2, (10,))
    memory = ReplayBuffer()
    train_strategy = NormalNoiseDecayStrategy(action_bounds)
    agent = DDPGAgent(actor, critic, memory, train_strategy, warm_up_batches=1, batch_size=5)
    yield agent


@pytest.mark.parametrize('train_mode', [False, True])
def test_get_action_train(agent, train_mode):
    agent.initialize(train_mode)
    state = np.random.randn(BATCH_SIZE, 4)
    res = agent.get_action(state)
    assert res.shape == (BATCH_SIZE, 2)
    assert np.logical_and(res[:, 0] >= 0, res[:, 0] <= 10).all()
    assert np.logical_and(res[:, 1] >= 0, res[:, 1] <= 100).all()


def test_training(agent):
    agent.initialize(True)
    state = np.random.randn(1, 4)
    for i in range(10):
        action = agent.get_action(state)
        next_state = np.random.randn(1, 4)
        reward = i * .1
        agent.step(state, action, reward, next_state, False)

    agent.episode_end()
