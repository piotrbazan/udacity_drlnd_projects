import numpy as np
from environment import BananaEnv


def create_env():
    return BananaEnv('Banana_Linux_NoVis/Banana.x86_64')


def test_banana_env():
    env = create_env()
    state = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (37,)

    next_state, reward, done, info = env.step(0)
    assert isinstance(state, np.ndarray)
    assert next_state.shape == (37,)

    assert isinstance(reward, float)
    assert isinstance(done, bool)

    assert env.nA == 4
    assert env.nS == 37
