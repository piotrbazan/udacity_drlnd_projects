import numpy as np
import pytest

from environments import UnityEnv, OpenAiEnv, MpEnv


@pytest.fixture
def env():
    env = UnityEnv('Reacher_Linux_NoVis_1/Reacher.x86_64')
    try:
        yield env
    finally:
        env.close()


def test_banana_env(env):
    state = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (33,)

    next_state, reward, done, info = env.step([0., 0., 0., 0.])
    assert isinstance(state, np.ndarray)
    assert next_state.shape == (33,)

    assert isinstance(reward, float)
    assert isinstance(done, bool)

    assert env.nA == 4
    assert env.nS == 33


def test_cart_pole():
    env = OpenAiEnv('CartPole-v1')
    env.initialize(True)
    state = env.reset()
    assert state.shape == (4,)
    state, rewards, done, info = env.step(0)
    assert state.shape == (4,)
    assert isinstance(rewards, float)
    assert isinstance(done, bool)


def test_mp_env_eval_mode():
    env = MpEnv('CartPole-v1')
    env.initialize(False)
    state = env.reset()
    assert state.shape == (4,)
    state, rewards, done, info = env.step(0)
    assert state.shape == (4,)
    assert isinstance(rewards, float)
    assert isinstance(done, bool)


def test_mp_env_train_mode():
    env = MpEnv('CartPole-v1', n_workers=2)
    try:
        env.initialize(True)
        state = env.reset(rank=0)
        assert state.shape == (4,)
        states = env.reset()
        assert states.shape == (2, 4)
        states, rewards, dones, infos = env.step([0, 0])
        assert states.shape == (2, 4)
        assert rewards.shape == (2, 1)
        assert dones.shape == (2, 1)
    finally:
        env.close()
