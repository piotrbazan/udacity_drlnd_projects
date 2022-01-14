import numpy as np
import pytest

from environments import UnityEnv, OpenAiEnv, MpEnv


@pytest.fixture
def env():
    env = UnityEnv('Tennis_Linux_NoVis/Tennis.x86_64')
    try:
        yield env
    finally:
        env.close()


def test_unity_env(env):
    assert env.num_agents == 2
    assert env.num_actions == 2
    assert env.state_shape == (2, 24)

    state = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape == (2, 24)

    next_state, reward, done, info = env.step([0., 0., 0., 0.])
    assert isinstance(state, np.ndarray)
    assert next_state.shape == (2, 24)
    assert isinstance(reward, list)
    assert len(reward) == 2
    assert isinstance(done, list)
    assert len(done) == 2


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
