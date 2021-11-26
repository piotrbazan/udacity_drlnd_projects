import numpy as np
import pytest

from agent import BaseAgent, DqnAgent
from environment import BaseEnvironment, BananaEnv
from experiment import Experiment
from tempfile import TemporaryDirectory

from model import DqnModel
from replay_buffer import ReplayBuffer
from strategy import LinearEpsilonGreedyStrategy


class TestEnv(BaseEnvironment):

    def initialize(self, train_mode):
        self.state = 1

    def reset(self):
        return self.state

    def step(self, action):
        reward = 1 if action > 0 else -1
        self.state += 1
        next_state = self.state
        done = self.state > 1000
        return next_state, reward, done, None

    def close(self):
        pass


class TestAgent(BaseAgent):

    def initialize(self, train_mode):
        self.train_mode = train_mode

    def get_action(self, state, train_mode=True):
        return 1 if self.train_mode else 0

    def step(self, state, action, reward, next_state, done):
        pass

    def episode_end(self):
        pass

    def store(self, filename):
        pass

    def load(self, filename):
        pass

    def state_dict(self):
        return dict(loss=1.1)


def test_experiment():
    env = TestEnv()
    agent = TestAgent()
    exp = Experiment(env, agent)
    exp.train(200, max_t=5)
    assert len(exp.history) == 200
    assert np.allclose(exp.history['score'], 5)

    with TemporaryDirectory() as dir:
        exp.store(dir)
        exp2 = Experiment(env, agent)
        exp2.load(dir)
        assert exp.history.equals(exp2.history)

    exp2.evaluate(2, 5)
    assert len(exp2.history) == 2
    assert np.allclose(exp2.history['score'], -5)


# @pytest.mark.skip(reason='Remove annotation to test agent learning')
def test_dqn_experiment_train():
    env = BananaEnv('Banana_Linux_NoVis/Banana.x86_64')
    model = DqnModel(input_dim=env.nS, output_dim=env.nA, hidden_dims=(64, 64))
    memory = ReplayBuffer(max_size=10_000)
    train_strategy = LinearEpsilonGreedyStrategy(eps_start=1., eps_min=.1, decay=.001)
    agent = DqnAgent(model, memory, train_strategy, ddqn=False, gamma=.9, batch_size=4, train_every_steps=4, update_target_every_steps=1, tau=1.)
    exp = Experiment(env, agent, stats_every_episode=1)
    exp.train(20)

# @pytest.mark.skip(reason='Remove annotation to test agent evaluate')
def test_dqn_experiment_evaluate():
    env = BananaEnv('Banana_Linux_NoVis/Banana.x86_64')
    model = DqnModel(input_dim=env.nS, output_dim=env.nA, hidden_dims=(64, 64))
    agent = DqnAgent(model)
    exp = Experiment(env, agent, stats_every_episode=1)
    res = exp.evaluate(3, max_t=10)
    assert len(res) == 3
