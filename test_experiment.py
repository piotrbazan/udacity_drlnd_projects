from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from base_agent import BaseAgent
from ddpg_agent import DDPGAgent
from environments import BaseEnvironment, OpenAiEnv, UnityEnv
from experiment import Experiment
from maddpg_agent import MIADDPG
from models import FC, FCDP, FCQV
from replay_buffer import ReplayBuffer
from strategy import NormalNoiseDecayStrategy
import pytest


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

    def state_dict(self):
        return dict(avg_loss=11.1)


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


@pytest.mark.skip(reason='unmark to run')
def test_reinforce_experiment():
    env = OpenAiEnv('CartPole-v1')
    model = FC(4, 2, (128, 64))
    agent = ReinforceAgent(model, gamma=1., lr=5e-4)
    exp = Experiment(env, agent, target_points=150, target_episodes=100)
    exp.train(100, max_t=500)


# @pytest.mark.skip(reason='unmark to run')
def test_miaddpg_experiment():
    env = UnityEnv('Tennis_Linux_NoVis/Tennis.x86_64')
    env.initialize(True)

    num_states = env.state_shape[1]
    num_agents = env.num_agents
    num_actions = env.num_actions
    action_bounds = (np.array([-1, -1]), np.array([1, 1]))

    agents = [create_miaddpg_agent(action_bounds, num_actions, num_states) for i in range(num_agents)]
    agent = MIADDPG(agents)

    exp = Experiment(env, agent, target_points=.5)
    exp.train(500)

@pytest.mark.skip(reason='unmark to run')
def test_evalutate_miaddpg():
    env = UnityEnv('Tennis_Linux_NoVis/Tennis.x86_64')
    env.initialize(True)

    num_states = env.state_shape[1]
    num_agents = env.num_agents
    num_actions = env.num_actions
    action_bounds = (np.array([-1, -1]), np.array([1, 1]))

    agents = [create_miaddpg_agent(action_bounds, num_actions, num_states) for i in range(num_agents)]
    agent = MIADDPG(agents)

    exp = Experiment(env, agent, target_points=.5)
    exp.load('play1')
    df = exp.evaluate(10)
    print(df)


def create_miaddpg_agent(action_bounds, num_actions, num_states):
    actor = FCDP(num_states, action_bounds, (128, 128))
    critic = FCQV(num_states, num_actions, (128, 128))
    memory = ReplayBuffer(50000)
    train_strategy = NormalNoiseDecayStrategy(action_bounds, noise_decay=1.)
    agent = DDPGAgent(actor, critic, memory, train_strategy,
                      gamma=.99,
                      batch_size=128,
                      actor_lr=3e-4,
                      critic_lr=3e-4,
                      actor_max_grad_norm=float('inf'),
                      critic_max_grad_norm=float('inf'),
                      train_every_steps=1,
                      update_target_every_steps=1,
                      tau=.005)
    return agent


def test_print_stats_empty():
    exp = Experiment(None, None)
    exp.print_stats()


def test_print_stats():
    exp = Experiment(None, None)
    exp.history = pd.DataFrame({
        'episode': [1, 2],
        'score': [1., 2.],
        'agent': [{'avg_loss': 123, 'avg_log_probs': .2}, {'avg_loss': 123, 'avg_log_probs': .2}]
    })
    exp.print_stats()
