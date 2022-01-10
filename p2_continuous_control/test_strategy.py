import numpy as np

from strategy import EpsilonGreedyStrategy, LinearEpsilonGreedyStrategy, ExponentialEpsilonGreedyStrategy, NormalNoiseDecayStrategy
from torch import nn
import torch


def test_epsilon_greedy_strategy():
    strat = EpsilonGreedyStrategy(.05)
    model = nn.Linear(10, 2)
    state = torch.randn(10)
    action = strat.select_action(model, state)
    assert 0 <= action <= 1

    strat.display(1000)  # uncomment to see chart
    assert strat.state_dict() == dict(epsilon=.05)


def test_linear_epsilon_greedy_strategy():
    strat = LinearEpsilonGreedyStrategy(1, .1, .005)
    model = nn.Linear(10, 2)
    state = torch.randn(10)
    action = strat.select_action(model, state)
    assert 0 <= action <= 1

    strat.display(1000)  # uncomment to see chart
    assert strat.state_dict() == dict(epsilon=.1)


def test_exponential_epsilon_greedy_strategy():
    strat = ExponentialEpsilonGreedyStrategy(1, .1, .995)
    model = nn.Linear(10, 2)
    state = torch.randn(10)
    action = strat.select_action(model, state)
    assert 0 <= action <= 1

    strat.display(1000)  # uncomment to see chart
    assert strat.state_dict() == dict(epsilon=.1)

def test_normal_noise_decay_strategy():
    min_vals = np.array([0, 0])
    max_vals = np.array([10, 100])
    strat = NormalNoiseDecayStrategy((min_vals, max_vals))
    strat.initialize()
    model = nn.Linear(10, 2)
    state = torch.randn(10)
    actions = strat.select_action(model, state)
    print(actions)
    # assert 0 <= actions <= 1

    strat.display(1000)  # uncomment to see chart
    assert strat.state_dict() == dict(epsilon=.1)
