from strategy import EpsilonGreedyStrategy, LinearEpsilonGreedyStrategy, ExponentialEpsilonGreedyStrategy
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
