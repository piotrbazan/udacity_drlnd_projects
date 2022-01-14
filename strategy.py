from abc import ABC, abstractmethod
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt


class BaseStrategy(ABC):

    def initialize(self):
        pass

    def update(self):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def display(self, episodes: int):
        pass


class DiscreteActionBaseStrategy(BaseStrategy):

    @abstractmethod
    def select_action(self, model: nn.Module, state) -> int:
        pass


class ContinuousActionBaseStrategy(BaseStrategy):

    @abstractmethod
    def select_action(self, model: nn.Module, state) -> np.array:
        pass


class GreedyStrategy(DiscreteActionBaseStrategy):

    def select_action(self, model: nn.Module, state):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy()
        return np.argmax(q_values)

    def state_dict(self, pretty=True):
        return dict(type='greedy strategy')

    def display(self, episodes: int):
        print('Nothing to display. Strategy is greedy.')


class EpsilonGreedyStrategy(DiscreteActionBaseStrategy):

    def __init__(self, eps_start):
        self.eps_start = eps_start
        self.epsilon = eps_start

    def initialize(self):
        self.epsilon = self.eps_start

    def select_action(self, model: nn.Module, state):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy()

        if np.random.rand() > self.epsilon:
            return np.argmax(q_values)
        else:
            return np.random.choice(len(q_values))

    def update(self):
        pass  # nothing to do here - epsilon is const

    def state_dict(self):
        return dict(epsilon=self.epsilon)

    def display(self, episodes: int, ax=None):
        self.initialize()
        epsilons = np.empty(episodes, dtype='float')
        for i in range(episodes):
            epsilons[i] = self.epsilon
            self.update()

        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(epsilons)
        ax.set_title(self.__class__.__name__)
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.grid(True)
        plt.show()


class LinearEpsilonGreedyStrategy(EpsilonGreedyStrategy):

    def __init__(self, eps_start=1., eps_min=.1, decay=.005):
        super().__init__(eps_start)
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.decay = decay

    def update(self):
        self.epsilon -= self.decay
        self.epsilon = max(self.epsilon, self.eps_min)


class ExponentialEpsilonGreedyStrategy(EpsilonGreedyStrategy):

    def __init__(self, eps_start=1., eps_min=.1, decay=.005):
        super().__init__(eps_start)
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.decay = decay

    def update(self):
        self.epsilon *= self.decay
        self.epsilon = max(self.epsilon, self.eps_min)


class ContinuousGreedyStrategy(ContinuousActionBaseStrategy):
    """
    Greedy strategy for continouous actions
    """

    def select_action(self, model: nn.Module, state) -> np.array:
        with torch.no_grad():
            actions = model(state).cpu().detach().data.numpy()
        return actions

    def state_dict(self):
        return dict(type='Continuous greedy strategy')

    def display(self, episodes: int):
        print('Nothing to display')


class NormalNoiseDecayStrategy(ContinuousActionBaseStrategy):

    noise: float

    def __init__(self, value_bounds, noise_start=.1, noise_min=.001, noise_decay=.999):
        self.low, self.high = value_bounds
        self.noise_start = noise_start
        self.noise_end = noise_min
        self.noise_decay = noise_decay

    def initialize(self):
        self.noise = self.noise_start

    def update(self):
        self.noise = max(self.noise_end, self.noise * self.noise_decay)

    def select_action(self, model: nn.Module, state) -> np.array:
        with torch.no_grad():
            actions = model(state).cpu().detach().data.numpy()

        ratio = self.noise * self.high
        noise = np.random.normal(loc=0, scale=ratio)
        actions += noise
        actions = np.clip(actions, self.low, self.high)
        return actions

    def state_dict(self):
        return dict(noise=self.noise)

    def display(self, episodes: int, ax=None):
        self.initialize()
        noises = np.empty(episodes, dtype='float')
        for i in range(episodes):
            noises[i] = self.noise
            self.update()

        if not ax:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.plot(noises, ls=':')
        twin_ax = ax.twinx()
        for h in self.high:
            twin_ax.plot(noises * h)
        ax.set_title(self.__class__.__name__)
        ax.set_ylabel('Noise')
        twin_ax.set_ylabel('Max noise value')
        plt.xlabel('Episodes')
        plt.grid(True)
        plt.show()
