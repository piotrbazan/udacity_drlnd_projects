from abc import ABC, abstractmethod
from torch import nn
import torch
import numpy as np
import matplotlib.pyplot as plt


class BaseStrategy(ABC):

    def initialize(self):
        pass

    @abstractmethod
    def select_action(self, model: nn.Module, state):
        pass

    def update(self):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def display(self, episodes: int):
        pass


class GreedyStrategy(BaseStrategy):

    def select_action(self, model: nn.Module, state):
        with torch.no_grad():
            q_values = model(state).cpu().detach().data.numpy()
        return np.argmax(q_values)

    def state_dict(self, pretty=True):
        return dict(type='greedy strategy')

    def display(self, episodes: int):
        print('Nothing to display. Strategy is greedy.')


class EpsilonGreedyStrategy(BaseStrategy):

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
