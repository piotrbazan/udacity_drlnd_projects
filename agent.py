from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from strategy import BaseStrategy, GreedyStrategy
import copy


class BaseAgent(ABC):

    @abstractmethod
    def initialize(self, train_mode):
        pass

    @abstractmethod
    def get_action(self, state):
        pass

    @abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abstractmethod
    def episode_end(self):
        pass

    @abstractmethod
    def store(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass

    @abstractmethod
    def state_dict(self):
        pass


class DqnAgent(BaseAgent):

    def __init__(self,
                 model: nn.Module,
                 memory: ReplayBuffer = None,  # not required for evaluation
                 train_strategy: BaseStrategy = None,  # not required for evaluation
                 evaluate_strategy: BaseStrategy = GreedyStrategy(),
                 ddqn: bool = False,
                 gamma: float = .9,
                 batch_size: int = 32,
                 warm_up_batches: int = 5,
                 lr: float = 0.001,
                 train_every_steps: int = 1,
                 update_target_every_steps: int = 1,
                 tau: float = 1.
                 ):

        self.online_model = model
        self.memory = memory
        self.train_strategy = train_strategy
        self.evaluate_strategy = evaluate_strategy
        self.ddqn = ddqn
        self.train_mode = False
        self.gamma = gamma
        self.batch_size = batch_size
        self.warm_up_batches = warm_up_batches
        self.lr = lr
        self.train_every_steps = train_every_steps
        self.update_target_every_steps = update_target_every_steps
        self.tau = tau
        self.steps = 0
        self.train_num = 0
        self.target_model = None
        self.optimizer = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.online_model = self.online_model.to(self.device)
        self.losses = []

    def initialize(self, train_mode):
        self.train_mode = train_mode
        self.online_model.train(train_mode)
        self.steps = 0
        if train_mode:
            self.target_model = copy.deepcopy(self.online_model)
            self.optimizer = optim.Adam(self.online_model.parameters(), lr=self.lr)
            self.train_strategy.initialize()
            self.losses = []
        else:
            self.evaluate_strategy.initialize()

    def get_action(self, state):
        state = self.make_tensor(state)
        if self.train_mode:
            return self.train_strategy.select_action(self.online_model, state)
        else:
            return self.evaluate_strategy.select_action(self.online_model, state)

    def step(self, state, action, reward, next_state, done):
        if self.train_mode:
            self.memory.store(state, action, reward, next_state, done)
            if len(self.memory) >= self.batch_size * self.warm_up_batches and self.steps % self.train_every_steps == 0:
                batch = self.memory.sample(self.batch_size)
                self.train_model(batch)
            if self.steps % self.update_target_every_steps == 0:
                self.update_target_model()
        self.steps += 1

    def episode_end(self):
        self.losses = []
        if self.train_mode:
            self.train_strategy.update()
        else:
            self.evaluate_strategy.update()

    def store(self, path):
        torch.save(self.online_model.state_dict(), path / 'checkpoint.pth')

    def load(self, path):
        self.online_model.load_state_dict(torch.load(path / 'checkpoint.pth'))
        self.online_model = self.online_model.to(self.device)

    def train_model(self, batch):
        states, actions, rewards, next_states, dones = self.make_tensor(*batch)
        actions = actions.long()
        with torch.no_grad():
            if self.ddqn:  # actions from online model, values from target model
                action_ind = self.online_model(next_states).detach().max(1)[1]
                q_values = self.target_model(next_states).detach()
                q_max = q_values[np.arange(len(q_values)), action_ind].unsqueeze(1)
            else:
                q_max = self.target_model(next_states).detach().max(1)[0].unsqueeze(1)

        assert q_max.shape == rewards.shape
        q_target = rewards + self.gamma * q_max * (1 - dones)
        q_expected = self.online_model(states).gather(1, actions)
        assert q_target.shape == q_expected.shape
        loss = F.mse_loss(q_expected, q_target)
        self.losses.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        """
        Updates target model with weights from online model
        if self.tau != 1 - there is Polyak averaging
        """
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            target.data.copy_((1 - self.tau) * target.data + self.tau * online.data)

    def make_tensor(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0]).float().to(self.device)
        else:
            return [torch.from_numpy(x).float().to(self.device) for x in args]

    def state_dict(self):
        strategy_stats = self.train_strategy.state_dict() if self.train_mode else self.evaluate_strategy.state_dict()
        return {
            'avg_loss': np.mean(self.losses),
            'memory': self.memory.state_dict(),
            'strategy': strategy_stats,
        }
