import copy
import gc

import numpy as np
import torch
from torch import nn, optim
from torch.nn import Module

from base_agent import BaseAgent
from replay_buffer import ReplayBuffer
from strategy import ContinuousGreedyStrategy, ContinuousActionBaseStrategy


class DDPGAgent(BaseAgent):
    steps: int
    target_actor: nn.Module
    target_critic: Module
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer
    train_mode: bool
    losses: list
    policy_losses: list
    value_losses: list
    train_iter: int
    update_iter: int

    def __init__(self,
                 actor: nn.Module,
                 critic: nn.Module,
                 memory: ReplayBuffer = None,
                 train_strategy: ContinuousActionBaseStrategy = None,  # not required for evaluation
                 evaluate_strategy: ContinuousActionBaseStrategy = ContinuousGreedyStrategy(),
                 gamma: float = 1,
                 batch_size: int = 32,
                 warm_up_batches: int = 5,
                 actor_lr: float = 5e-4,
                 critic_lr: float = 5e-4,
                 actor_max_grad_norm: float = 1,
                 critic_max_grad_norm: float = 1,
                 train_every_steps: int = 1,
                 update_target_every_steps: int = 1,
                 tau: float = .001):
        super().__init__()
        self.online_actor = actor.to(self.device)
        self.online_critic = critic.to(self.device)
        self.memory = memory
        self.train_strategy = train_strategy
        self.evaluate_strategy = evaluate_strategy
        self.gamma = gamma
        self.batch_size = batch_size
        self.warm_up_batches = warm_up_batches
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_max_grad_norm = actor_max_grad_norm
        self.critic_max_grad_norm = critic_max_grad_norm
        self.train_every_steps = train_every_steps
        self.update_target_every_steps = update_target_every_steps
        self.tau = tau

    def initialize(self, train_mode):
        self.train_mode = train_mode
        self.online_actor.train(train_mode)
        self.online_critic.train(train_mode)
        self.steps = 0
        if train_mode:
            self.target_actor = copy.deepcopy(self.online_actor)
            self.target_critic = copy.deepcopy(self.online_critic)
            self.actor_optimizer = optim.Adam(self.online_actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = optim.Adam(self.online_critic.parameters(), lr=self.critic_lr)
            self.train_strategy.initialize()
            self.losses = []
            self.policy_losses = []
            self.value_losses = []
            self.train_iter = 0
            self.update_iter = 0
        else:
            self.evaluate_strategy.initialize()

    def get_action(self, state):
        state = self.make_tensor(state)
        if self.train_mode:
            return self.train_strategy.select_action(self.online_actor, state)
        else:
            return self.evaluate_strategy.select_action(self.online_actor, state)

    def step(self, state, action, reward, next_state, done):
        if self.train_mode:
            self.memory.store(state, action, reward, next_state, done)
            if len(self.memory) >= self.batch_size * self.warm_up_batches:
                if self.steps % self.train_every_steps == 0:
                    sample = self.memory.sample(self.batch_size)
                    self.train_models(sample)
                    self.train_iter += 1

                if self.steps % self.update_target_every_steps == 0:
                    self.update_target_models()
                    self.update_iter += 1

            if done:
                gc.collect()
        self.steps += 1

    def episode_end(self):
        self.losses = []; self.policy_losses = []; self.value_losses = []
        if self.train_mode:
            self.train_strategy.update()
        else:
            self.evaluate_strategy.update()

    def train_models(self, sample):
        states, actions, rewards, next_states, dones = self.make_tensor(*sample)
        next_actions = self.target_actor(next_states)
        q_max = self.target_critic(next_states, next_actions)
        assert q_max.shape == rewards.shape == dones.shape
        q_target = rewards + self.gamma * q_max * (1 - dones)
        q_expected = self.online_critic(states, actions)
        td_error = q_target.detach() - q_expected
        value_loss = td_error.pow(2).mul(.5).mean()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_critic.parameters(), self.critic_max_grad_norm)
        self.critic_optimizer.step()

        actions = self.online_actor(states)
        q_value = self.online_critic(states, actions)
        policy_loss = -q_value.mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_actor.parameters(), self.actor_max_grad_norm)
        self.actor_optimizer.step()
        self.losses.append(value_loss.item() + policy_loss.item())
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())

    def update_target_models(self):
        self.copy_parameters(self.online_actor, self.target_actor)
        self.copy_parameters(self.online_critic, self.target_critic)

    def copy_parameters(self, online_model, target_model):
        for online, target in zip(online_model.parameters(), target_model.parameters()):
            target.data.copy_((1 - self.tau) * target.data + self.tau * online.data)

    def load(self, path):
        self.online_actor.load_state_dict(torch.load(path / 'actor.pth', map_location=torch.device(self.device)))
        self.online_critic.load_state_dict(torch.load(path / 'critic.pth', map_location=torch.device(self.device)))
        self.online_actor = self.online_actor.to(self.device)
        self.online_critic = self.online_critic.to(self.device)

    def store(self, path):
        torch.save(self.online_actor.state_dict(), path / 'actor.pth')
        torch.save(self.online_critic.state_dict(), path / 'critic.pth')

    def state_dict(self):
        strategy_stats = self.train_strategy if self.train_mode else self.evaluate_strategy
        return {
            'avg_loss': np.mean(self.losses),
            'avg_policy_loss' : np.mean(self.policy_losses),
            'avg_value_loss': np.mean(self.value_losses),
            'train_iter' : self.train_iter,
            'update_iter': self.update_iter,
            'memory_size': self.memory.state_dict()['size'],
            # 'strategy': strategy_stats.state_dict(),
        }
