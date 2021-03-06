import random
from collections import deque
from pathlib import Path
from itertools import count
import numpy as np
import pandas as pd
import torch
from tqdm import trange
from base_agent import BaseAgent
from environments import BaseEnvironment
import gc


class Experiment:
    """
    Class to conduct experiments. Use train/evaluate.
    Results can be stored/loaded with store/load methods.
    Object contains history - pandas dataframe with statistics
    """

    def __init__(self, env: BaseEnvironment, agent: BaseAgent,
                 target_points: float = 13.,
                 target_episodes: int = 100,
                 stats_every_episode: int = 5,
                 seed: int = 42):
        self.env = env
        self.agent = agent
        self.target_points = target_points
        self.target_episodes = target_episodes
        self.seed = seed
        self.history = pd.DataFrame()
        self.stats_every_episode = stats_every_episode

    def train(self, episodes, max_t=1000):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.env.initialize(train_mode=True)
        self.agent.initialize(train_mode=True)

        stats, scores = [], deque(maxlen=self.target_episodes)
        check_is_done = (lambda d: d) if self.env.num_agents == 1 else (lambda d: all(d))
        for e in range(episodes):
            state = self.env.reset()
            score = np.zeros(self.env.num_agents)
            for t in range(max_t):
                action = self.agent.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                score += reward
                state = next_state
                if check_is_done(done):
                    gc.collect()
                    break
            stats.append({'episode': e, 'steps': t, 'score': max(score), 'agent': self.agent.state_dict()})
            if e % self.stats_every_episode == 0:
                self.update_history(stats)
                self.print_stats()
            self.agent.episode_end()
            scores.append(score)
            if len(scores) == self.target_episodes and np.mean(scores) >= self.target_points:
                min_v, mean_v = np.min(scores), np.mean(scores)
                print(f'\nAgent passed grading achieving min score:{min_v:.1}, mean score: {mean_v:.1}')
                break
        self.update_history(stats)

    def evaluate(self, episodes=1, max_t=1000):
        self.env.initialize(train_mode=True) # train mode is true to avoid rendering
        self.agent.initialize(train_mode=False)
        scores = []
        moves_mse = []
        check_is_done = (lambda d: d) if self.env.num_agents == 1 else (lambda d: all(d))
        for e in trange(episodes):
            state = self.env.reset()
            score = np.zeros(self.env.num_agents)
            moves = []
            for t in range(max_t):
                action = self.agent.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                score += reward
                state = next_state
                moves.append(np.array(action).reshape(-1))
                if check_is_done(done):
                    break
            scores.append(np.mean(score))
            moves = np.array(moves)
            mse = np.power(moves[1:] - moves[:-1], 2).mean()
            moves_mse.append(mse)
        return pd.DataFrame(dict(scores=scores, moves_mse=moves_mse))

    def sample_play(self):
        self.env.initialize(train_mode=False)
        self.agent.initialize(train_mode=False)
        state = self.env.reset()
        check_is_done = (lambda d: d) if self.env.num_agents == 1 else (lambda d: all(d))
        while True:
            action = self.agent.get_action(state)
            next_state, reward, done, info = self.env.step(action)
            self.agent.step(state, action, reward, next_state, done)
            state = next_state
            if check_is_done(done):
                break

    def store(self, path):
        path = Path(path)
        model_path = path / 'model'
        history_path = path / 'history.parquet'
        model_path.mkdir(parents=True, exist_ok=True)
        self.agent.store(model_path)
        self.history.to_parquet(history_path)

    def load(self, path):
        model_path = Path(path) / 'model'
        history_path = Path(path) / 'history.parquet'
        self.agent.load(model_path)
        self.history = pd.read_parquet(history_path)

    def update_history(self, stats):
        df = pd.DataFrame(stats)
        self.history = pd.concat([self.history, df], ignore_index=True)
        del stats[:]
        gc.collect()

    def print_stats(self):
        if self.history.empty:
            print('Empty history')
        else:
            last = self.history.iloc[-1]
            fmt = '\rEpisode: {}, steps: {}, score: {:.2f}, avg_score: {:.2f}'
            args = [last['episode'], last['steps'], last['score'], self.history['score'][-100:].mean()]
            agent_stats = last['agent']
            if agent_stats:
                fmt += ' agent ???'
                for k, v in agent_stats.items():
                    if isinstance(v, dict):
                        fmt += ' ' + k + ': {}'
                    elif isinstance(v, int):
                        fmt += ' ' + k + ': {}'
                    else:
                        fmt += ' ' + k + ': {:.1f}'
                    args.append(v)
            print(fmt.format(*args), end='')
