from abc import ABC, abstractmethod
from multiprocessing import Process
from multiprocessing.connection import Connection
from typing import List, Tuple

import gym
import numpy as np
from unityagents import UnityEnvironment
import multiprocessing as mp


class BaseEnvironment(ABC):

    @abstractmethod
    def initialize(self, train_mode):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def close(self):
        pass


class OpenAiEnv(BaseEnvironment):
    env: gym.Env

    def __init__(self, name, render=None, seed=42):
        self.name = name
        self.render = render
        self.seed = seed

    def initialize(self, train_mode):
        self.env = gym.make(self.name)
        if self.render:
            self.env.render(self.render)
        self.env.seed(self.seed)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


class MpEnv(BaseEnvironment):
    """
    In training mode is a multiprocess environment i.e. n_workers each having its own environment and seed
    In eval mode - there is only one env.

    Note: call close after finishing train_mode to close subprocesses
    """

    train_mode: bool
    env: gym.Env
    pipes: List[Tuple[Connection, Connection]]
    workers: List[Process]

    def __init__(self, env_name, n_workers: int = 1, seed: int = 42):
        self.env_name = env_name
        self.n_workers = n_workers
        self.seed = seed

    def initialize(self, train_mode):
        self.train_mode = train_mode
        if train_mode:
            self.pipes = [mp.Pipe() for _ in range(self.n_workers)]
            self.workers = [mp.Process(target=self._work, args=(rank, self.pipes[rank][1])) for rank in range(self.n_workers)]
            [w.start() for w in self.workers]
        else:
            self.env = gym.make(self.env_name)
            self.env.seed(self.seed)

    def reset(self, rank=None):
        if self.train_mode:
            if rank is not None:
                self._send_msg(rank, 'reset')
                parent_conn, child_conn = self.pipes[rank]
                return parent_conn.recv()
            else:
                self._broadcast_msg('reset')
                return np.array([parent_conn.recv() for parent_conn, child_conn in self.pipes])
        else:
            return self.env.reset()

    def receive_step_from_children(self):
        res = np.array([parent_conn.recv() for parent_conn, child_conn in self.pipes])
        assert res.shape == (self.n_workers, 4)  # state, reward, done, info
        return [np.vstack(obj) for obj in res.T]

    def step(self, action):
        if self.train_mode:
            # action is actually table of actions
            [self._send_msg(rank, 'step', a) for rank, a in enumerate(action)]
            return self.receive_step_from_children()
        else:
            return self.env.step(action)

    def close(self):
        if self.train_mode:
            self._broadcast_msg('close')
            [w.join() for w in self.workers]
            [w.close() for w in self.workers]
            [p.close() for p, _ in self.pipes]
        else:
            self.env.close()
            self.env = None

    def _work(self, rank: int, child_conn: Connection):
        env = gym.make(self.env_name)
        env.seed(self.seed + rank)
        while True:
            cmd, arg = child_conn.recv()
            if cmd == 'reset':
                child_conn.send(env.reset())
            elif cmd == 'step':
                child_conn.send(env.step(arg))
            elif cmd == 'close':
                env.close()
                child_conn.close()
                return
            else:
                print(f'Unknown cmd: {cmd}')

    def _send_msg(self, rank, cmd, arg=None):
        parent_conn, child_conn = self.pipes[rank]
        parent_conn.send((cmd, arg))

    def _broadcast_msg(self, cmd, arg=None):
        for parend_conn, child_conn in self.pipes:
            parend_conn.send((cmd, arg))


class UnityEnv(BaseEnvironment):
    """
    Adapter for Unity environment
    """

    def __init__(self, file_name) -> None:
        """
        :param file_name: path to unity binary
        """
        self.env = UnityEnvironment(file_name=file_name)
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.state_shape = env_info.vector_observations.shape
        self.num_actions = brain.vector_action_space_size
        self.num_agents = len(env_info.agents)
        self.train_mode = False

    def initialize(self, train_mode):
        self.train_mode = train_mode

    def reset(self):
        """
        Seems train_mode=False causes environment to slow down by 20x!
        :return:
        """
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        state = env_info.vector_observations
        if self.num_agents == 1:
            state = state[0] # only 1 agent - remove extra axis
        return state

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        if self.num_agents == 1:
            next_states = next_states[0]
            rewards = rewards[0]
            dones = dones[0]
        return next_states, rewards, dones, env_info

    def close(self):
        self.env.close()
