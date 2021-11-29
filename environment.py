from abc import ABC, abstractmethod

from unityagents import UnityEnvironment


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


class BananaEnv(BaseEnvironment):
    """
    Banana environment
    """

    def __init__(self, file_name) -> None:
        """
        Creates banana environment
        :param file_name: path to unity binary
        """
        self.env = UnityEnvironment(file_name=file_name)
        self.brain_name = self.env.brain_names[0]
        brain = self.env.brains[self.brain_name]
        self.train_mode = False
        state = self.reset()
        self.nS = len(state)
        self.nA = brain.vector_action_space_size

    def initialize(self, train_mode):
        self.train_mode = train_mode

    def reset(self):
        """
        Seems train_mode=False causes environment to slow down by 20x!
        :return:
        """
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        # Make environment run faster
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        state = env_info.vector_observations[0]
        return state

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        return next_state, reward, done, env_info

    def close(self):
        self.env.close()
