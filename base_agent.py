from abc import ABC

import torch


class BaseAgent(ABC):

    def __init__(self, device=None):
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def initialize(self, train_mode):
        pass

    def get_action(self, state):
        pass

    def step(self, state, action, reward, next_state, done):
        pass

    def episode_end(self):
        pass

    def store(self, filename):
        pass

    def load(self, filename):
        pass

    def state_dict(self):
        pass

    def make_tensor(self, *args, dtype=torch.float32):
        kwargs = dict(device=self.device, dtype=dtype)
        if len(args) == 1:
            return torch.from_numpy(args[0]).to(**kwargs)
        else:
            return [torch.from_numpy(x).to(**kwargs) for x in args]
