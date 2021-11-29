import numpy as np


class ReplayBuffer:
    """
    Stores (state, action, reward, next state, done) vectors in numpy arrays (instead of deque to save space)
    """

    def __init__(self, max_size=1000):
        self.idx = 0
        self.size = 0
        self.max_size = max_size
        self.states = np.empty(max_size, dtype=np.ndarray)
        self.actions = np.empty(max_size, dtype=np.ndarray)
        self.rewards = np.empty(max_size, dtype=np.ndarray)
        self.next_states = np.empty(max_size, dtype=np.ndarray)
        self.dones = np.empty(max_size, dtype=np.ndarray)

    def store(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.dones[self.idx] = done
        self.idx += 1
        self.idx %= self.max_size
        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        states = np.vstack(self.states[indices])
        actions = np.vstack(self.actions[indices])
        rewards = np.vstack(self.rewards[indices])
        next_states = np.vstack(self.next_states[indices])
        dones = np.vstack(self.dones[indices])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size

    def state_dict(self):
        return dict(size=self.size)
