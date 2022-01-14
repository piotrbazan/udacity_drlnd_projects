from pathlib import Path
from typing import List

from base_agent import BaseAgent
from ddpg_agent import DDPGAgent


class MIADDPG(BaseAgent):
    """
    Multi Independent Agent DDPG: multiple DDPG agents playing in the same environment but each is independent i.e. has its own
    memory of observations and critic is trained only on those local observations.

    This is in contrast to MADDPG where agents share memory (observations, actions) to train critic.

    The agent support alternagion of DDPG agents - each initialization can switch agents based on alternate_agents flag.
    """

    def __init__(self, agents: List[DDPGAgent], alternate_agents: bool = False):
        super().__init__()
        self.agents = agents
        self.alternate_agents = alternate_agents

    def initialize(self, train_mode):
        if self.alternate_agents:
            self.agents = self.agents[::-1]
        [agent.initialize(train_mode) for agent in self.agents]

    def get_action(self, state):
        actions = [agent.get_action(state[i]) for i, agent in enumerate(self.agents)]
        return actions

    def step(self, state, action, reward, next_state, done):
        [agent.step(state[i], action[i], reward[i], next_state[i], done[i]) for i, agent in enumerate(self.agents)]

    def episode_end(self):
        [agent.episode_end() for agent in self.agents]

    def store(self, path:Path):
        for i, agent in enumerate(self.agents):
            agent_path = path / f'agent-{i}'
            agent_path.mkdir(parents=True, exist_ok=True)
            agent.store(agent_path)

    def load(self, path):
        [agent.load(path / f'agent-{i}') for i, agent in enumerate(self.agents)]

    def state_dict(self):
        result = dict()
        for i, agent in enumerate(self.agents):
            for k, v in agent.state_dict().items():
                result[f'{k}-{i}'] = v
        return result
