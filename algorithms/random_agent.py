import random
from typing import Any, List
from algorithms.agent_interface import Agent

class RandomAgent(Agent):
    """
    Agent that selects actions uniformly at random from the set of legal actions.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, observation: Any, legal_actions: List[int] | None = None) -> int:
        if legal_actions:
            return random.choice(legal_actions)
        return self.action_space.sample()
