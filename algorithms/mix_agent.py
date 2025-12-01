import random
import numpy as np
from typing import Any, List
from algorithms.agent_interface import Agent
from algorithms.medium_agent import MediumAgent
from services.game_controller import RandomAgent

class MixAgent(Agent):
    """
    MixAgent: A deterministic agent that plays greedily to maximize immediate captures.
    """
    def __init__(self, action_space, epsilon: float = 0.2):
        self.action_space = action_space
        self.medium_agent = MediumAgent(action_space)
        self.random_agent = RandomAgent(action_space)
        self.epsilon = epsilon
    def choose_action(self, observation: Any, legal_actions: List[int] | None = None) -> int:
        random_value = random.random()
        if random_value < self.epsilon:
            return self.random_agent.choose_action(observation, legal_actions)
        
        return self.medium_agent.choose_action(observation, legal_actions)