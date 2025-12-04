import random
import numpy as np
from typing import Any, List
from algorithms.agent_interface import Agent
from algorithms.medium_agent import MediumAgent
from algorithms.defensive_agent import DefensiveAgent
from algorithms.random_agent import RandomAgent

class MixAgent(Agent):
    """
    MixAgent: A deterministic agent that plays greedily to maximize immediate captures.
    """
    def __init__(self, action_space, epsilon: float = 0.2):
        self.action_space = action_space
        self.medium_agent = MediumAgent(action_space)
        self.defensive_agent = DefensiveAgent(action_space)
        self.random_agent = RandomAgent(action_space)
        self.epsilon = epsilon
    def choose_action(self, observation: Any, legal_actions: List[int] | None = None) -> int:
        # 1. Random Exploration
        if random.random() < self.epsilon:
            return self.random_agent.choose_action(observation, legal_actions)
        
        # 2. Strategy Selection
        # "He is the first to move" -> Player 0 (index 0)
        # "He is currently winning" -> My stones > Opponent stones
        
        to_move = int(observation["to_move"])
        ownership = observation["ownership"]
        
        # Board values: 1=Player0, 2=Player1
        my_board_id = to_move + 1 
        opp_board_id = 3 - my_board_id
        
        my_score = np.count_nonzero(ownership == my_board_id)
        total_score = np.count_nonzero(ownership)
        is_winning = (my_score > total_score - my_score)

        defensive_move= 0.7 * is_winning + 0.2 * (not is_winning)
        
        if random.random() < defensive_move or total_score == 0:
            return self.defensive_agent.choose_action(observation, legal_actions)
        
        return self.medium_agent.choose_action(observation, legal_actions)