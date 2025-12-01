import random
import numpy as np
from typing import Any, List
from algorithms.agent_interface import Agent

class MediumAgent(Agent):
    """
    MediumAgent: A deterministic agent that plays greedily to maximize immediate captures.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, observation: Any, legal_actions: List[int] | None = None) -> int:
        if not legal_actions:
            return self.action_space.sample()

        # Parse observation
        # ownership: (rows, cols) where 0=empty, 1=P0, 2=P1
        ownership = observation["ownership"]
        # board_stats: (rows, cols, 4) [N, S, E, W]
        board_stats = observation["board_stats"]
        # hand_stats: (2, max_slots, 4) [N, S, E, W]
        hand_stats = observation["hand_stats"]
        
        current_player_idx = int(observation["to_move"]) # 0 or 1
        my_id = current_player_idx + 1 # 1 or 2
        enemy_id = 2 if my_id == 1 else 1
        
        rows, cols = ownership.shape
        max_slots = hand_stats.shape[1]
        
        best_action = -1
        max_captures = -1
        
        # Shuffle legal actions to break ties randomly
        random.shuffle(legal_actions)
        
        for action in legal_actions:
            # Decode action
            slot = action // (rows * cols)
            cell_idx = action % (rows * cols)
            r = cell_idx // cols
            c = cell_idx % cols
            
            # Get stats of the stone we are placing
            # hand_stats[current_player_idx][slot] -> [N, S, E, W]
            my_stone_stats = hand_stats[current_player_idx][slot]
            n, s, e, w = my_stone_stats
            
            captures = 0
            
            # Check neighbors
            # North (r-1, c)
            if r > 0:
                target_owner = ownership[r-1, c]
                if target_owner == enemy_id:
                    # My N vs Enemy S
                    enemy_s = board_stats[r-1, c][1]
                    if n > enemy_s:
                        captures += 1
            
            # South (r+1, c)
            if r < rows - 1:
                target_owner = ownership[r+1, c]
                if target_owner == enemy_id:
                    # My S vs Enemy N
                    enemy_n = board_stats[r+1, c][0]
                    if s > enemy_n:
                        captures += 1
                        
            # West (r, c-1)
            if c > 0:
                target_owner = ownership[r, c-1]
                if target_owner == enemy_id:
                    # My W vs Enemy E
                    enemy_e = board_stats[r, c-1][2]
                    if w > enemy_e:
                        captures += 1
                        
            # East (r, c+1)
            if c < cols - 1:
                target_owner = ownership[r, c+1]
                if target_owner == enemy_id:
                    # My E vs Enemy W
                    enemy_w = board_stats[r, c+1][3]
                    if e > enemy_w:
                        captures += 1
            
            if captures > max_captures:
                max_captures = captures
                best_action = action
        
        return best_action
