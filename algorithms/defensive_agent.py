import random
import numpy as np
from typing import Any, List
from algorithms.agent_interface import Agent

class DefensiveAgent(Agent):
    """
    DefensiveAgent: Chooses the most defensive move.

    A move is evaluated by:
    1. Minimizing the number of sides exposed to empty neighboring cells.
    2. Among moves with the same number of exposed sides, maximizing the
       sum of the stats on those exposed sides (N, S, E, W adjacent to empties).

    Capturing enemy stones is ignored; only future vulnerability matters.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, observation: Any, legal_actions: List[int] | None = None) -> int:
        if not legal_actions:
            return self.action_space.sample()

        ownership = observation["ownership"]          # (rows, cols) 0=empty, 1=P0, 2=P1
        board_stats = observation["board_stats"]      # (rows, cols, 4) [N, S, E, W] (unused here)
        hand_stats = observation["hand_stats"]        # (2, max_slots, 4) [N, S, E, W]

        current_player_idx = int(observation["to_move"])  # 0 or 1

        rows, cols = ownership.shape

        best_action = -1
        best_exposed_sides = None
        best_exposed_sum = None

        # Shuffle legal actions to break ties randomly
        random.shuffle(legal_actions)

        for action in legal_actions:
            # Decode action -> (slot, row, col)
            slot = action // (rows * cols)
            cell_idx = action % (rows * cols)
            r = cell_idx // cols
            c = cell_idx % cols

            # Stats of the stone to be placed: [N, S, E, W]
            my_stone_stats = hand_stats[current_player_idx][slot]
            n, s, e, w = my_stone_stats

            exposed_sides = 0
            exposed_stats_sum = 0

            # Check neighbors and only care about EMPTY neighbors (defensive exposure)

            # North (r-1, c)
            if r > 0 and ownership[r - 1, c] == 0:
                exposed_sides += 1
                exposed_stats_sum += n

            # South (r+1, c)
            if r < rows - 1 and ownership[r + 1, c] == 0:
                exposed_sides += 1
                exposed_stats_sum += s

            # West (r, c-1)
            if c > 0 and ownership[r, c - 1] == 0:
                exposed_sides += 1
                exposed_stats_sum += w

            # East (r, c+1)
            if c < cols - 1 and ownership[r, c + 1] == 0:
                exposed_sides += 1
                exposed_stats_sum += e

            # Select the move that:
            # 1. Minimizes exposed_sides
            # 2. For equal exposed_sides, maximizes exposed_stats_sum
            if best_action == -1:
                best_action = action
                best_exposed_sides = exposed_sides
                best_exposed_sum = exposed_stats_sum
            else:
                if exposed_sides < best_exposed_sides:
                    best_action = action
                    best_exposed_sides = exposed_sides
                    best_exposed_sum = exposed_stats_sum
                elif exposed_sides == best_exposed_sides and exposed_stats_sum > best_exposed_sum:
                    best_action = action
                    best_exposed_sides = exposed_sides
                    best_exposed_sum = exposed_stats_sum

        return best_action
