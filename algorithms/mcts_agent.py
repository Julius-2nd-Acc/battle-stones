import math
import random
import time
import numpy as np
from typing import Any, List, Tuple, Optional
from algorithms.agent_interface import Agent

class LightGameState:
    """
    A lightweight, numpy-based representation of the game state for fast cloning and simulation.
    """
    def __init__(self, ownership, board_stats, hand_stats, to_move, valid_slots):
        self.ownership = ownership.copy() # (rows, cols) 0=empty, 1=P0, 2=P1
        self.board_stats = board_stats.copy() # (rows, cols, 4)
        self.hand_stats = hand_stats.copy() # (2, max_slots, 4)
        self.to_move = to_move # 0 or 1
        self.valid_slots = [list(valid_slots[0]), list(valid_slots[1])] # List of valid indices for each player
        
        self.rows, self.cols = self.ownership.shape
        self.max_slots = self.hand_stats.shape[1]

    @classmethod
    def from_obs(cls, obs):
        """Create a LightGameState from a Gym observation."""
        ownership = obs["ownership"]
        board_stats = obs["board_stats"]
        hand_stats = obs["hand_stats"]
        to_move = int(obs["to_move"])
        
        # Identify valid slots (non-zero stats)
        # Note: This assumes no valid stone has exactly (0,0,0,0) stats.
        p0_slots = []
        for i in range(hand_stats.shape[1]):
            if np.any(hand_stats[0][i] != 0):
                p0_slots.append(i)
                
        p1_slots = []
        for i in range(hand_stats.shape[1]):
            if np.any(hand_stats[1][i] != 0):
                p1_slots.append(i)
                
        return cls(ownership, board_stats, hand_stats, to_move, [p0_slots, p1_slots])

    def get_legal_actions(self):
        """
        Return list of legal actions.
        Action = slot * (rows*cols) + (row*cols + col)
        """
        actions = []
        
        # 1. Find empty cells
        empty_indices = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.ownership[r, c] == 0:
                    empty_indices.append(r * self.cols + c)
                    
        if not empty_indices:
            return []
            
        # 2. Get valid slots for current player
        current_slots = self.valid_slots[self.to_move]
        if not current_slots:
            return []
            
        # 3. Combine
        cells_per_board = self.rows * self.cols
        for slot in current_slots:
            for cell_idx in empty_indices:
                actions.append(slot * cells_per_board + cell_idx)
                
        return actions

    def step(self, action):
        """
        Apply action and return a NEW LightGameState.
        """
        # Clone state
        new_ownership = self.ownership.copy()
        new_board_stats = self.board_stats.copy()
        new_hand_stats = self.hand_stats.copy()
        
        # Deep copy valid slots
        new_valid_slots = [list(self.valid_slots[0]), list(self.valid_slots[1])]
        
        cells_per_board = self.rows * self.cols
        slot = action // cells_per_board
        cell_idx = action % cells_per_board
        r = cell_idx // self.cols
        c = cell_idx % self.cols
        
        player_idx = self.to_move
        my_id = player_idx + 1 # 1 or 2
        enemy_id = 2 if my_id == 1 else 1
        
        # 1. Place Stone
        new_ownership[r, c] = my_id
        
        # Get stone stats
        stats = new_hand_stats[player_idx][slot]
        new_board_stats[r, c] = stats
        
        # 2. Remove Stone from Hand
        # Just zero it out and remove from valid_slots
        new_hand_stats[player_idx][slot] = 0
        if slot in new_valid_slots[player_idx]:
            new_valid_slots[player_idx].remove(slot)
        
        # 3. Resolve Captures
        # Neighbors: N, S, E, W
        # Directions: (-1,0), (1,0), (0,-1), (0,1)
        # Stats indices: 0=N, 1=S, 2=E, 3=W
        
        # N (r-1, c)
        if r > 0:
            self._resolve_capture(r, c, r-1, c, 0, 1, my_id, enemy_id, new_ownership, new_board_stats)
        # S (r+1, c)
        if r < self.rows - 1:
            self._resolve_capture(r, c, r+1, c, 1, 0, my_id, enemy_id, new_ownership, new_board_stats)
        # W (r, c-1)
        if c > 0:
            self._resolve_capture(r, c, r, c-1, 3, 2, my_id, enemy_id, new_ownership, new_board_stats)
        # E (r, c+1)
        if c < self.cols - 1:
            self._resolve_capture(r, c, r, c+1, 2, 3, my_id, enemy_id, new_ownership, new_board_stats)
            
        # 4. Switch Turn
        new_to_move = 1 - self.to_move
        
        return LightGameState(new_ownership, new_board_stats, new_hand_stats, new_to_move, new_valid_slots)

    def _resolve_capture(self, r1, c1, r2, c2, stat_idx1, stat_idx2, my_id, enemy_id, ownership, board_stats):
        if ownership[r2, c2] == enemy_id:
            att = board_stats[r1, c1][stat_idx1]
            defn = board_stats[r2, c2][stat_idx2]
            if att > defn:
                ownership[r2, c2] = my_id

    def is_terminal(self):
        # Game over if no stones left for either player OR no empty spots
        p0_empty = len(self.valid_slots[0]) == 0
        p1_empty = len(self.valid_slots[1]) == 0
        
        if p0_empty and p1_empty:
            return True
            
        if np.all(self.ownership != 0):
            return True
            
        # If current player has no stones, they must pass?
        # In this simplified model, if I have no stones, I can't move.
        # If opponent has stones, they should move.
        # But step() switches turn unconditionally.
        # If I have no stones, get_legal_actions returns [].
        # MCTS loop handles this?
        # If get_legal_actions is empty but game not terminal (opponent has stones),
        # we should probably return a "Pass" action or handle it.
        # But our action space doesn't have "Pass".
        # For now, let's assume if I have no stones, it's terminal FOR ME?
        # No, MCTS needs to simulate the opponent playing out.
        # But if I can't move, I can't generate a child node.
        # So the tree stops here.
        # If the tree stops, we evaluate the state.
        # This is acceptable for MCTS: if I can't move, the game ends for this branch.
        return False
        
    def get_result(self, player_idx):
        """
        Return 1.0 if player_idx wins, 0.0 for draw, -1.0 for loss.
        """
        p0_count = np.count_nonzero(self.ownership == 1)
        p1_count = np.count_nonzero(self.ownership == 2)
        
        if p0_count > p1_count:
            winner = 0
        elif p1_count > p0_count:
            winner = 1
        else:
            return 0.0 # Draw
            
        return 1.0 if winner == player_idx else -1.0


class MCTSNode:
    def __init__(self, state: LightGameState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = state.get_legal_actions()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.414):
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

class MCTSAgent(Agent):
    def __init__(self, action_space, simulations: int = 50):
        self.action_space = action_space
        self.simulations = simulations
        
    def choose_action(self, observation: Any, legal_actions: List[int] | None = None) -> int:
        root_state = LightGameState.from_obs(observation)
        
        # If no legal actions (or empty), return random
        if not root_state.get_legal_actions():
             return self.action_space.sample()
             
        root = MCTSNode(root_state)
        
        start_time = time.time()
        # Run simulations
        for _ in range(self.simulations):
            node = self._tree_policy(root)
            reward = self._default_policy(node.state)
            self._backpropagate(node, reward)
            
        # Select best move (most visited)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def _tree_policy(self, node):
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                return self._expand(node)
            else:
                if not node.children:
                    # Should not happen if not terminal and fully expanded implies children exist
                    # Unless no legal actions (pass)
                    return node 
                node = node.best_child()
        return node

    def _expand(self, node):
        action = node.untried_actions.pop()
        next_state = node.state.step(action)
        child_node = MCTSNode(next_state, parent=node, action=action)
        node.children.append(child_node)
        return child_node

    def _default_policy(self, state):
        # Random rollout
        current_state = state
        # Limit depth to avoid infinite loops
        depth = 0
        while not current_state.is_terminal() and depth < 20:
            legal = current_state.get_legal_actions()
            if not legal:
                break
            action = random.choice(legal)
            current_state = current_state.step(action)
            depth += 1
            
        # Return reward from perspective of the player who moved to create 'state'?
        # MCTS value usually is "value for the player at node.parent.state.to_move"
        # Standard: Backpropagate result.
        # If result is +1 for P0.
        # Root is P0 to move.
        # Child is P1 to move.
        # If Child leads to P0 win, Child value should be high for P0.
        # We need to be careful with "who is maximizing".
        # Let's return result for the player at the ROOT node.
        
        # Actually, simpler: always return result for Player 0.
        # Then in backprop, if node.parent.state.to_move == 0, we want high value.
        # If node.parent.state.to_move == 1, we want low value (minimax) or invert?
        # Standard MCTS:
        # Node stores "value for the player who just moved" OR "value for the player to move at this node".
        # Let's store "Average Reward for the player who made the move to get here".
        # So `node.action` was taken by `node.parent.state.to_move`.
        # We want to maximize that player's reward.
        
        # Let's just return result for Player 0.
        # And in backprop:
        # If node.parent.state.to_move == 0: value += result
        # If node.parent.state.to_move == 1: value -= result (or += -result)
        
        # Wait, get_result(0) returns 1 if P0 wins.
        return current_state.get_result(0) # Reward for P0

    def _backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            
            # Whose turn was it at the parent?
            if node.parent:
                mover = node.parent.state.to_move
                if mover == 0:
                    node.value += result
                else:
                    node.value -= result
            else:
                # Root node. Nobody moved to get here (it's the start).
                # But we don't use root value for selection, only visits.
                pass
                
            node = node.parent

    def save(self, filepath: str):
        # MCTS has no weights, but Trainer expects a file to be created.
        with open(filepath, "wb") as f:
            f.write(b"MCTS_NO_WEIGHTS")

    @classmethod
    def load(cls, filepath: str, action_space):
        return cls(action_space)
