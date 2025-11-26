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
    def __init__(self, ownership, board_stats, hand_stats, to_move, stones_left):
        self.ownership = ownership.copy() # (rows, cols) 0=empty, 1=P0, 2=P1
        self.board_stats = board_stats.copy() # (rows, cols, 4)
        self.hand_stats = hand_stats.copy() # (2, max_slots, 4)
        self.to_move = to_move # 0 or 1
        self.stones_left = list(stones_left) # [p0_count, p1_count]
        
        self.rows, self.cols = self.ownership.shape
        self.max_slots = self.hand_stats.shape[1]

    @classmethod
    def from_obs(cls, obs):
        """Create a LightGameState from a Gym observation."""
        ownership = obs["ownership"]
        board_stats = obs["board_stats"]
        hand_stats = obs["hand_stats"]
        to_move = int(obs["to_move"])
        
        # Infer stones left from hand_stats
        # We assume a stone is "present" if its stats are not all 0 (or some other marker)
        # But wait, the observation doesn't explicitly mask used stones in hand_stats usually.
        # However, SkystonesEnv doesn't clear hand_stats for used stones in the observation?
        # Let's check SkystonesEnv. It uses StateBuilder.
        # StateBuilder.build_gym_observation:
        # "hand_stats": ... for s in player.stones ...
        # So hand_stats ONLY contains stones currently in hand.
        # But the array is fixed size (max_slots).
        # We need to know WHICH slots are valid.
        # The observation doesn't explicitly give us a "valid slots" mask, 
        # but we can infer it: if we try to place a stone that doesn't exist, it's illegal.
        # Actually, StateBuilder pads with 0s. A stone with 0,0,0,0 is likely invalid or weak.
        # But a real stone could be 0,0,0,0? Unlikely in this game design but possible.
        # BETTER APPROACH: The observation doesn't fully capture "which slot is valid" if 0s are valid stats.
        # However, for MCTS simulation, we can track it if we start from a valid state.
        # But from_obs is hard.
        # Workaround: We will assume the agent only calls from_obs at the root.
        # We can use `legal_actions` passed to choose_action to determine valid slots?
        # No, `from_obs` is static.
        
        # Let's assume for now that we can trust the hand_stats slots that correspond to legal actions.
        # But we need to track stones_left for the simulation.
        # We'll count non-zero entries? No, risky.
        # Let's just track "stones_left" as a count, and for the simulation, 
        # we need to know which slots are available.
        # We will add a `valid_slots` mask to LightGameState.
        
        # For the root state, we might need to rely on the fact that we only expand legal actions.
        # But for deep simulation, we need to know what's left.
        # Let's initialize `valid_slots` assuming all slots in hand_stats are valid 
        # UNLESS we can check against legal actions.
        
        # Actually, `hand_stats` in `StateBuilder` is built from `player.stones`.
        # It does NOT preserve original slot indices. It just lists current stones.
        # So slot 0 is always the first stone in hand, etc.
        # This means `action = slot * ...` refers to the index in the CURRENT hand list?
        # Let's verify `StateBuilder.get_legal_actions`.
        # It iterates `enumerate(player.stones)`.
        # So yes, slot 0 is the first available stone.
        # This simplifies things! We don't need to track original slots.
        # We just need to know how many stones are left.
        
        p0_stones = np.count_nonzero(np.sum(hand_stats[0], axis=1) >= 0) # This is always true for padded 0s?
        # Wait, StateBuilder pads with 0s?
        # "if len(names) < max_slots: names += [None] ..." in GameInstance.
        # But StateBuilder iterates `player.stones`.
        # `hand_stats` size is `(2, max_slots, 4)`.
        # It fills `len(player.stones)` and leaves the rest as 0?
        # We need to distinguish "Real Stone 0,0,0,0" from "Empty Slot".
        # Given the game design, stones usually have stats > 0.
        # Let's assume we can just use the count of stones provided in the list.
        # Actually, `hand_stats` might contain garbage or 0s for empty slots.
        
        # Let's count how many valid stones are in the hand_stats based on the game logic.
        # In `StateBuilder`, it fills the array with current stones.
        # So if I have 3 stones, indices 0, 1, 2 are valid. 3, 4 are 0s.
        # We can just track `stones_left` count for each player.
        # But we can't easily know `stones_left` just from `hand_stats` if 0s are ambiguous.
        # However, `ownership` tells us how many stones are on the board.
        # Total stones per player is fixed (4 or 5).
        # stones_left = Total - stones_on_board.
        # Let's assume 4 stones per player for now (standard).
        
        p0_on_board = np.count_nonzero(ownership == 1)
        p1_on_board = np.count_nonzero(ownership == 2)
        
        # This is a bit fragile if total stones change.
        # But for MCTS, we can just assume the `hand_stats` entries are valid up to some count.
        # Let's try to infer from the non-zero rows, or just pass it in if possible.
        # For now, let's assume valid stones are packed at the start of the array.
        
        # We'll count rows that are not all zeros? 
        # Or better: we can deduce it from the number of empty cells? No.
        
        # Let's use a heuristic: count non-zero rows.
        # If a stone is truly 0,0,0,0, this breaks. But that stone is useless anyway.
        p0_count = 0
        for i in range(hand_stats.shape[1]):
            if np.any(hand_stats[0][i] != 0):
                p0_count += 1
                
        p1_count = 0
        for i in range(hand_stats.shape[1]):
            if np.any(hand_stats[1][i] != 0):
                p1_count += 1
                
        return cls(ownership, board_stats, hand_stats, to_move, [p0_count, p1_count])

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
            
        # 2. Find valid stone slots
        # We assume stones are packed at the start of the array up to stones_left[player]
        num_stones = self.stones_left[self.to_move]
        if num_stones == 0:
            return []
            
        # 3. Combine
        cells_per_board = self.rows * self.cols
        for slot in range(num_stones):
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
        new_stones_left = list(self.stones_left)
        
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
        # Since we assume packed array, we remove this stone by shifting others down?
        # Or just swap with the last one and decrement count?
        # Swapping is faster and keeps packing.
        last_idx = new_stones_left[player_idx] - 1
        if slot != last_idx:
            new_hand_stats[player_idx][slot] = new_hand_stats[player_idx][last_idx]
        
        # Clear the last one (optional, but good for debugging)
        new_hand_stats[player_idx][last_idx] = 0
        new_stones_left[player_idx] -= 1
        
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
        
        return LightGameState(new_ownership, new_board_stats, new_hand_stats, new_to_move, new_stones_left)

    def _resolve_capture(self, r1, c1, r2, c2, stat_idx1, stat_idx2, my_id, enemy_id, ownership, board_stats):
        if ownership[r2, c2] == enemy_id:
            att = board_stats[r1, c1][stat_idx1]
            defn = board_stats[r2, c2][stat_idx2]
            if att > defn:
                ownership[r2, c2] = my_id

    def is_terminal(self):
        # Game over if no stones left for either player OR no empty spots
        if self.stones_left[0] == 0 and self.stones_left[1] == 0:
            return True
        if np.all(self.ownership != 0):
            return True
        # Also if current player has no stones?
        if self.stones_left[self.to_move] == 0:
            # If current player has no stones, but board has space and other player has stones,
            # usually the turn passes? 
            # But in this simplified model, let's assume game ends or we handle pass.
            # SkystonesEnv checks: "no_player_stones = all(len==0)"
            # If one player runs out, they skip turn?
            # GameInstance: "if len(current_player.stones) == 0: ... continue"
            # So we should probably check if BOTH run out.
            # But for MCTS, let's simplify: if I can't move, is it terminal?
            # If I have no stones, I can't move.
            # If opponent has stones, they keep playing.
            # We need to handle "Pass".
            # But `get_legal_actions` returns empty if no stones.
            # If we return empty actions, MCTS loop breaks.
            pass
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
