import gymnasium as gym
from gymnasium import spaces
import numpy as np

from obj.board import Field
from services.game_instance import GameInstance
from services.state_builder import StateBuilder


class SkystonesEnv(gym.Env):
    """
    Gymnasium wrapper around your Skystones-like GameInstance.

    - Two players (index 0 and 1)
    - Single policy controls both players (self-play).
    - Action = slot * (rows * cols) + cell_index.
    - Reward = final outcome (win/loss/draw) + per-capture shaping.
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, capture_reward: float = 1.0, rows: int = 3, cols: int = 3):
        super().__init__()
        self.render_mode = render_mode
        self.capture_reward = capture_reward
        self.rows = rows
        self.cols = cols

        # Use a temporary game to infer board and stone info
        tmp_game = GameInstance()
        tmp_game.setup_game(col=cols, row=rows)
        # self.rows/cols already set above
        self.max_slots = max(len(slots) for slots in tmp_game.initial_slots.values())

        # Build a mapping from stone (n,s,e,w) -> type_id
        # Note: We now use direct stats in the observation, so this registry 
        # is no longer needed for the observation space, but we keep the 
        # max_slots calculation above.

        # Underlying game (created on reset)
        self.game = None
        self.current_player_idx = 0

        # Action: choose slot + cell
        self.action_space = spaces.Discrete(self.max_slots * self.rows * self.cols)

        # Observation:
        # - ownership: 0 empty, 1 player0, 2 player1
        # - board_stats: (rows, cols, 4) -> (n, s, e, w) values
        # - hand_stats: (2, max_slots, 4) -> (n, s, e, w) values
        # - to_move: which player (0 or 1)
        self.observation_space = spaces.Dict(
            {
                "ownership": spaces.Box(
                    low=0,
                    high=2,
                    shape=(self.rows, self.cols),
                    dtype=np.int8,
                ),
                "board_stats": spaces.Box(
                    low=0,
                    high=20, # Assuming stats don't exceed 20
                    shape=(self.rows, self.cols, 4),
                    dtype=np.int8,
                ),
                "hand_stats": spaces.Box(
                    low=0,
                    high=20,
                    shape=(2, self.max_slots, 4),
                    dtype=np.int8,
                ),
                "to_move": spaces.Discrete(2),
            }
        )

        self.num_types = int(self.observation_space["board_stats"].high.max())

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.game = GameInstance()
        self.game.setup_game(col=self.cols, row=self.rows)
        self.current_player_idx = 0

        obs = self._build_observation()
        info = {}
        return obs, info

    def step(self, action: int):
        """
        One environment step = one move by current_player_idx.

        Reward components (from current player's perspective):
          - capture/loss shaping each move
          - final win/loss/draw reward when game ends

        Invalid action → immediate loss for current player.
        """
        assert self.game is not None, "Call reset() before step()."

        acting_player_idx = self.current_player_idx
        player = self.game.players[acting_player_idx]

        # Decode action
        slot, row, col = self._decode_action(action)

        # Map slot → actual stone object
        chosen_stone = self._get_stone_for_slot(
            player_idx=acting_player_idx, slot=slot
        )

        # Check legality
        legal = (
            chosen_stone is not None
            and self.game.board.isValidMove((row, col))
        )

        if not legal:
            # Illegal move → current player loses
            reward = self._terminal_reward(acting_player_idx=self.current_player_idx)
            terminated = True
            truncated = False
            obs = self._build_observation()
            info = {"illegal_move": True}
            return obs, reward, terminated, truncated, info

        # --------- measure current player's stones before the move ----------
        owner_counts_before = self.game.board.get_current_stone_count()
        current_player_before = owner_counts_before.get(player, 0)

        # Apply the move (this may cause captures)
        self.game.place_stone(player, (row, col), chosen_stone)

        # --------- measure current player's stones after the move -----------
        owner_counts_after = self.game.board.get_current_stone_count()
        current_player_after = owner_counts_after.get(player, 0)

        # Calculate captures from current player's perspective
        # delta = (stones after) - (stones before)
        # This includes +1 for placing the stone, so captures = delta - 1
        delta = current_player_after - current_player_before
        net_captures = delta - 1  # Remove the +1 from placing our own stone

        capture_reward = self.capture_reward * net_captures

        terminated = self._is_terminal()
        truncated = False

        reward_from_p0 = capture_reward
        if terminated:
            reward_from_p0 += self._final_outcome_reward(acting_player_idx=self.current_player_idx)
        else:
            # Switch side to move for next step
            self.current_player_idx = 1 - acting_player_idx

        # Finally: convert from P0 perspective to acting player's perspective
        reward = reward_from_p0 if acting_player_idx == 0 else -reward_from_p0

        obs = self._build_observation()
        info = {"net_captures": net_captures}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info


    def render(self):
        if self.render_mode == "human" and self.game is not None:
            print(f"Player to move: {self.current_player_idx}")
            self.game.board.draw_board()

    def close(self):
        self.game = None

    # ------------------------------------------------------------------ #
    # Helper functions
    # ------------------------------------------------------------------ #

    def _decode_action(self, action: int):
        cells = self.rows * self.cols
        slot = action // cells
        cell_idx = action % cells
        row = cell_idx // self.cols
        col = cell_idx % self.cols
        return slot, row, col

    def _get_stone_for_slot(self, player_idx: int, slot: int):
        """
        Translate a slot index into the actual Stone object, if that stone is
        still in the player's hand, using initial_slots name mapping.
        """
        slot_names = self.game.initial_slots.get(player_idx, [])
        if not (0 <= slot < len(slot_names)):
            return None

        stone_name = slot_names[slot]
        if stone_name is None:
            return None

        player = self.game.players[player_idx]
        for s in player.stones:
            if s.name == stone_name:
                return s
        return None
    
    def get_legal_actions(self, player_idx: int | None = None):
        """
        Return a list of legal action indices for the given player
        (or for current_player_idx if None).
        """
        if self.game is None:
            return []

        if player_idx is None:
            player_idx = self.current_player_idx
            
        return StateBuilder.get_legal_actions(self.game, player_idx)

    def _build_observation(self):
        return StateBuilder.build_gym_observation(self.game, self.current_player_idx)

    def _is_terminal(self) -> bool:
        no_player_stones = all(len(p.stones) == 0 for p in self.game.players)
        no_empty_fields = not any(
            self.game.board.isValidMove((r, c))
            for r in range(self.rows)
            for c in range(self.cols)
        )
        return no_player_stones or no_empty_fields 
    

    def _final_outcome_reward(self, acting_player_idx: int) -> float:
        """
        Final reward from the acting player's perspective:
          +1 if acting player wins, -1 if acting player loses, 0 for draw.
        """
        owner_counts = self.game.board.get_current_stone_count()

        if not owner_counts:
            return 0.0

        max_count = max(owner_counts.values())
        winners = [p for p, c in owner_counts.items() if c == max_count]

        if len(winners) != 1:
            return -3.0

        winner = winners[0]
        acting_player = self.game.players[acting_player_idx]
        
        if winner == acting_player:
            return 3.0
        else:
            return -3.0

    def _terminal_reward(self, acting_player_idx: int) -> float:
        """
        Reward when the acting player plays an illegal move.
        Always -1.0 because the acting player loses.
        """
        return -1.0
