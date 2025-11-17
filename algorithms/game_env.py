import json
import random
from typing import Any, Tuple

from services.game_instance import GameInstance


class GameEnv:
    """A small Gym-like wrapper around GameInstance for training.

    Action encoding (fixed-size):
        action_id = stone_slot * (rows*cols) + (row*cols + col)

    stone_slot refers to the initial ordering of stones given to each player on setup.
    If a stone in that slot has already been played, the action is invalid.
    """

    def __init__(self, game_factory=GameInstance, rows=3, cols=3, max_stones=4, train_player_idx=0):
        self.game_factory = game_factory
        self.rows = rows
        self.cols = cols
        self.max_stones = max_stones
        self.n_actions = max_stones * rows * cols
        self.train_player_idx = train_player_idx

        self.game: GameInstance | None = None
        # mapping player -> list of initial stone names (slot mapping)
        self.initial_slots = {}

    def reset(self) -> Any:
        self.game = self.game_factory()
        # existing API: setup_game() creates board and players and generates stones
        self.game.setup_game(col=self.cols, row=self.rows)

        # remember initial stone slots by name for each player
        self.initial_slots = {}
        for i, p in enumerate(self.game.players):
            names = [s.name for s in p.stones]
            # pad to max_stones
            names += [None] * max(0, self.max_stones - len(names))
            self.initial_slots[i] = names[: self.max_stones]

        return self._get_state()

    def step(self, action_id: int) -> Tuple[Any, float, bool, dict]:
        # decode action
        stone_slot = action_id // (self.rows * self.cols)
        cell_idx = action_id % (self.rows * self.cols)
        r = cell_idx // self.cols
        c = cell_idx % self.cols

        train_player = self.game.players[self.train_player_idx]

        info = {"invalid": False}

        # map slot -> stone object (by name)
        slot_name = self.initial_slots[self.train_player_idx][stone_slot] if stone_slot < len(self.initial_slots[self.train_player_idx]) else None
        if slot_name is None:
            # invalid slot
            info["invalid"] = True
            return self._get_state(), -0.1, False, info

        # find the stone object with that name in current player's stones
        chosen_stone = None
        for s in train_player.stones:
            if s.name == slot_name:
                chosen_stone = s
                break

        if chosen_stone is None:
            # stone already played -> invalid
            info["invalid"] = True
            return self._get_state(), -0.1, False, info

        if not self.game.board.isValidMove((r, c)):
            info["invalid"] = True
            return self._get_state(), -0.1, False, info
        previous_count = self.game.board.get_current_stone_count()
        
        # place the stone using existing API
        self.game.place_stone(train_player, (r, c), chosen_stone)

        # simple opponent: perform one random legal move for the other player (if not done)
        done = all(len(p.stones) == 0 for p in self.game.players) or not any(
            self.game.board.isValidMove((rr, cc)) for rr in range(self.rows) for cc in range(self.cols)
        )

        if not done:
            # opponent index (naive 2-player assumption)
            opp_idx = 1 - self.train_player_idx
            opp = self.game.players[opp_idx]
            # pick a random valid position and random stone from opp
            valid_positions = [(rr, cc) for rr in range(self.rows) for cc in range(self.cols) if self.game.board.isValidMove((rr, cc))]
            if valid_positions and len(opp.stones) > 0:
                pos = random.choice(valid_positions)
                stone = random.choice(opp.stones)
                try:
                    self.game.place_stone(opp, pos, stone)
                except Exception:
                    # if something goes wrong, ignore and continue
                    pass

        done = all(len(p.stones) == 0 for p in self.game.players) or not any(
            self.game.board.isValidMove((rr, cc)) for rr in range(self.rows) for cc in range(self.cols)
        )

        reward = 0.0
        if done:
            counts = self.game.board.get_current_stone_count()
            train_count = counts.get(train_player, 0)
            other_count = sum(c for p, c in counts.items() if p != train_player)
            if train_count > other_count:
                reward = 1.0
            elif train_count < other_count:
                reward = -1.0
            else:
                reward = 0.0
        
        counts = self.game.board.get_current_stone_count()
        train_count = counts.get(train_player, 0)
        other_count = sum(c for p, c in counts.items() if p != train_player)
        if train_count > other_count:
                reward = 1.0
        elif train_count < other_count:
                reward = -1.0
        else:
                reward = 0.0
        
        reward += self.evaluate_reward(previous_count,self.game.board.get_current_stone_count())

        return self._get_state(), reward, done, info
    
    def evaluate_reward(self,previous_count,current_count) -> float:
            # simple reward: +1 for each additional stone owned, -1 for each lost stone
            reward = 0.0
            train_player = self.game.players[self.train_player_idx]
            prev = previous_count.get(train_player, 0)
            curr = current_count.get(train_player, 0)
            reward += (curr - prev) * 1.0
            return reward

    def _get_state(self) -> Any:
        # simple JSON string state (deterministic ordering)
        board_repr = []
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.game.board.getField(r, c)
                if cell is None or getattr(cell, "player", None) is None:
                    board_repr.append(".")
                else:
                    owner_idx = 0 if self.game.players[0] == getattr(cell, "player") else 1
                    # represent stone by owner and its name
                    board_repr.append(f"{owner_idx}:{cell.name}")

        players_stones = []
        for p in self.game.players:
            players_stones.append([s.name for s in p.stones])

        payload = {
            "board": board_repr,
            "players": players_stones,
            "to_move": self.train_player_idx,
        }

        return json.dumps(payload, sort_keys=True)
# algorithms/game_env.py
from typing import Tuple, Any, List
from services.game_instance import GameInstance
from obj.stone import Stone
import json

class GameEnv:
    def __init__(self, game_factory=GameInstance, rows=3, cols=3, max_stones=4, train_player_idx=0):
        self.game_factory = game_factory
        self.rows = rows
        self.cols = cols
        self.max_stones = max_stones
        self.n_actions = max_stones * rows * cols
        self.train_player_idx = train_player_idx
        self.game = None

    def reset(self) -> Any:
        self.game = self.game_factory()
        self.game.setup_game(col=self.cols, row=self.rows)
        # Optionally load stones from DTO instead of random
        return self._get_state()

    def step(self, action_id: int) -> Tuple[Any, float, bool, dict]:
        # Decode action
        stone_idx = action_id // (self.rows * self.cols)
        cell_idx = action_id % (self.rows * self.cols)
        r = cell_idx // self.cols
        c = cell_idx % self.cols

        player = self.game.players[self.game.players.index(next(p for p in self.game.players if p.name == self.game.players[self.train_player_idx].name))]
        # Validate stone availability
        if stone_idx < 0 or stone_idx >= len(player.stones):
            # invalid: penalty
            return self._get_state(), -0.1, False, {"invalid": True}
        stone = player.stones[stone_idx]
        if not self.game.board.isValidMove((r, c)):
            return self._get_state(), -0.1, False, {"invalid": True}

        # Place stone (this already sets owner via set_Owner in your code)
        self.game.place_stone(player, (r, c), stone)

        # After placement, advance turn (your start_game loop handles order; here we can flip player index)
        # Here we assume env controls turns for training; implement a minimal turn rotation:
        # (for simple training, you may let the env choose actions for both players using agent or random.)
        done = all(len(p.stones) == 0 for p in self.game.players) or not any(self.game.board.isValidMove((rr, cc)) for rr in range(self.rows) for cc in range(self.cols))

        reward = 0.0
        if done:
            counts = self.game.board.get_current_stone_count()
            # Determine winner counts relative to the training player's identity
            train_player_obj = self.game.players[self.train_player_idx]
            train_count = counts.get(train_player_obj, 0)
            other_counts = sum(c for p, c in counts.items() if p != train_player_obj)
            if train_count > other_counts:
                reward = 1.0
            elif train_count < other_counts:
                reward = -1.0
            else:
                reward = 0.0

        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        # Create a canonical hashable state (simple JSON string)
        board_repr = []
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.game.board.getField(r, c)
                if cell is None or getattr(cell, "player", None) is None:
                    board_repr.append(".")
                else:
                    owner_idx = 0 if self.game.players[0] == getattr(cell, "player") else 1
                    board_repr.append(f"{owner_idx}:{cell.name}")
        payload = {"board": board_repr, "players": [[(s.n,s.s,s.e,s.w) for s in p.stones] for p in self.game.players], "to_move": 0}
        return json.dumps(payload, sort_keys=True)