from enum import Enum
import logging
import os
import numpy as np

from algorithms.greedy_mc import MCAgent
from algorithms.q_learning import QLearningAgent


class PlayerType(Enum):
    HUMAN = 0
    RANDOM = 1
    MC = 2
    Q = 3


class Player:
    def __init__(self, name: str, player_type: PlayerType, stones: list | None = None):
        self.name = name
        self.player_type = player_type
        # avoid mutable default argument — ensure each Player gets its own list
        self.stones = list(stones) if stones is not None else []
        # policy is a callable(state) -> action_id used at inference time
        self.policy = None
        # ai_kind kept for backward compatibility; primary control is `player_type`
        self.ai_kind: str | None = None
        # loaded agent instance (MCAgent, QLearningAgent, etc.)
        self._agent = None
        # metadata about loaded agent
        # type is derived from the enum name (e.g., 'human','random','mc','q')
        self._agent_info = {"type": self.player_type.name.lower(), "present": False, "loaded_from": None}

    def add_stone(self, stone):
        self.stones.append(stone)
        
    def remove_stone(self, stone):
        self.stones.remove(stone)

    def set_policy(self, policy_callable):
        """Set a policy for inference: callable(state)->action_id"""
        self.policy = policy_callable

    def choose_action(self, game_instance, player_idx: int):
        """Choose an action for this player.

        Parameters:
        - game_instance: the `GameInstance` calling into the player
        - player_idx: index of this player in the game

        Returns: integer action id
        """
        # If an explicit policy callable was injected, prefer it (backwards compatible)
        if getattr(self, "policy", None) is not None:
            return self.policy(game_instance.get_canonical_state(player_idx))

        # Human players should not make automated choices here
        if self.player_type == PlayerType.HUMAN:
            raise RuntimeError("Human player requires external action input")

        # For AI players, choose based on player_type
        kind_map = {
            PlayerType.MC: "mc",
            PlayerType.Q: "qlearn",
            PlayerType.RANDOM: "random",
        }
        kind = kind_map.get(self.player_type, (self.ai_kind or "random")).lower()

        # compute action space params
        rows = game_instance.board.rows
        cols = game_instance.board.cols
        max_slots = max((len(s) for s in game_instance.initial_slots.values()), default=0)
        action_n = max_slots * rows * cols

        # simple action_space implementation compatible with agents' expectations
        class _ActionSpace:
            def __init__(self, n: int):
                self.n = int(n)
            def sample(self):
                import random as _r
                return int(_r.randrange(self.n))

        action_space = _ActionSpace(action_n)
        logging.basicConfig(level=logging.INFO)
        # lazy-load agent if needed
        if kind in ("mc", "qlearn") and self._agent is None:
            model_path = None
            try:
                if kind == "mc":
                    model_path = os.path.join("models", "mc_agent_skystones.pkl")
                    logging.warning("Attempting to load MC agent from %s", model_path)
                    if os.path.exists(model_path):
                        self._agent = MCAgent.load(model_path, action_space)
                        print("MC agent loaded from", model_path)
                        self._agent_info = {"type": "mc", "present": True, "loaded_from": model_path}
                    else:
                        self._agent = MCAgent(action_space)
                        self._agent_info = {"type": "mc", "present": True, "loaded_from": None}
                else:
                    model_path = os.path.join("models", "q_agent_skystones.pkl")
                    if os.path.exists(model_path):
                        self._agent = QLearningAgent.load(model_path, action_space)
                        print("Q-learning agent loaded from", model_path)
                        self._agent_info = {"type": "qlearn", "present": True, "loaded_from": model_path}
                    else:
                        self._agent = QLearningAgent(action_space)
                        self._agent_info = {"type": "qlearn", "present": True, "loaded_from": None}
            except Exception:
                # failed to load; fall back to None (will use random)
                self._agent = None
                self._agent_info = {"type": kind, "present": False, "loaded_from": model_path}

        # build observation in the same shape agents expect
        board_owner = np.zeros((rows, cols), dtype=np.int8)
        board_type = np.full((rows, cols), fill_value=-1, dtype=np.int8)

        attr_to_type = {}
        type_id = 0
        for p_idx, slot_names in game_instance.initial_slots.items():
            player = game_instance.players[p_idx]
            for name in slot_names:
                if name is None:
                    continue
                stone_obj = next((s for s in player.stones if s.name == name), None)
                if stone_obj is None:
                    for rr in range(rows):
                        for cc in range(cols):
                            cell = game_instance.board.getField(rr, cc)
                            if cell is not None and getattr(cell, 'name', None) == name:
                                stone_obj = cell
                                break
                        if stone_obj is not None:
                            break
                if stone_obj is None:
                    continue
                try:
                    attrs = tuple(stone_obj.get_Attributes())
                except Exception:
                    attrs = (0, 0, 0, 0)
                if attrs not in attr_to_type:
                    attr_to_type[attrs] = type_id
                    type_id += 1

        for rr in range(rows):
            for cc in range(cols):
                cell = game_instance.board.getField(rr, cc)
                if cell is None or getattr(cell, 'player', None) is None:
                    continue
                try:
                    attrs = tuple(cell.get_Attributes())
                except Exception:
                    attrs = (0, 0, 0, 0)
                tid = attr_to_type.get(attrs, -1)
                owner = 0 if cell.player == game_instance.players[0] else 1
                board_owner[rr, cc] = owner
                board_type[rr, cc] = tid

        hand_types = np.full((2, max_slots), fill_value=-1, dtype=np.int8)
        for p_idx, player in enumerate(game_instance.players):
            slot_names = game_instance.initial_slots.get(p_idx, [])
            names_in_hand = {s.name: s for s in player.stones}
            for slot_idx, stone_name in enumerate(slot_names):
                if slot_idx >= max_slots:
                    break
                if stone_name is None:
                    continue
                stone_obj = names_in_hand.get(stone_name, None)
                if stone_obj is None:
                    continue
                try:
                    attrs = tuple(stone_obj.get_Attributes())
                except Exception:
                    attrs = (0, 0, 0, 0)
                hand_types[p_idx, slot_idx] = attr_to_type.get(attrs, -1)

        to_move = np.array(player_idx, dtype=np.int8)
        obs = {"board_owner": board_owner, "board_type": board_type, "hand_types": hand_types, "to_move": to_move}

        # compute legal actions
        legal_actions = []
        cells = rows * cols
        slot_names = game_instance.initial_slots.get(player_idx, [])
        available_names = {s.name for s in self.stones}
        for slot, sname in enumerate(slot_names):
            if sname is None:
                continue
            if sname not in available_names:
                continue
            for rr in range(rows):
                for cc in range(cols):
                    if game_instance.board.isValidMove((rr, cc)):
                        legal_actions.append(slot * cells + (rr * cols + cc))

        # select action via agent if available
        if self._agent is not None and legal_actions:
            agent = self._agent
            if hasattr(agent, 'policy_action_masked'):
                return int(agent.policy_action_masked(obs, legal_actions))
            if hasattr(agent, 'greedy_action_masked'):
                return int(agent.greedy_action_masked(obs, legal_actions))
            if hasattr(agent, 'greedy_action'):
                return int(agent.greedy_action(obs))

        # fallback to random legal action
        import random as _rand
        if legal_actions:
            return int(_rand.choice(legal_actions))

        # final fallback
        return int(action_space.sample())

    def set_name(self, name: str):
        """Set or update the player's display name."""
        self.name = name