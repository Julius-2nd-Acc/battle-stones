from typing import Dict, Any, Optional
import uuid
import os
import random
import threading
import time
import logging

import numpy as np

from services.game_instance import GameInstance
from obj.player import PlayerType


class _ActionSpace:
    def __init__(self, n: int, rng: random.Random):
        self.n = int(n)
        self._rng = rng

    def sample(self) -> int:
        return int(self._rng.randrange(self.n))


class GameController:
    def __init__(self):
        self.games: Dict[str, GameInstance] = {}
        self.rng = random.Random()
        self._runners: Dict[str, tuple] = {}
        self.logger = logging.getLogger(__name__)

    def create_game_with_players(self, rows: int = 3, cols: int = 3, player0: str = "human", player1: str = "random") -> str:
        gid = str(uuid.uuid4())
        gi = GameInstance()
        gi.setup_game(col=cols, row=rows, p1=PlayerType[player0.upper()], p2=PlayerType[player1.upper()])

        controls = [player0, player1]
        for idx, ctl in enumerate(controls):
            try:
                player = gi.players[idx]
            except Exception:
                continue

            ctl_str = (str(ctl) if ctl is not None else "").lower()
            # If `GameInstance.setup_game` already set the player's type/ai_kind,
            # prefer those values. Only update `ai_kind` when it is not present.
            if getattr(player, 'ai_kind', None) is None:
                if ctl_str in ('', 'human'):
                    player.ai_kind = None
                elif ctl_str == 'random':
                    player.ai_kind = 'random'
                elif ctl_str in ('mc', 'mca', 'montecarlo'):
                    player.ai_kind = 'mc'
                elif ctl_str in ('q', 'qlearn', 'q-learning'):
                    player.ai_kind = 'qlearn'
                else:
                    # unknown control string -> default to random
                    player.ai_kind = 'random'

            # Ensure player has a consistent _agent_info entry for the API
            if not getattr(player, '_agent_info', None):
                ptype = (getattr(player, 'player_type', None).name.lower() if getattr(player, 'player_type', None) is not None else (player.ai_kind or 'unknown'))
                player._agent_info = {"type": ptype, "present": False, "loaded_from": None}

            # If an on-disk model file exists for MC or Q, record its path now so
            # `GET /games/{id}` can report it before the player is first used.
            try:
                if player.ai_kind == 'mc':
                    mpath = os.path.join('models', 'mc_agent_skystones.pkl')
                    if os.path.exists(mpath):
                        player._agent_info['loaded_from'] = mpath
                        player._agent_info['present'] = True
                elif player.ai_kind == 'qlearn':
                    mpath = os.path.join('models', 'q_agent_skystones.pkl')
                    if os.path.exists(mpath):
                        player._agent_info['loaded_from'] = mpath
                        player._agent_info['present'] = True
            except Exception:
                # non-fatal; leave _agent_info as-is
                pass

        self.games[gid] = gi
        return gid

    def list_games(self) -> Dict[str, Any]:
        return {gid: {"started": getattr(gi, 'started', False)} for gid, gi in self.games.items()}

    def get_state(self, game_id: str, player_idx: int = 0) -> Dict[str, Any]:
        gi = self.games[game_id]
        state = gi.get_canonical_state(player_idx)
        started = getattr(gi, 'started', False)

        winner = None
        counts = gi.board.get_current_stone_count() or {}
        if not started and counts:
            max_count = max(counts.values())
            winners = [p for p, c in counts.items() if c == max_count]
            if len(winners) == 1:
                winner = winners[0].name
            else:
                winner = "draw"

        return {
            "state": state,
            "started": started,
            "winner": winner,
            "counts": {p.name: counts.get(p, 0) for p in gi.players},
            "agents": {i: getattr(p, "_agent_info", {"type": (getattr(p, "player_type", None).name.lower() if getattr(p, "player_type", None) is not None else "unknown"), "present": False, "loaded_from": None}) for i, p in enumerate(gi.players)},
        }

    def seed(self, game_id: str, seed: Optional[int] = None) -> int:
        if seed is None:
            seed = random.randrange(2 ** 30)
        self.rng.seed(seed)
        gi = self.games.get(game_id)
        if gi is not None:
            gi._controller_rng = random.Random(seed)
        return seed

    def step(self, game_id: str, action: int, player_idx: int = 0) -> Dict[str, Any]:
        gi = self.games[game_id]
        rows = gi.board.rows
        cols = gi.board.cols
        max_slots = len(gi.initial_slots.get(player_idx, []))

        slot = action // (rows * cols)
        cell_idx = action % (rows * cols)
        r = cell_idx // cols
        c = cell_idx % cols

        slot_name = None
        if 0 <= slot < max_slots:
            slot_name = gi.initial_slots[player_idx][slot]

        chosen_stone = None
        if slot_name is not None:
            for s in gi.players[player_idx].stones:
                if s.name == slot_name:
                    chosen_stone = s
                    break

        if chosen_stone is None:
            return {"error": "invalid action or stone not available", "state": gi.get_canonical_state(player_idx)}

        if not gi.board.isValidMove((r, c)):
            return {"error": "invalid move", "state": gi.get_canonical_state(player_idx)}

        gi.place_stone(gi.players[player_idx], (r, c), chosen_stone)
        gi.check_game_over()
        return self.get_state(game_id, player_idx=player_idx)

    def delete_game(self, game_id: str) -> bool:
        self.stop_autoplay(game_id)
        if game_id in self.games:
            self.games.pop(game_id, None)
            return True
        return False

    def start_autoplay(self, game_id: str, delay: float = 1.0) -> bool:
        if game_id not in self.games:
            return False
        if game_id in self._runners:
            return False

        gi = self.games[game_id]
        stop_event = threading.Event()

        def _runner():
            gi.started = True
            try:
                while gi.started and not stop_event.is_set():
                    for player in list(gi.players):
                        if getattr(player, "player_type", None) == PlayerType.HUMAN and getattr(player, "policy", None) is None:
                            continue
                        if stop_event.is_set() or not gi.started:
                            break
                        try:
                            gi.player_turn(player)
                        except Exception:
                            pass
                        gi.check_game_over()
                        if not gi.started or stop_event.is_set():
                            break
                        time.sleep(max(0.0, float(delay)))
            finally:
                self._runners.pop(game_id, None)

        t = threading.Thread(target=_runner, name=f"autoplay-{game_id}", daemon=True)
        self._runners[game_id] = (t, stop_event)
        t.start()
        return True

    def stop_autoplay(self, game_id: str, timeout: float = 2.0) -> bool:
        entry = self._runners.get(game_id)
        if not entry:
            return False
        t, ev = entry
        ev.set()
        gi = self.games.get(game_id)
        if gi:
            gi.started = False
        t.join(timeout)
        self._runners.pop(game_id, None)
        return True

    def is_autoplaying(self, game_id: str) -> bool:
        return game_id in self._runners

    # ----------------- Policy factory ----------------- #
    def _make_policy(self, control: str, gi: GameInstance, player_idx: int):
        control = (control or "").lower()
        if control == 'human':
            return None

        rows = gi.board.rows
        cols = gi.board.cols
        max_slots = max((len(s) for s in gi.initial_slots.values()), default=0)
        action_n = max_slots * rows * cols

        per_rng = random.Random()
        try:
            per_rng.setstate(self.rng.getstate())
        except Exception:
            per_rng.seed(self.rng.randrange(2 ** 30))

        action_space = _ActionSpace(action_n, per_rng)

        def _build_obs():
            board_owner = np.zeros((rows, cols), dtype=np.int8)
            board_type = np.full((rows, cols), fill_value=-1, dtype=np.int8)

            attr_to_type = {}
            type_id = 0
            for p_idx, slot_names in gi.initial_slots.items():
                player = gi.players[p_idx]
                for name in slot_names:
                    if name is None:
                        continue
                    stone_obj = next((s for s in player.stones if s.name == name), None)
                    if stone_obj is None:
                        for rr in range(rows):
                            for cc in range(cols):
                                cell = gi.board.getField(rr, cc)
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
                    cell = gi.board.getField(rr, cc)
                    if cell is None or getattr(cell, 'player', None) is None:
                        continue
                    try:
                        attrs = tuple(cell.get_Attributes())
                    except Exception:
                        attrs = (0, 0, 0, 0)
                    tid = attr_to_type.get(attrs, -1)
                    owner = 0 if cell.player == gi.players[0] else 1
                    board_owner[rr, cc] = owner
                    board_type[rr, cc] = tid

            hand_types = np.full((2, max_slots), fill_value=-1, dtype=np.int8)
            for p_idx, player in enumerate(gi.players):
                slot_names = gi.initial_slots.get(p_idx, [])
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
            return {"board_owner": board_owner, "board_type": board_type, "hand_types": hand_types, "to_move": to_move}

        agent = None
        model_path = None
        if control == 'mc':
            try:
                from algorithms.greedy_mc import MCAgent
                model_path = os.path.join('models', 'mc_agent_skystones.pkl')
                if os.path.exists(model_path):
                    agent = MCAgent.load(model_path, action_space)
                    self.logger.info(f"Loaded MCAgent from {model_path}")
                else:
                    agent = MCAgent(action_space)
                    self.logger.info("Created new (untrained) MCAgent")
            except Exception:
                agent = None

        if control == 'qlearn':
            try:
                from algorithms.q_learning import QLearningAgent
                model_path = os.path.join('models', 'q_agent_skystones.pkl')
                if os.path.exists(model_path):
                    agent = QLearningAgent.load(model_path, action_space)
                    self.logger.info(f"Loaded QLearningAgent from {model_path}")
                else:
                    agent = QLearningAgent(action_space)
                    self.logger.info("Created new (untrained) QLearningAgent")
            except Exception:
                agent = None

        def _policy(state_json: str):
            obs = _build_obs()

            legal_actions = []
            cells = rows * cols
            slot_names = gi.initial_slots.get(player_idx, [])
            available_names = {s.name for s in gi.players[player_idx].stones}
            for slot, sname in enumerate(slot_names):
                if sname is None:
                    continue
                if sname not in available_names:
                    continue
                for rr in range(rows):
                    for cc in range(cols):
                        if gi.board.isValidMove((rr, cc)):
                            legal_actions.append(slot * cells + (rr * cols + cc))

            if agent is not None:
                if hasattr(agent, 'policy_action_masked') and legal_actions:
                    return int(agent.policy_action_masked(obs, legal_actions))
                if hasattr(agent, 'greedy_action_masked') and legal_actions:
                    return int(agent.greedy_action_masked(obs, legal_actions))
                if hasattr(agent, 'greedy_action'):
                    return int(agent.greedy_action(obs))

            if legal_actions:
                return int(self.rng.choice(legal_actions))

            try:
                return int(action_space.sample())
            except Exception:
                return 0

        # attach metadata to the returned callable so callers (and the controller)
        # can inspect whether a model was loaded and where it came from.
        agent_loaded_from = model_path if (model_path) else None
        _policy._agent_info = {"type": control, "present": bool(agent is not None), "loaded_from": agent_loaded_from}

        return _policy
