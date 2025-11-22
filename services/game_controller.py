from typing import Dict, Any, Optional, List
import uuid
import os
import random
import threading
import time
import logging
import numpy as np

from services.game_instance import GameInstance
from services.state_builder import StateBuilder
from obj.player import PlayerType
from algorithms.agent_interface import Agent

class _ActionSpace:
    def __init__(self, n: int, rng: random.Random):
        self.n = int(n)
        self._rng = rng

    def sample(self) -> int:
        return int(self._rng.randrange(self.n))

class RandomAgent(Agent):
    def __init__(self, action_space):
        self.action_space = action_space
        
    def choose_action(self, observation: Any, legal_actions: List[int] | None = None) -> int:
        if legal_actions:
            return random.choice(legal_actions)
        return self.action_space.sample()

class GameContext:
    def __init__(self, game_instance: GameInstance, agents: List[Optional[Agent]]):
        self.game = game_instance
        self.agents = agents
        self.started = False
        self.lock = threading.Lock()

class GameController:
    def __init__(self):
        self.games: Dict[str, GameContext] = {}
        self.rng = random.Random()
        self._runners: Dict[str, tuple] = {}
        self.logger = logging.getLogger(__name__)
        
        # Global stats tracking
        self.stats = {
            "total_games": 0,
            "agent_wins": {} # e.g., "q": 10, "mc": 5, "random": 2
        }

    def create_game_with_players(self, rows: int = 3, cols: int = 3, player0: str = "human", player1: str = "random") -> str:
        gid = str(uuid.uuid4())
        gi = GameInstance()
        
        def _resolve_player_type(p_str: str) -> PlayerType:
            p_str = p_str.lower()
            if p_str == 'qlearn':
                return PlayerType.Q
            try:
                return PlayerType[p_str.upper()]
            except KeyError:
                raise ValueError(f"Unknown player type: {p_str}")

        p1_type = _resolve_player_type(player0)
        p2_type = _resolve_player_type(player1)

        gi.setup_game(col=cols, row=rows, p1=p1_type, p2=p2_type)

        agents = [None, None]
        controls = [player0, player1]
        
        for idx, ctl in enumerate(controls):
            agent, loaded_path = self._make_agent(ctl, gi, idx)
            agents[idx] = agent
            player = gi.players[idx]
            
            if agent:
                player.assign_agent(agent, loaded_from=loaded_path)

        self.games[gid] = GameContext(gi, agents)

        # Auto-start autoplay if no human players
        if all(c.lower() != 'human' for c in controls):
            self.start_autoplay(gid, delay=3.0)

        return gid

    def list_games(self) -> Dict[str, Any]:
        return {gid: {"started": ctx.started} for gid, ctx in self.games.items()}

    def get_state(self, game_id: str, player_idx: int = 0) -> Dict[str, Any]:
        ctx = self.games[game_id]
        gi = ctx.game
        
        state = StateBuilder.build_canonical_state(gi, player_idx)
        
        winner = None
        counts = gi.board.get_current_stone_count() or {}
        if not ctx.started and counts: # Game Over check approximation
             # This logic is a bit duplicated from GameInstance.check_game_over but useful for API
            max_count = max(counts.values())
            winners = [p for p, c in counts.items() if c == max_count]
            if len(winners) == 1:
                winner = winners[0].name
            else:
                winner = "draw"

        return {
            "state": state,
            "started": ctx.started,
            "winner": winner,
            "counts": {p.name: counts.get(p, 0) for p in gi.players},
            "agents": {i: getattr(p, "_agent_info", {}) for i, p in enumerate(gi.players)},
        }

    def seed(self, game_id: str, seed: Optional[int] = None) -> int:
        if seed is None:
            seed = random.randrange(2 ** 30)
        self.rng.seed(seed)
        # We might want to seed agents here too if they support it
        return seed

    def step(self, game_id: str, action: int, player_idx: int = 0) -> Dict[str, Any]:
        ctx = self.games[game_id]
        gi = ctx.game
        
        # Validate turn
        # In a real app we'd check if it's actually this player's turn, 
        # but for now we assume the API caller knows what they are doing or it's a human move.
        
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
            return {"error": "invalid action or stone not available", "state": StateBuilder.build_canonical_state(gi, player_idx)}

        if not gi.board.isValidMove((r, c)):
            return {"error": "invalid move", "state": StateBuilder.build_canonical_state(gi, player_idx)}

        gi.place_stone(gi.players[player_idx], (r, c), chosen_stone)
        gi.check_game_over()
        if not gi.started:
            ctx.started = False
            self._update_stats(gi)
            
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

        ctx = self.games[game_id]
        stop_event = threading.Event()

        def _runner():
            ctx.started = True
            ctx.game.started = True # Sync
            try:
                while ctx.started and not stop_event.is_set():
                    # Determine whose turn it is? 
                    # The GameInstance doesn't strictly track "current player" index in a variable, 
                    # it usually iterates. But here we need to know who moves next.
                    # We'll just iterate players and try to move if they have stones.
                    
                    moves_made = 0
                    for p_idx, player in enumerate(ctx.game.players):
                        if stop_event.is_set() or not ctx.started: break
                        
                        if len(player.stones) == 0: continue
                        
                        agent = ctx.agents[p_idx]
                        if agent is None: continue # Human player, skip in autoplay
                        
                        # Get Observation
                        obs = StateBuilder.build_gym_observation(ctx.game, p_idx)
                        
                        # Get Legal Actions
                        legal_actions = self._get_legal_actions(ctx.game, p_idx)
                        
                        if not legal_actions: continue
                        
                        # Ask Player (who delegates to Agent)
                        try:
                            action = player.choose_action(obs, legal_actions)
                            self.step(game_id, action, p_idx)
                            moves_made += 1
                            time.sleep(max(0.0, float(delay)))
                        except Exception as e:
                            self.logger.error(f"Agent error: {e}")
                            
                    if moves_made == 0:
                        # Game might be over or stuck
                        ctx.game.check_game_over()
                        if not ctx.game.started:
                            ctx.started = False
                            self._update_stats(ctx.game)
                            break
                        # If no one moved but game is started, maybe waiting for human?
                        # Just sleep a bit to avoid busy loop
                        time.sleep(0.1)

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
        
        ctx = self.games.get(game_id)
        if ctx:
            ctx.started = False
            ctx.game.started = False
            
        t.join(timeout)
        self._runners.pop(game_id, None)
        return True

    def is_autoplaying(self, game_id: str) -> bool:
        return game_id in self._runners

    def _get_legal_actions(self, gi: GameInstance, player_idx: int) -> List[int]:
        return StateBuilder.get_legal_actions(gi, player_idx)

    def _make_agent(self, control: str, gi: GameInstance, player_idx: int) -> tuple[Optional[Agent], Optional[str]]:
        control = (control or "").lower()
        if control == 'human':
            return None, None

        rows = gi.board.rows
        cols = gi.board.cols
        max_slots = max((len(s) for s in gi.initial_slots.values()), default=0)
        action_n = max_slots * rows * cols

        # Create a local RNG for the action space
        per_rng = random.Random()
        try:
            per_rng.setstate(self.rng.getstate())
        except Exception:
            per_rng.seed(self.rng.randrange(2 ** 30))

        action_space = _ActionSpace(action_n, per_rng)

        if control == 'mc':
            try:
                from algorithms.greedy_mc import MCAgent
                model_path = os.path.join('models', 'mc_agent_skystones.pkl')
                if os.path.exists(model_path):
                    agent = MCAgent.load(model_path, action_space)
                    self.logger.info(f"Loaded MCAgent from {model_path}")
                    return agent, model_path
                else:
                    raise FileNotFoundError(f"MC agent model not found at {model_path}")
            except ImportError as e:
                self.logger.error(f"Failed to import MC agent: {e}")
                raise ValueError(f"MC agent implementation not found: {e}")

        if control in ('qlearn', 'q'):
            try:
                from algorithms.q_learning import QLearningAgent
                model_path = os.path.join('models', 'q_agent_skystones.pkl')
                if os.path.exists(model_path):
                    agent = QLearningAgent.load(model_path, action_space)
                    self.logger.info(f"Loaded QLearningAgent from {model_path}")
                    return agent, model_path
                else:
                    raise FileNotFoundError(f"Q-learning agent model not found at {model_path}")
            except ImportError as e:
                self.logger.error(f"Failed to import Q agent: {e}")
                raise ValueError(f"Q-learning agent implementation not found: {e}")

        if control == 'random':
            return RandomAgent(action_space), None

        raise ValueError(f"Unknown agent type: {control}")

    def _update_stats(self, gi: GameInstance):
        self.stats["total_games"] += 1
        
        counts = gi.board.get_current_stone_count() or {}
        if not counts: return
        
        max_count = max(counts.values())
        winners = [p for p, c in counts.items() if c == max_count]
        
        if len(winners) == 1:
            winner = winners[0]
            agent_info = getattr(winner, "_agent_info", {})
            
            atype = "unknown"
            loaded = agent_info.get("loaded_from", "")
            if loaded:
                if "mc" in loaded.lower(): atype = "mc"
                elif "q" in loaded.lower(): atype = "q"
            else:
                 if hasattr(winner, "agent") and winner.agent:
                     atype = "random"
                 else:
                     atype = "human"
            
            self.stats["agent_wins"][atype] = self.stats["agent_wins"].get(atype, 0) + 1
        else:
            self.stats["agent_wins"]["draw"] = self.stats["agent_wins"].get("draw", 0) + 1

