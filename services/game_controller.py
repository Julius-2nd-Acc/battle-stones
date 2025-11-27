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

def _convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {_convert_numpy_types(k): _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_convert_numpy_types(item) for item in obj)
    return obj

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

    def create_game_with_players(self, rows: int = 3, cols: int = 3, player0: str = "human", player1: str = "random", autoplay: bool = True) -> str:
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

        # Auto-start autoplay if requested (default True)
        if autoplay:
            self.start_autoplay(gid, delay=3.0)

        return gid

    def list_games(self) -> Dict[str, Any]:
        return {gid: {"started": ctx.started} for gid, ctx in self.games.items()}

    def get_state(self, game_id: str, player_idx: int = 0) -> Dict[str, Any]:
        ctx = self.games[game_id]
        gi = ctx.game
        
        state = StateBuilder.build_canonical_state(gi, player_idx)
        print(state)
        
        winner = None
        counts = gi.board.get_current_stone_count() or {}
        # Convert counts dict to use player names as keys instead of Player objects
        counts_by_name = {p.name: int(count) for p, count in counts.items()}
        
        if not ctx.started and counts: # Game Over check approximation
             # This logic is a bit duplicated from GameInstance.check_game_over but useful for API
            max_count = max(counts.values())
            winners = [p for p, c in counts.items() if c == max_count]
            if len(winners) == 1:
                winner = winners[0].name
            else:
                winner = "draw"

        # Determine current player (alternates, starts with player 0)
        current_player_idx = None
        awaiting_human = False
        if gi.started:
            # Simple heuristic: count stones played to determine whose turn
            total_played = sum(len(gi.initial_slots.get(i, [])) - len(p.stones) for i, p in enumerate(gi.players))
            current_player_idx = total_played % 2
            awaiting_human = gi.players[current_player_idx].agent is None
        
        # Build structured board with full stone details
        structured_board = []
        for r in range(gi.board.rows):
            row = []
            for c in range(gi.board.cols):
                cell = gi.board.getField(r, c)
                # Check if cell has a stone (not None and not empty string ".")
                if cell is None or cell == "." or not hasattr(cell, 'n'):
                    row.append(None)
                else:
                    # cell is a Stone object
                    owner = None
                    if hasattr(cell, 'get_Owner'):
                        try:
                            owner_obj = cell.get_Owner()
                            if owner_obj:
                                owner = owner_obj.name
                        except:
                            pass
                    
                    row.append({
                        "name": cell.name,
                        "n": int(cell.n),
                        "s": int(cell.s),
                        "e": int(cell.e),
                        "w": int(cell.w),
                        "owner": cell.owner.name if hasattr(cell.owner, "name") else None
                    })
            structured_board.append(row)
        
        result = {
            "state": state,
            "started": ctx.started,
            "winner": winner,
            "counts": counts_by_name,
            "agents": {str(i): getattr(p, "_agent_info", {}) for i, p in enumerate(gi.players)},
            "current_player": int(current_player_idx) if current_player_idx is not None else None,
            "awaiting_human": awaiting_human,
            "available_stones": [
                {
                    "name": s.name,
                    "n": int(s.n),
                    "s": int(s.s),
                    "e": int(s.e),
                    "w": int(s.w),
                    "owner": s.owner.name if hasattr(s.owner, "name") else None
                } for s in gi.players[player_idx].stones
            ] if gi.started else [],
            "board_size": {"rows": int(gi.board.rows), "cols": int(gi.board.cols)},
            "board": structured_board,
        }
        
        # Convert any remaining numpy types to native Python types
        return _convert_numpy_types(result)

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

    def step_human(self, game_id: str, stone_index: int, row: int, col: int, player_idx: int = 0) -> Dict[str, Any]:
        """
        Human-friendly step method that accepts stone index and board position.
        Validates input and translates to action before calling step().
        """
        ctx = self.games[game_id]
        gi = ctx.game
        
        # Validate player has stones
        player = gi.players[player_idx]
        if not player.stones:
            raise ValueError(f"Player {player_idx} has no stones left")
        
        # Validate stone_index
        if stone_index < 0 or stone_index >= len(player.stones):
            raise ValueError(f"Invalid stone_index: must be between 0 and {len(player.stones) - 1}")
        
        # Get the stone at this index
        chosen_stone = player.stones[stone_index]
        
        # Validate board position
        rows = gi.board.rows
        cols = gi.board.cols
        if row < 0 or row >= rows:
            raise ValueError(f"Invalid row: must be between 0 and {rows - 1}")
        if col < 0 or col >= cols:
            raise ValueError(f"Invalid col: must be between 0 and {cols - 1}")
        
        # Validate cell is empty
        if not gi.board.isValidMove((row, col)):
            raise ValueError(f"Cell ({row}, {col}) is already occupied or invalid")
        
        # Find slot index for this stone
        slot = None
        for slot_idx, slot_name in enumerate(gi.initial_slots.get(player_idx, [])):
            if chosen_stone.name == slot_name:
                slot = slot_idx
                break
        
        if slot is None:
            raise ValueError(f"Stone '{chosen_stone.name}' not found in initial slots")
        
        # Compute action: slot * (rows * cols) + (row * cols + col)
        action = slot * (rows * cols) + (row * cols + col)
        
        # Use existing step method
        return self.step(game_id, action, player_idx)

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
                    # Determine whose turn it is based on stones played
                    total_played = sum(len(ctx.game.initial_slots.get(i, [])) - len(p.stones) 
                                     for i, p in enumerate(ctx.game.players))
                    current_player_idx = total_played % 2
                    
                    # Check if game is over
                    if not ctx.game.started:
                        ctx.started = False
                        self._update_stats(ctx.game)
                        break
                    
                    # Check if current player has stones left
                    current_player = ctx.game.players[current_player_idx]
                    if len(current_player.stones) == 0:
                        # This player is out of stones, game should be over
                        ctx.game.check_game_over()
                        if not ctx.game.started:
                            ctx.started = False
                            self._update_stats(ctx.game)
                            break
                        time.sleep(0.1)
                        continue
                    
                    # Check if current player is AI
                    agent = ctx.agents[current_player_idx]
                    if agent is None:
                        # Human player's turn - pause autoplay
                        time.sleep(0.1)
                        continue
                    
                    # Make ONE move for the current AI player
                    obs = StateBuilder.build_gym_observation(ctx.game, current_player_idx)
                    legal_actions = self._get_legal_actions(ctx.game, current_player_idx)
                    
                    if not legal_actions:
                        time.sleep(0.1)
                        continue
                    
                    try:
                        action = current_player.choose_action(obs, legal_actions)
                        self.step(game_id, action, current_player_idx)
                        time.sleep(max(0.0, float(delay)))
                    except Exception as e:
                        self.logger.error(f"Agent error: {e}")
                        break

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
                filename = f"mc_agent_{rows}x{cols}.pkl.gz"
                model_path = os.path.join('models', filename)
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
                filename = f"q_agent_{rows}x{cols}.pkl.gz"
                model_path = os.path.join('models', filename)
                if os.path.exists(model_path):
                    agent = QLearningAgent.load(model_path, action_space)
                    self.logger.info(f"Loaded QLearningAgent from {model_path}")
                    return agent, model_path
                else:
                    raise FileNotFoundError(f"Q-learning agent model not found at {model_path}")
            except ImportError as e:
                self.logger.error(f"Failed to import Q agent: {e}")
                raise ValueError(f"Q-learning agent implementation not found: {e}")

        if control in ('reinforce', 'rf'):
            try:
                from algorithms.reinforce_agent import ReinforceAgent
                filename = "reinforce_agent.pth"
                model_path = os.path.join('models', filename)
                if os.path.exists(model_path):
                    # Calculate input dimension
                    input_dim = (rows * cols) + (rows * cols * 4) + (2 * max_slots * 4)
                    agent = ReinforceAgent(
                        action_space=action_space,
                        input_dim=input_dim,
                        gamma=0.99,
                        lr=1e-3,
                        hidden_dim=256  # Match training config
                    )
                    agent.load(model_path)
                    self.logger.info(f"Loaded ReinforceAgent from {model_path}")
                    return agent, model_path
                else:
                    raise FileNotFoundError(f"REINFORCE agent model not found at {model_path}")
            except ImportError as e:
                self.logger.error(f"Failed to import REINFORCE agent: {e}")
                raise ValueError(f"REINFORCE agent implementation not found: {e}")

        if control == 'medium':
            from algorithms.medium_agent import MediumAgent
            return MediumAgent(action_space), None

        if control == 'mcts':
            from algorithms.mcts_agent import MCTSAgent
            return MCTSAgent(action_space, simulations=1000), None

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
