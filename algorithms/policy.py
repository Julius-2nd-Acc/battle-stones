import json
import random
from typing import Callable, List, Tuple


def _decode_action(action_id: int, rows: int, cols: int) -> Tuple[int, int, int]:
    slot = action_id // (rows * cols)
    cell_idx = action_id % (rows * cols)
    r = cell_idx // cols
    c = cell_idx % cols
    return slot, r, c


def _encode_action(slot: int, r: int, c: int, rows: int, cols: int) -> int:
    return slot * (rows * cols) + (r * cols + c)


def _game_state_json(game, train_player_idx: int, rows: int, cols: int) -> str:
    board_repr = []
    for r in range(rows):
        for c in range(cols):
            cell = game.board.getField(r, c)
            if cell is None or getattr(cell, "player", None) is None:
                board_repr.append(".")
            else:
                owner_idx = 0 if game.players[0] == getattr(cell, "player") else 1
                board_repr.append(f"{owner_idx}:{cell.name}")

    players_stones = []
    for p in game.players:
        players_stones.append([s.name for s in p.stones])

    payload = {"board": board_repr, "players": players_stones, "to_move": train_player_idx}
    return json.dumps(payload, sort_keys=True)


def _generate_legal_actions(game, player_idx: int, rows: int, cols: int, max_stones: int, initial_slots: List[str]) -> List[int]:
    """Return list of legal action_ids for the given player based on current game state.

    initial_slots is a list mapping slot index -> stone name (may be None for empty slots).
    """
    actions = []
    player = game.players[player_idx]
    stone_names = {s.name for s in player.stones}

    for slot_idx, name in enumerate(initial_slots):
        if name is None:
            continue
        if name not in stone_names:
            # stone already played
            continue

        for r in range(rows):
            for c in range(cols):
                if game.board.isValidMove((r, c)):
                    actions.append(_encode_action(slot_idx, r, c, rows, cols))

    return actions


def make_inference_policy(agent, game, player_idx: int = 0, rows: int = 3, cols: int = 3, max_stones: int = 4) -> Callable[[str], int]:
    """Create a policy callable suitable for injecting into a Player.

    Behavior:
      - Query agent.greedy_action(state) to get a proposed action_id.
      - If invalid, evaluate all legal actions by agent.Q and pick the best available.
      - If no legal actions, pick a random legal move (shouldn't normally happen).

    The returned callable ignores the incoming state string and derives the state from the
    provided `game` object to keep decisions consistent with the live game.
    """
    # build initial slot mapping from current stones at injection time
    initial_slots = [s.name for s in game.players[player_idx].stones]
    if len(initial_slots) < max_stones:
        initial_slots += [None] * (max_stones - len(initial_slots))

    def policy(_state: str = None) -> int:
        # produce a canonical state string matching training
        state = _game_state_json(game, player_idx, rows, cols)

        # ask agent for greedy action
        try:
            proposed = agent.greedy_action(state)
        except Exception:
            # if agent cannot provide greedy action, fall back to random
            proposed = None

        legal_actions = _generate_legal_actions(game, player_idx, rows, cols, max_stones, initial_slots)

        if not legal_actions:
            # no legal actions -> return a random action (will likely be invalid)
            return 0

        if proposed in legal_actions:
            return proposed

        # proposed invalid: choose best legal by Q value
        best = None
        best_val = float('-inf')
        for a in legal_actions:
            val = agent.Q.get((state, a), 0.0)
            if val > best_val:
                best_val = val
                best = a

        if best is not None:
            return best

        # fallback: random legal action
        return random.choice(legal_actions)

    return policy
