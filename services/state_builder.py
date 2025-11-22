import numpy as np
import json
from typing import Dict, Any, List, Tuple

class StateBuilder:
    """
    Shared logic for building game state observations for both
    the API (GameController) and the RL Environment (SkystonesEnv).
    """

    @staticmethod
    def build_gym_observation(game_instance, player_idx: int) -> Dict[str, Any]:
        """
        Builds the dictionary observation expected by the RL agents.
        
        Args:
            game_instance: The GameInstance object.
            player_idx: The index of the player whose perspective we are building.
            
        Returns:
            Dict containing 'board_owner', 'board_type', 'hand_types', 'to_move'.
        """
        rows = game_instance.board.rows
        cols = game_instance.board.cols
        max_slots = max((len(s) for s in game_instance.initial_slots.values()), default=0)
        
        # --- board: owner + type_id ---
        board_owner = np.zeros((rows, cols), dtype=np.int8)
        board_type = np.full((rows, cols), fill_value=-1, dtype=np.int8)
        
        # Build type mapping (Stone Attributes -> Type ID)
        # This should ideally be cached or constant, but we build it dynamically for now to match legacy behavior
        attr_to_type = {}
        type_id = 0
        
        # Scan all initial slots to build the type registry
        for p_idx, slot_names in game_instance.initial_slots.items():
            player = game_instance.players[p_idx]
            for name in slot_names:
                if name is None: continue
                
                # Find the stone object to get attributes
                stone_obj = StateBuilder._find_stone_by_name(game_instance, player, name)
                if stone_obj is None: continue
                
                attrs = StateBuilder._safe_get_attributes(stone_obj)
                if attrs not in attr_to_type:
                    attr_to_type[attrs] = type_id
                    type_id += 1

        # Fill Board Grids
        for r in range(rows):
            for c in range(cols):
                cell = game_instance.board.getField(r, c)
                # Check if cell is not empty (Field.EMPTY is enum 0, but we check for object presence/attributes)
                if hasattr(cell, "player"): 
                    owner = 0 if cell.player == game_instance.players[0] else 1
                    board_owner[r, c] = 1 if owner == 0 else 2 # 1 for P0, 2 for P1 (matching legacy env)
                    
                    attrs = StateBuilder._safe_get_attributes(cell)
                    board_type[r, c] = attr_to_type.get(attrs, -1)

        # Fill Hand Grids
        hand_types = np.full((2, max_slots), fill_value=-1, dtype=np.int8)
        for p_idx, player in enumerate(game_instance.players):
            slot_names = game_instance.initial_slots.get(p_idx, [])
            names_in_hand = {s.name: s for s in player.stones}
            
            for slot_idx, stone_name in enumerate(slot_names):
                if slot_idx >= max_slots: break
                if stone_name is None: continue
                
                stone_obj = names_in_hand.get(stone_name, None)
                if stone_obj is None: continue # Stone played
                
                attrs = StateBuilder._safe_get_attributes(stone_obj)
                hand_types[p_idx, slot_idx] = attr_to_type.get(attrs, -1)

        to_move = np.array(player_idx, dtype=np.int8)

        return {
            "board_owner": board_owner,
            "board_type": board_type,
            "hand_types": hand_types,
            "to_move": to_move,
        }

    @staticmethod
    def build_canonical_state(game_instance, player_idx: int) -> str:
        """
        Builds the JSON string state used by the API and some legacy policy calls.
        """
        board_repr = []
        for r in range(game_instance.board.rows):
            for c in range(game_instance.board.cols):
                cell = game_instance.board.getField(r, c)
                if cell is None or not hasattr(cell, "player"):
                    board_repr.append(".")
                else:
                    owner_idx = 0 if game_instance.players[0] == cell.player else 1
                    board_repr.append(f"{owner_idx}:{cell.name}")

        players_stones = []
        for p in game_instance.players:
            players_stones.append([s.name for s in p.stones])

        payload = {"board": board_repr, "players": players_stones, "to_move": player_idx}
        return json.dumps(payload, sort_keys=True)

    @staticmethod
    def _find_stone_by_name(game_instance, player, name):
        # Check player's hand
        for s in player.stones:
            if s.name == name: return s
            
        # Check board
        for r in range(game_instance.board.rows):
            for c in range(game_instance.board.cols):
                cell = game_instance.board.getField(r, c)
                if hasattr(cell, 'name') and cell.name == name:
                    return cell
        return None

    @staticmethod
    def _safe_get_attributes(stone_obj):
        try:
            return tuple(stone_obj.get_Attributes())
        except Exception:
            return (0, 0, 0, 0)

    @staticmethod
    def get_legal_actions(game_instance, player_idx: int) -> List[int]:
        """
        Returns a list of legal action indices for the given player.
        """
        legal_actions = []
        rows = game_instance.board.rows
        cols = game_instance.board.cols
        cells = rows * cols
        
        player = game_instance.players[player_idx]
        slot_names = game_instance.initial_slots.get(player_idx, [])
        available_names = {s.name for s in player.stones}
        
        for slot, sname in enumerate(slot_names):
            if sname is None: continue
            if sname not in available_names: continue
            
            for rr in range(rows):
                for cc in range(cols):
                    if game_instance.board.isValidMove((rr, cc)):
                        legal_actions.append(slot * cells + (rr * cols + cc))
        return legal_actions
