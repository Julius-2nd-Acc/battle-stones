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
        
        # --- board: ownership + stats (n,s,e,w) ---
        ownership = np.zeros((rows, cols), dtype=np.int8)
        board_stats = np.zeros((rows, cols, 4), dtype=np.int8)
        
        # Fill Board Grids
        for r in range(rows):
            for c in range(cols):
                cell = game_instance.board.getField(r, c)
                # Check if cell is not empty
                if hasattr(cell, "player"): 
                    owner = 0 if cell.player == game_instance.players[0] else 1
                    ownership[r, c] = 1 if owner == 0 else 2 # 1 for P0, 2 for P1
                    
                    attrs = StateBuilder._safe_get_attributes(cell) # (n, s, e, w)
                    board_stats[r, c] = np.array(attrs, dtype=np.int8)

        # Fill Hand Grids
        hand_stats = np.zeros((2, max_slots, 4), dtype=np.int8)
        for p_idx, player in enumerate(game_instance.players):
            slot_names = game_instance.initial_slots.get(p_idx, [])
            names_in_hand = {s.name: s for s in player.stones}
            
            for slot_idx, stone_name in enumerate(slot_names):
                if slot_idx >= max_slots: break
                if stone_name is None: continue
                
                stone_obj = names_in_hand.get(stone_name, None)
                if stone_obj is None: continue # Stone played
                
                attrs = StateBuilder._safe_get_attributes(stone_obj)
                hand_stats[p_idx, slot_idx] = np.array(attrs, dtype=np.int8)

        to_move = np.array(player_idx, dtype=np.int8)

        return {
            "ownership": ownership,
            "board_stats": board_stats,
            "hand_stats": hand_stats,
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
        print(payload)
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
