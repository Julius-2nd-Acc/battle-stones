import numpy as np
from typing import Tuple, List

class CompactStateBuilder:
    """
    Helper class to build compact, abstracted state representations for tabular RL agents.
    Reduces state space size by quantizing stats and canonicalizing hands.
    """
    
    @staticmethod
    def quantize(val: int) -> int:
        """
        Quantize a 0-20 stat value into buckets.
        0 -> 0 (Weak)
        1-2 -> 1 (Medium)
        3+ -> 2 (Strong)
        """
        if val == 0: return 0
        if val <= 2: return 1
        if val >= 3: return 2

    @staticmethod
    def get_hand_signature(hand_stats: np.ndarray) -> Tuple:
        """
        Create a canonical signature for a hand of stones.
        Sorts stones by total power to remove permutation redundancy.
        
        Args:
            hand_stats: (max_slots, 4) array of stats
            
        Returns:
            Tuple of sorted stone tuples.
        """
        stones = []
        for i in range(hand_stats.shape[0]):
            stats = hand_stats[i]
            # Check if slot is empty (all zeros)
            if np.all(stats == 0):
                continue
                
            # Quantize stats
            q_stats = tuple(CompactStateBuilder.quantize(x) for x in stats)
            
            # Calculate total power for sorting
            power = sum(stats) # Use raw power for sorting to be more precise, or quantized?
            # Let's use quantized stats for the signature itself, but sort deterministically.
            # If we sort by raw power, we might split "identical" quantized stones.
            # Let's sort by the quantized tuple itself.
            stones.append(q_stats)
            
        # Sort stones to canonicalize
        stones.sort()
        return tuple(stones)

    @staticmethod
    def build_compact_state_key(obs) -> Tuple:
        """
        Convert a Gym observation into a compact, hashable state key.
        
        Args:
            obs: Gym observation dict
            
        Returns:
            Hashable tuple representing the state.
        """
        ownership = obs["ownership"] # (rows, cols)
        board_stats = obs["board_stats"] # (rows, cols, 4)
        hand_stats = obs["hand_stats"] # (2, max_slots, 4)
        to_move = int(obs["to_move"])
        
        # 1. Compact Board
        # Normalize based on current player (to_move)
        # If to_move == 0: Me=1, Opp=2
        # If to_move == 1: Me=2, Opp=1
        
        me_id = to_move + 1
        opp_id = 2 if me_id == 1 else 1
        
        board_list = []
        rows, cols = ownership.shape
        for r in range(rows):
            for c in range(cols):
                owner = ownership[r, c]
                if owner == 0:
                    board_list.append((0, (0,0,0,0)))
                else:
                    stats = board_stats[r, c]
                    q_stats = tuple(CompactStateBuilder.quantize(x) for x in stats)
                    
                    # Normalize owner: 1 if Me, 2 if Opponent
                    if owner == me_id:
                        norm_owner = 1
                    else:
                        norm_owner = 2
                        
                    board_list.append((norm_owner, q_stats))
        
        board_tuple = tuple(board_list)
        
        # 2. Compact Hands
        # Normalize hands: My Hand, Opponent Hand
        my_hand_stats = hand_stats[to_move]
        opp_hand_stats = hand_stats[1 - to_move]
        
        my_hand = CompactStateBuilder.get_hand_signature(my_hand_stats)
        opp_hand = CompactStateBuilder.get_hand_signature(opp_hand_stats)
        
        # We no longer need 'to_move' in the key because the state is relative to "Me"
        return (board_tuple, my_hand, opp_hand)
