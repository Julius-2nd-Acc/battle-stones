from enum import Enum


class PlayerType(Enum):
    HUMAN = 0
    AI = 1

class Player:
    def __init__(self, name: str, player_type: PlayerType, stones: list = []):
        self.name = name
        self.player_type = player_type
        self.stones = stones
        # policy is a callable(state) -> action_id used at inference time
        self.policy = None

    def add_stone(self, stone):
        self.stones.append(stone)
        
    def remove_stone(self, stone):
        self.stones.remove(stone)

    def set_policy(self, policy_callable):
        """Set a policy for inference: callable(state)->action_id"""
        self.policy = policy_callable

    def choose_action(self, state):
        if self.policy is None:
            raise RuntimeError("No policy set for player")
        return self.policy(state)

    def set_name(self, name: str):
        """Set or update the player's display name."""
        self.name = name