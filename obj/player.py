from enum import Enum
from typing import Any, Optional, List
from algorithms.agent_interface import Agent

class PlayerType(Enum):
    HUMAN = 0
    RANDOM = 1
    MC = 2
    Q = 3
    MEDIUM = 4

class Player:
    def __init__(self, name: str, player_type: PlayerType, stones: list | None = None):
        self.name = name
        self.player_type = player_type
        # avoid mutable default argument — ensure each Player gets its own list
        self.stones = list(stones) if stones is not None else []
        
        # The agent controlling this player (if any)
        self.agent: Optional[Agent] = None
        
        # Metadata about loaded agent (kept for API compatibility)
        self._agent_info = {"type": self.player_type.name.lower(), "present": False, "loaded_from": None}

    def add_stone(self, stone):
        self.stones.append(stone)
        
    def remove_stone(self, stone):
        self.stones.remove(stone)

    def assign_agent(self, agent: Agent, loaded_from: str | None = None):
        """Assign an agent to control this player."""
        self.agent = agent
        self._agent_info["present"] = True
        if loaded_from:
            self._agent_info["loaded_from"] = loaded_from

    def choose_action(self, observation: Any, legal_actions: List[int] | None = None) -> int:
        """
        Choose an action using the assigned agent.
        If no agent is assigned, raises an error (or could fallback to random/human input logic if desired).
        """
        if self.agent is None:
            raise RuntimeError(f"Player {self.name} has no agent assigned to choose action.")
            
        return self.agent.choose_action(observation, legal_actions)

    def set_name(self, name: str):
        """Set or update the player's display name."""
        self.name = name