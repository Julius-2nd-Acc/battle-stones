from abc import ABC, abstractmethod
from typing import Any, List

class Agent(ABC):
    """
    Abstract base class for any entity that controls a Player in Battle Stones.
    """
    
    @abstractmethod
    def choose_action(self, observation: Any, legal_actions: List[int] | None = None) -> int:
        """
        Choose an action based on the observation.
        
        Args:
            observation: The current state of the game (format depends on implementation, usually dict).
            legal_actions: Optional list of legal action indices.
            
        Returns:
            int: The chosen action index.
        """
        pass
