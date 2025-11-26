import random
import pickle
import gzip
from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
from algorithms.agent_interface import Agent
from services.compact_state import CompactStateBuilder


def obs_to_state(obs) -> tuple:
    """
    Convert SkystonesEnv observation dict into a hashable state key (tuple).
    Matches the current observation structure in SkystonesEnv.
    """
    ownership = obs["ownership"]
    board_stats = obs["board_stats"]
    hand_stats = obs["hand_stats"]
    to_move = int(obs["to_move"])

    # Normalize ownership
    me_id = to_move + 1
    # Create a copy or new array for normalized ownership
    # 0 -> 0
    # me_id -> 1
    # other -> 2
    norm_ownership = np.zeros_like(ownership)
    norm_ownership[ownership == me_id] = 1
    norm_ownership[(ownership != 0) & (ownership != me_id)] = 2
    
    ownership_flat = tuple(norm_ownership.astype(int).ravel())
    board_stats_flat = tuple(board_stats.astype(int).ravel())
    
    # Normalize hands
    my_hand = tuple(hand_stats[to_move].astype(int).ravel())
    opp_hand = tuple(hand_stats[1 - to_move].astype(int).ravel())
    
    return ownership_flat + board_stats_flat + my_hand + opp_hand


class QLearningAgent(Agent):
    """
    Tabular Q-learning with ε-greedy *masked* policy.

    Q: state -> action values
    """

    def __init__(
        self,
        action_space,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        use_compact_state: bool = True
    ):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.use_compact_state = use_compact_state

        # Q(s)[a] = value
        self.Q = defaultdict(self._zeros_for_state)

    def _zeros_for_state(self):
        return np.zeros(self.action_space.n, dtype=np.float16)
    
    def choose_action(self, observation: Any, legal_actions: List[int] | None = None) -> int:
        """
        Implementation of Agent interface.
        Uses GREEDY masked policy (no epsilon) for inference.
        """
        if legal_actions is not None:
            return self.greedy_action_masked(observation, legal_actions)
        return self.greedy_action(observation)

    # --------- state / policy helpers --------- #

    def get_state_key(self, obs):
        if self.use_compact_state:
            return CompactStateBuilder.build_compact_state_key(obs)
        return obs_to_state(obs)

    # Unmasked (fallback) versions, if you ever want them
    def select_action(self, obs) -> int:
        state = self.get_state_key(obs)
        if random.random() < self.epsilon:
            return self.action_space.sample()
        q_values = self.Q[state]
        return int(np.argmax(q_values))

    def greedy_action(self, obs) -> int:
        state = self.get_state_key(obs)
        q_values = self.Q[state]
        return int(np.argmax(q_values))

    # Masked versions (recommended for training & playing)

    def policy_action_masked(self, obs, legal_actions) -> int:
        """
        ε-greedy action selection restricted to legal_actions.
        """
        if not legal_actions:
            # No legal actions (should only happen at terminal), fallback:
            return self.action_space.sample()

        state = self.get_state_key(obs)

        # Exploration: random legal action
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        # Exploitation: legal action with max Q(s,a)
        q_values = self.Q[state]
        best_a = max(legal_actions, key=lambda a: q_values[a])
        return int(best_a)

    def greedy_action_masked(self, obs, legal_actions) -> int:
        """
        Pure greedy selection over legal_actions (for evaluation).
        """
        if not legal_actions:
            return self.action_space.sample()

        state = self.get_state_key(obs)
        q_values = self.Q[state]
        best_a = max(legal_actions, key=lambda a: q_values[a])
        return int(best_a)

    # --------- Q-learning update --------- #

    def update(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        done: bool,
        legal_next_actions=None,
    ):
        """
        Standard Q-learning update, but you *can* pass legal_next_actions to
        restrict the max over actions in the next state.
        """
        state = self.get_state_key(obs)
        next_state = self.get_state_key(next_obs)

        q_values = self.Q[state]
        q_next = self.Q[next_state]

        if done:
            best_next = 0.0
        else:
            if legal_next_actions:
                best_next = max(q_next[a] for a in legal_next_actions)
            else:
                best_next = float(np.max(q_next))

        td_target = reward + self.gamma * best_next
        td_error = td_target - q_values[action]

        q_values[action] += self.alpha * td_error

    # --------- save / load --------- #

    def save(self, filepath: str):
        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "use_compact_state": self.use_compact_state,
            "Q": {state: q.tolist() for state, q in self.Q.items()},
        }
        # Use gzip for compression
        with gzip.open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str, action_space):
        # Use gzip for decompression
        with gzip.open(filepath, "rb") as f:
            data = pickle.load(f)

        # Handle legacy models
        use_compact = data.get("use_compact_state", False)

        agent = cls(
            action_space=action_space,
            alpha=data["alpha"],
            gamma=data["gamma"],
            epsilon=data["epsilon"],
            use_compact_state=use_compact
        )

        agent.Q = defaultdict(
            agent._zeros_for_state,
            {state: np.array(q, dtype=np.float16) for state, q in data["Q"].items()},
        )
        return agent
