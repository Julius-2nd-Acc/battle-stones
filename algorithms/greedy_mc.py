import random
import pickle
from collections import defaultdict
from typing import Any, List

import numpy as np
from algorithms.agent_interface import Agent

def obs_to_state(obs) -> tuple:
    """
    Convert SkystonesEnv observation dict into a hashable state key (tuple).
    Matches the *current* observation structure in SkystonesEnv.
    """
    board_owner_flat = tuple(obs["board_owner"].astype(int).ravel())
    board_type_flat = tuple(obs["board_type"].astype(int).ravel())
    hand_flat = tuple(obs["hand_types"].astype(int).ravel())
    to_move = int(obs["to_move"])
    return board_owner_flat + board_type_flat + hand_flat + (to_move,)

class MCAgent(Agent):
    """
    First-visit Monte Carlo control (ε-greedy).
    Q: state -> action values (tabular)
    """

    def __init__(self, action_space, gamma: float = 0.99, epsilon: float = 0.1):
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon

        # Q(s)[a] = value
        self.Q = defaultdict(self._zeros_for_state)

        # For MC returns
        self.returns_sum = defaultdict(float)   # (s,a) -> sum of returns
        self.returns_count = defaultdict(int)   # (s,a) -> count of returns

    def _zeros_for_state(self):
        return np.zeros(self.action_space.n, dtype=np.float32)
    
    def choose_action(self, observation: Any, legal_actions: List[int] | None = None) -> int:
        """
        Implementation of Agent interface.
        Uses GREEDY masked policy (no epsilon) for inference.
        """
        if legal_actions is not None:
            return self.greedy_action_masked(observation, legal_actions)
        return self.greedy_action(observation)

    # --------- policy / state helpers --------- #

    def get_state_key(self, obs):
        return obs_to_state(obs)
    
    def policy_action_masked(self, obs, legal_actions):
        """
        ε-greedy action selection but restricted to legal_actions.
        """
        # If no legal actions are available, fall back to something safe
        if not legal_actions:
            # caller should normally handle terminal states before this,
            # but we guard anyway:
            return self.action_space.sample()

        state = self.get_state_key(obs)

        # Exploration: random legal action
        if random.random() < self.epsilon:
            return random.choice(legal_actions)

        # Exploitation: choose legal action with max Q(s,a)
        q_values = self.Q[state]
        best_a = max(legal_actions, key=lambda a: q_values[a])
        return int(best_a)

    def greedy_action_masked(self, obs, legal_actions):
        """
        Greedy action over legal_actions (for evaluation).
        """
        if not legal_actions:
            return self.action_space.sample()

        state = self.get_state_key(obs)
        q_values = self.Q[state]
        best_a = max(legal_actions, key=lambda a: q_values[a])
        return int(best_a)

    def policy_action(self, obs) -> int:
        """
        ε-greedy policy used during training.
        """
        state = self.get_state_key(obs)
        if random.random() < self.epsilon:
            return self.action_space.sample()
        q_values = self.Q[state]
        return int(np.argmax(q_values))

    def greedy_action(self, obs) -> int:
        """
        Pure greedy action (for evaluation).
        """
        state = self.get_state_key(obs)
        q_values = self.Q[state]
        return int(np.argmax(q_values))

    # --------- MC update --------- #

    def update_from_episode(self, episode):
        """
        episode: list of (state_key, action, reward)

        First-visit Monte Carlo:
        - walk backwards through the episode
        - compute return G
        - for the first time (s,a) appears (from the end), update its Q(s,a)
        """
        G = 0.0
        visited = set()

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = self.gamma * G + r

            if (s, a) in visited:
                continue
            visited.add((s, a))

            self.returns_sum[(s, a)] += G
            self.returns_count[(s, a)] += 1
            self.Q[s][a] = self.returns_sum[(s, a)] / self.returns_count[(s, a)]

    # --------- save / load --------- #

    def save(self, filepath: str):
        """
        Save the agent to disk.
        """
        data = {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "Q": {state: q.tolist() for state, q in self.Q.items()},
            "returns_sum": dict(self.returns_sum),
            "returns_count": dict(self.returns_count),
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str, action_space):
        """
        Load an agent from disk. You must pass the env's action_space.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        agent = cls(
            action_space=action_space,
            gamma=data["gamma"],
            epsilon=data["epsilon"],
        )

        agent.Q = defaultdict(
            agent._zeros_for_state,
            {state: np.array(q, dtype=np.float32) for state, q in data["Q"].items()},
        )
        agent.returns_sum = defaultdict(float, data["returns_sum"])
        agent.returns_count = defaultdict(int, data["returns_count"])

        return agent
