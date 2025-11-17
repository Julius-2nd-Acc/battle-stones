import random
from collections import defaultdict
from typing import Any, List, Tuple


class EpsilonGreedyMCAgent:
    """
    ε-Greedy Monte Carlo Control agent (first-visit).
    
    - Works with discrete action spaces: actions = 0, 1, ..., n_actions-1
    - Learns Q(s, a) from complete episodes.
    - Uses ε-greedy exploration and optional ε decay.
    """

    def __init__(
        self,
        n_actions: int,
        gamma: float = 1.0,
        epsilon_start: float = 0.1,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 1.0,
    ):
        """
        Parameters
        ----------
        n_actions : int
            Number of discrete actions in your game.
        gamma : float
            Discount factor (0 <= gamma <= 1).
        epsilon_start : float
            Initial exploration rate ε.
        epsilon_min : float
            Minimum ε after decay.
        epsilon_decay : float
            Multiplicative decay applied after each episode (e.g. 0.999).
            Use 1.0 for constant ε.
        """
        self.n_actions = n_actions
        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q[(state, action)] = estimated return
        self.Q = defaultdict(float)

        # N[(state, action)] = number of first-visit updates
        self.N = defaultdict(int)

    # ------------------------------------------------------------------
    # Policy: ε-greedy with respect to current Q
    # ------------------------------------------------------------------
    def select_action(self, state: Any) -> int:
        """
        Select an action using ε-greedy policy based on current Q.
        """
        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        # Build Q-values using a safe lookup; initialize missing entries to 0.0
        q_values = []
        for a in range(self.n_actions):
            key = (state, a)
            if key not in self.Q:
                # initialize unseen state-action with 0.0 so future accesses won't KeyError
                self.Q[key] = 0.0
            q_values.append(self.Q[key])

        max_q = max(q_values)
        max_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(max_actions)

    # ------------------------------------------------------------------
    # Generate one episode by interacting with the environment
    # ------------------------------------------------------------------
    def generate_episode(
        self,
        env,
        max_steps: int = 10_000,
    ) -> List[Tuple[Any, int, float]]:
        """
        Play one episode in the environment using the current ε-greedy policy.

        Returns
        -------
        episode : list of (state, action, reward)
        """
        state = env.reset()
        episode = []

        for _ in range(max_steps):
            action = self.select_action(state)
            # Classic Gym API: next_state, reward, done, info
            step_result = env.step(action)
            if len(step_result) == 4:
                next_state, reward, done, info = step_result
            else:
                # If using gymnasium-style (obs, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            episode.append((state, action, reward))
            state = next_state

            if done:
                break

        return episode

    # ------------------------------------------------------------------
    # First-visit Monte Carlo update from a complete episode
    # ------------------------------------------------------------------
    def update_from_episode(self, episode: List[Tuple[Any, int, float]]) -> None:
        """
        Update Q using first-visit Monte Carlo from a single episode.

        episode: list of (state, action, reward)
        """
        G = 0.0
        visited = set()  # to enforce first-visit behavior

        # Traverse the episode backwards
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward

            if (state, action) in visited:
                continue  # every-visit would skip this check

            visited.add((state, action))

            # Increment first-visit count
            self.N[(state, action)] += 1
            n = self.N[(state, action)]

            # Use safe lookup for old Q value; initialize if missing
            key = (state, action)
            old_q = self.Q.get(key, 0.0)
            if key not in self.Q:
                self.Q[key] = old_q

            # Incremental mean update:
            # Q <- Q + 1/n * (G - Q)
            self.Q[key] = old_q + (G - old_q) / n

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(
        self,
        env,
        num_episodes: int,
        max_steps_per_episode: int = 10_000,
        verbose: bool = False,
    ):
        """
        Run Monte Carlo control for a number of episodes.

        Parameters
        ----------
        env :
            Your game environment (must support reset() and step(action)).
        num_episodes : int
            Number of episodes to run.
        max_steps_per_episode : int
            Safety cap for episode length.
        verbose : bool
            If True, prints basic progress.
        """
        for i in range(1, num_episodes + 1):
            episode = self.generate_episode(env, max_steps=max_steps_per_episode)
            self.update_from_episode(episode)

            # ε decay (for GLIE-style behavior)
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

            if verbose and i % max(1, num_episodes // 10) == 0:
                print(
                    f"Episode {i}/{num_episodes} | "
                    f"epsilon={self.epsilon:.4f} | episode_length={len(episode)}"
                )

    # ------------------------------------------------------------------
    # Utility: get greedy action for a state (after training)
    # ------------------------------------------------------------------
    def greedy_action(self, state: Any) -> int:
        """
        Return the greedy action for a given state (no ε-exploration).
        """
        # Use safe lookups and initialize missing entries
        q_values = []
        for a in range(self.n_actions):
            key = (state, a)
            if key not in self.Q:
                self.Q[key] = 0.0
            q_values.append(self.Q[key])

        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        import random
        return random.choice(best_actions)

    def get_Q_table(self):
        """Return underlying Q-dict (useful for inspection / debugging)."""
        return dict(self.Q)

    def load_Q_table(self, qdict):
        """Load a Q dict (mapping (state,action) -> value) into the agent."""
        from collections import defaultdict

        self.Q = defaultdict(float, qdict)
