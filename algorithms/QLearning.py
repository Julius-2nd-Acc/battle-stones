import random
from collections import defaultdict
from typing import Any, List, Tuple


class EpsilonGreedyQLearningAgent:
    """
    ε-Greedy Q-Learning agent (off-policy TD control).

    - Works with discrete action spaces: actions = 0, 1, ..., n_actions-1
    - Learns Q(s, a) from step-by-step TD updates (no need to wait for episode end).
    - Uses ε-greedy exploration and optional ε decay.
    - Supports constant or decaying learning rate α.
    """

    def __init__(
        self,
        n_actions: int,
        gamma: float = 1.0,
        alpha: float = 0.1,
        alpha_min: float = 0.0001,
        alpha_decay: float = 1.0,
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
        alpha : float
            Initial learning rate α.
        alpha_min : float
            Minimum α after decay.
        alpha_decay : float
            Multiplicative decay applied after each episode (e.g. 0.999).
            Use 1.0 for constant α.
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

        # Learning rate
        self.alpha = alpha
        self.alpha_start = alpha
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q[(state, action)] = estimated action-value
        self.Q = defaultdict(float)

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

        # Build Q-values using a safe lookup; initialize unseen keys to 0.0
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
    # Single Q-learning update
    # ------------------------------------------------------------------
    def _q_learning_update(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
    ) -> None:
        """
        Perform one Q-learning update:

        Q(s, a) <- Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
        """
        # Replace direct indexing that can raise KeyError with safe get/initialization
        key = (state, action)
        old_q = self.Q.get(key, 0.0)
        # ensure the key exists so other code won't KeyError later
        if key not in self.Q:
            self.Q[key] = old_q

        # compute max next Q safely
        if done:
            max_next_q = 0.0
        else:
            next_qs = []
            for a in range(self.n_actions):
                next_key = (next_state, a)
                if next_key not in self.Q:
                    # initialize unseen next state-action to 0.0
                    self.Q[next_key] = 0.0
                next_qs.append(self.Q[next_key])
            max_next_q = max(next_qs) if next_qs else 0.0

        # standard Q-learning update using old_q and max_next_q
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.Q[key] = new_q

    # ------------------------------------------------------------------
    # Run one episode and learn online via TD updates
    # ------------------------------------------------------------------
    def run_episode(
        self,
        env,
        max_steps: int = 10_000,
    ) -> Tuple[float, int]:
        """
        Play one episode in the environment using the current ε-greedy policy,
        updating Q at every step via Q-learning.

        Returns
        -------
        total_reward : float
            Sum of rewards in the episode.
        steps : int
            Number of steps taken in the episode.
        """
        state = env.reset()
        total_reward = 0.0
        steps = 0

        for _ in range(max_steps):
            action = self.select_action(state)
            step_result = env.step(action)

            # Classic Gym API: next_state, reward, done, info
            if len(step_result) == 4:
                next_state, reward, done, info = step_result
            else:
                # Gymnasium-style: obs, reward, terminated, truncated, info
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated

            # TD update
            self._q_learning_update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            total_reward += reward
            steps += 1
            state = next_state

            if done:
                break

        return total_reward, steps

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
        Run Q-learning for a number of episodes.

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
            total_reward, steps = self.run_episode(
                env, max_steps=max_steps_per_episode
            )

            # ε decay (for GLIE-style behavior)
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay,
            )

            # α decay (optional)
            self.alpha = max(
                self.alpha_min,
                self.alpha * self.alpha_decay,
            )

            if verbose and i % max(1, num_episodes // 10) == 0:
                print(
                    f"Episode {i}/{num_episodes} | "
                    f"epsilon={self.epsilon:.4f} | "
                    f"alpha={self.alpha:.5f} | "
                    f"reward={total_reward:.2f} | "
                    f"steps={steps}"
                )

    # ------------------------------------------------------------------
    # Utility: get greedy action for a state (after training)
    # ------------------------------------------------------------------
    def greedy_action(self, state: Any) -> int:
        """
        Return the greedy action for a given state (no ε-exploration).
        """
        q_values = [self.Q[(state, a)] for a in range(self.n_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def get_Q_table(self):
        """Return underlying Q-dict (useful for inspection / debugging)."""
        return dict(self.Q)

    def load_Q_table(self, qdict):
        """Load a Q dict (mapping (state,action) -> value) into the agent."""
        from collections import defaultdict

        self.Q = defaultdict(float, qdict)
