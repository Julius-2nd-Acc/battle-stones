from pathlib import Path
import random

from algorithms.game_env import SkystonesEnv
from algorithms.q_learning import QLearningAgent


def train_qlearning(
    num_episodes: int = 10000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 0.2,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.999,
    model_path: str = "models/q_agent_skystones.pkl",
    log_interval: int = 1000,
    save_interval: int = 5000,
):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    env = SkystonesEnv(render_mode=None, capture_reward=1.0)

    agent = QLearningAgent(
        action_space=env.action_space,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon_start,
    )

    print(f"Starting Q-learning training: Player 0 = Q, Player 1 = random")
    print(f"Episodes: {num_episodes}")

    for episode_idx in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        # After env.reset(), current_player_idx should be 0 (Player 0 starts)
        while not done:
            # ----- Player 0 (Q-learning agent) move -----
            if env.current_player_idx != 0:
                # Safety check – in this setup we always expect P0 at this point
                # but if not, we can break to avoid weirdness.
                # Alternatively, you could loop until current_player_idx == 0.
                pass

            legal_actions_p0 = env.get_legal_actions(player_idx=0)

            # ε-greedy over legal actions for Player 0
            action_p0 = agent.policy_action_masked(obs, legal_actions_p0)

            next_obs, reward_p0, terminated, truncated, info = env.step(action_p0)
            done = terminated or truncated
            total_reward += reward_p0

            # Legal actions in the *next* state (whatever player is to move there)
            if not done:
                legal_next_actions = env.get_legal_actions()
            else:
                legal_next_actions = []

            # Q-learning update for Player 0 move
            agent.update(
                obs,
                action_p0,
                reward_p0,
                next_obs,
                done,
                legal_next_actions=legal_next_actions,
            )

            obs = next_obs

            if done:
                break

            # ----- Player 1 (random opponent) move -----
            # Now current_player_idx should be 1
            legal_actions_p1 = env.get_legal_actions(player_idx=1)
            if legal_actions_p1:
                action_p1 = random.choice(legal_actions_p1)
            else:
                # No legal moves – environment will handle this (likely terminal)
                action_p1 = env.action_space.sample()

            next_obs, reward_p1, terminated, truncated, info = env.step(action_p1)
            done = terminated or truncated
            total_reward += reward_p1

            # We DO NOT update Q-learning on Player 1's moves
            obs = next_obs

        # epsilon decay
        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

        # logging
        if episode_idx % log_interval == 0:
            print(
                f"[Episode {episode_idx}/{num_episodes}] "
                f"total_reward={total_reward:.3f}  epsilon={agent.epsilon:.4f}"
            )

        # periodic saving
        if episode_idx % save_interval == 0:
            tmp_path = model_path.with_suffix(".tmp.pkl")
            agent.save(tmp_path)
            tmp_path.replace(model_path)
            print(f"Saved Q-learning agent checkpoint to: {model_path}")

    # final save
    agent.save(model_path)
    print(f"Training finished. Final Q-learning agent saved to: {model_path}")

    env.close()
    return agent


if __name__ == "__main__":
    train_qlearning()
