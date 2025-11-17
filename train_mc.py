from pathlib import Path
import random

from algorithms.greedy_mc import MCAgent
from algorithms.game_env import SkystonesEnv


def train_mc(
    num_episodes: int = 50000,
    gamma: float = 0.99,
    epsilon_start: float = 0.2,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.999,
    model_path: str = "models/mc_agent_skystones.pkl",
    log_interval: int = 1000,
    save_interval: int = 5000,
):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    env = SkystonesEnv(render_mode=None, capture_reward=1.0)
    agent = MCAgent(action_space=env.action_space, gamma=gamma, epsilon=epsilon_start)

    print(f"Starting MC training (P0 vs random P1) for {num_episodes} episodes")

    for episode_idx in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        episode = []
        total_reward = 0.0

        while not done:
            state_key = agent.get_state_key(obs)
            player_to_act = env.current_player_idx

            legal_actions = env.get_legal_actions(player_to_act)

            if player_to_act == 0:
                # MC controls Player 0 with masked ε-greedy
                action = agent.policy_action_masked(obs, legal_actions)
            else:
                # Random opponent for Player 1: random legal move
                action = random.choice(legal_actions) if legal_actions else env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode.append((state_key, action, reward))
            total_reward += reward
            obs = next_obs

        # MC update (only uses P0 states internally, if you implemented that)
        agent.update_from_episode(episode)

        agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

        if episode_idx % log_interval == 0:
            print(
                f"[Episode {episode_idx}/{num_episodes}] "
                f"total_reward={total_reward:.3f}  epsilon={agent.epsilon:.4f}"
            )

        if episode_idx % save_interval == 0:
            tmp_path = model_path.with_suffix(".tmp.pkl")
            agent.save(tmp_path)
            tmp_path.replace(model_path)
            print(f"Saved MC agent checkpoint to: {model_path}")

    agent.save(model_path)
    print(f"Training finished. Final MC agent saved to: {model_path}")
    env.close()
    return agent


if __name__ == "__main__":
    train_mc()
