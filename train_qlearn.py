from algorithms.game_env import SkystonesEnv
from algorithms.q_learning import QLearningAgent
from algorithms.trainer import Trainer

def train_qlearning(
    num_episodes: int = 10000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 0.2,
    model_path: str = "models/q_agent_skystones.pkl",
):
    env = SkystonesEnv(render_mode=None, capture_reward=1.0)

    agent = QLearningAgent(
        action_space=env.action_space,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon_start,
    )
    
    # Inject decay params into agent for Trainer to use
    agent.epsilon_min = 0.01
    agent.epsilon_decay = 0.999

    trainer = Trainer(
        agent=agent,
        env=env,
        model_path=model_path,
        save_interval=5000,
        log_interval=1000
    )
    
    trainer.train(num_episodes)
    env.close()
    return agent


if __name__ == "__main__":
    train_qlearning(100000)
