from algorithms.game_env import SkystonesEnv
from algorithms.greedy_mc import MCAgent
from algorithms.trainer import Trainer

def train_mc(
    num_episodes: int = 100000,
    gamma: float = 0.99,
    epsilon_start: float = 0.2,
    model_path: str = "models/mc_agent_skystones.pkl",
):
    env = SkystonesEnv(render_mode=None, capture_reward=1.0)
    agent = MCAgent(action_space=env.action_space, gamma=gamma, epsilon=epsilon_start)
    
    # Inject decay params
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
    train_mc()
