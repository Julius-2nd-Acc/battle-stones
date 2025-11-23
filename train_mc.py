import os
from algorithms.game_env import SkystonesEnv
from algorithms.greedy_mc import MCAgent
from algorithms.trainer import Trainer

def train_mc(
    num_episodes: int = 100000,
    gamma: float = 0.99,
    epsilon_start: float = 0.2,
    rows: int = 3,
    cols: int = 3,
    model_path: str | None = None,
):
    if model_path is None:
        model_path = f"models/mc_agent_{rows}x{cols}.pkl.gz"
        
    env = SkystonesEnv(render_mode=None, capture_reward=1.0, rows=rows, cols=cols)
    
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        agent = MCAgent.load(model_path, env.action_space)
    else:
        print("Starting fresh training...")
        agent = MCAgent(action_space=env.action_space, gamma=gamma, epsilon=epsilon_start)
    
    # Inject decay params
    agent.epsilon_min = 0.01
    agent.epsilon_decay = 0.999

    trainer = Trainer(
        agent=agent,
        env=env,
        model_path=model_path,
        save_interval=50000,
        log_interval=10000
    )
    
    trainer.train(num_episodes)
    env.close()
    return agent


if __name__ == "__main__":
    train_mc(rows=3, cols=3, epsilon_start = 0.7, num_episodes=10000)
