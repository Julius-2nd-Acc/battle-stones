import os
from algorithms.game_env import SkystonesEnv
from algorithms.mix_agent import MixAgent
from algorithms.q_learning import QLearningAgent
from algorithms.trainer import Trainer

def train_qlearning(
    num_episodes: int = 10000,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon_start: float = 0.2,
    rows: int = 3,
    cols: int = 3,
    model_path: str | None = None,
):
    if model_path is None:
        model_path = f"models/q_agent_{rows}x{cols}.pkl.gz"

    env = SkystonesEnv(render_mode=None, capture_reward=1.0, rows=rows, cols=cols)

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        agent = QLearningAgent.load(model_path, env.action_space)
    else:
        print("Starting fresh training...")
        agent = QLearningAgent(
            action_space=env.action_space,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon_start,
        )
    
    # Inject decay params into agent for Trainer to use
    agent.epsilon_min = 0.10
    agent.epsilon_decay = 0.999
    opponent = MixAgent(action_space=env.action_space, epsilon=0.5)

    trainer = Trainer(
        agent=agent,
        env=env,
        model_path=model_path,
        save_interval=1000000,
        log_interval=1000,
        opponent= opponent
    )
    
    trainer.train(num_episodes)
    env.close()
    return agent


if __name__ == "__main__":
    train_qlearning(rows=3, cols=3, epsilon_start = 0.7, num_episodes=100000)
