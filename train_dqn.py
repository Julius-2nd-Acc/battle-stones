from algorithms.game_env import SkystonesEnv
from algorithms.trainer import Trainer
from algorithms.function_approximation.dqn_agent import DQNAgent
from algorithms.function_approximation.model_classes import SmallQNet
from algorithms.mix_agent import MixAgent
import os

def train(
    model_cls,
    hidden_dim=128,
    gamma=0.8,
    lr=1e-3,
    epsilon=0.99,
    epsilon_min=0.05,
    epsilon_decay=0.9,
    model_path='models/dqn.pkl',
    target_update_freq=1000,
    model_kwargs=None   
):
    env = SkystonesEnv()

    
    opponent = MixAgent(action_space=env.action_space, epsilon=0.4)
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        agent = DQNAgent.load(model_path, env=env,
        model_cls=model_cls,
        hidden_dim=hidden_dim,
        model_kwargs=model_kwargs,
        gamma=gamma,
        lr=lr,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq)
    else:
        print("Starting fresh training...")
        agent = DQNAgent(
        env=env,
        model_cls=model_cls,
        hidden_dim=hidden_dim,
        model_kwargs=model_kwargs,
        gamma=gamma,
        lr=lr,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq
    )

    trainer = Trainer(
        agent=agent,
        opponent=opponent,
        env=env,
        model_path=model_path,
        save_interval=2000,
        log_interval=1000,
        randomize_player=True
    )

    trainer.train(num_episodes=20000)

if __name__ == '__main__':
    train(SmallQNet, hidden_dim=256, model_path='models/dqn_3x3.pkl')