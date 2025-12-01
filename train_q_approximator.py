from algorithms.game_env import SkystonesEnv
from algorithms.trainer import Trainer
from algorithms.function_approximation.approximator_agent import FunctionApproximationAgent
from algorithms.function_approximation.model_classes import SmallQNet

def train(
    model_cls,
    hidden_dim=128,
    gamma=0.99,
    lr=1e-3,
    epsilon=0.99,
    epsilon_min=0.05,
    epsilon_decay=0.999,
    model_path='models/small_qnet.pkl',
    model_kwargs=None   
):
    env = SkystonesEnv()

    agent = FunctionApproximationAgent(
        env=env,
        model_cls=model_cls,
        hidden_dim=hidden_dim,
        model_kwargs=model_kwargs,
        gamma=gamma,
        lr=lr,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    trainer = Trainer(
        agent=agent,
        env=env,
        model_path=model_path,
        save_interval=5000,
        log_interval=1000
    )

    trainer.train(num_episodes=50000)

if __name__ == '__main__':
    train(SmallQNet, hidden_dim=128)