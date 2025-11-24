import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.game_env import SkystonesEnv
from algorithms.greedy_mc import MCAgent
from algorithms.trainer import Trainer
from algorithms.random_agent import RandomAgent

def test_random_opponent():
    print("Testing RandomAgent opponent...")
    env = SkystonesEnv(rows=3, cols=3)
    agent = MCAgent(env.action_space)
    opponent = RandomAgent(env.action_space)
    
    trainer = Trainer(
        agent=agent,
        env=env,
        model_path="models/test_agent.pkl",
        opponent=opponent,
        save_interval=10,
        log_interval=10
    )
    
    trainer.train(num_episodes=20)
    print("RandomAgent opponent test passed.")

def test_mc_opponent():
    print("\nTesting MCAgent opponent...")
    env = SkystonesEnv(rows=3, cols=3)
    agent = MCAgent(env.action_space)
    
    # Create a dummy opponent model
    opponent = MCAgent(env.action_space)
    opponent_path = "models/test_opponent.pkl.gz"
    opponent.save(opponent_path)
    
    # Load it back
    loaded_opponent = MCAgent.load(opponent_path, env.action_space)
    
    trainer = Trainer(
        agent=agent,
        env=env,
        model_path="models/test_agent_vs_mc.pkl",
        opponent=loaded_opponent,
        save_interval=10,
        log_interval=10
    )
    
    trainer.train(num_episodes=20)
    print("MCAgent opponent test passed.")
    
    # Clean up
    if os.path.exists(opponent_path):
        os.remove(opponent_path)

if __name__ == "__main__":
    test_random_opponent()
    test_mc_opponent()
