
import os
import numpy as np
import matplotlib.pyplot as plt
from algorithms.game_env import SkystonesEnv
from algorithms.reinforce_agent import ReinforceAgent
from algorithms.mix_agent import MixAgent
from algorithms.stats_recorder import StatsRecorder

def train_reinforce(
    num_episodes: int = 5000,
    gamma: float = 0.99,
    lr: float = 1e-3,
    rows: int = 3,
    cols: int = 3,
    model_path: str = "models/reinforce_agent.pth",
    hidden_dim: int = 128
):
    env = SkystonesEnv(render_mode=None, capture_reward=1.0, rows=rows, cols=cols)
    
    # Calculate input dimension
    # ownership (rows*cols) + board_stats (rows*cols*4) + hand_stats (2*max_slots*4)
    input_dim = (rows * cols) + (rows * cols * 4) + (2 * env.max_slots * 4)
    
    agent = ReinforceAgent(
        action_space=env.action_space,
        input_dim=input_dim,
        gamma=gamma,
        lr=lr,
        hidden_dim=hidden_dim
    )
    
    # Opponent (MixAgent for robustness)
    opponent = MixAgent(action_space=env.action_space, epsilon=0.4)
    
    # Initialize stats recorder
    stats_recorder = StatsRecorder(save_dir="stats", model="reinforce_agent")
    
    rewards_history = []
    loss_history = []
    
    print(f"Starting REINFORCE training for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        # Randomize player assignment (P0 or P1)
        # 0 = Agent is P0, 1 = Agent is P1
        agent_player_idx = np.random.randint(0, 2)
        
        while not done:
            current_player = env.current_player_idx
            legal_actions = env.get_legal_actions(current_player)
            
            if current_player == agent_player_idx:
                # Agent's turn
                action = agent.choose_action(obs, legal_actions)
            else:
                # Opponent's turn
                action = opponent.choose_action(obs, legal_actions)
                
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if current_player == agent_player_idx:
                agent.store_reward(reward)
            else:
                agent.store_reward(-reward) 
                
            obs = next_obs
            
            if current_player == agent_player_idx:
                total_reward += reward
                
        # Update agent
        loss = agent.update()
        loss_history.append(loss)
        rewards_history.append(total_reward)
        
        # Log episode stats
        stats_recorder.log_episode(
            episode=episode,
            reward=total_reward,
            winner=None,  # We don't track winner explicitly in REINFORCE
            epsilon=0.0,  # REINFORCE doesn't use epsilon
            steps=0  # Could track steps if needed
        )
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
            
    # Save model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save plots using StatsRecorder
    stats_recorder.save_plots()
    
    # Also save loss plot separately
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("stats/reinforce_loss.png")
    plt.close()
    print("Loss plot saved to stats/reinforce_loss.png")

if __name__ == "__main__":
    train_reinforce(num_episodes=100000, hidden_dim=256) 
