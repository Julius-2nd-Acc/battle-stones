import os
import numpy as np
import matplotlib.pyplot as plt
from algorithms.game_env import SkystonesEnv
from algorithms.ppo_agent import PPOAgent
from algorithms.mix_agent import MixAgent
from algorithms.stats_recorder import StatsRecorder

def train_ppo(
    num_episodes: int = 10000,
    gamma: float = 0.99,
    lr: float = 3e-4,
    rows: int = 3,
    cols: int = 3,
    model_path: str = "models/ppo_agent.pth",
    hidden_dim: int = 256,
    batch_size: int = 64,
    n_epochs: int = 10,
    update_frequency: int = 20  # Update after N episodes
):
    env = SkystonesEnv(render_mode=None, capture_reward=1.0, rows=rows, cols=cols)
    
    # Calculate input dimension
    input_dim = (rows * cols) + (rows * cols * 4) + (2 * env.max_slots * 4)
    
    agent = PPOAgent(
        action_space=env.action_space,
        input_dim=input_dim,
        gamma=gamma,
        lr=lr,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        n_epochs=n_epochs
    )
    
    # Opponent (MixAgent for robustness)
    opponent = MixAgent(action_space=env.action_space, epsilon=0.2)
    
    # Initialize stats recorder
    stats_recorder = StatsRecorder(save_dir="stats", model="ppo_agent")
    
    rewards_history = []
    loss_history = []
    
    print(f"Starting PPO training for {num_episodes} episodes...")
    print(f"Update frequency: every {update_frequency} episodes")
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        # Randomize player assignment (P0 or P1)
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
            
            # Store transition for PPO
            if current_player == agent_player_idx:
                agent.memory.store_memory(
                    agent.last_state,
                    action,
                    agent.last_log_prob,
                    agent.last_value,
                    reward,
                    done
                )
                total_reward += reward
            else:
                # Store negative reward for opponent's moves
                # This helps the agent learn that opponent gains are bad
                if agent.last_state is not None:
                    agent.memory.store_memory(
                        agent.last_state,
                        action,
                        agent.last_log_prob,
                        agent.last_value,
                        -reward,
                        done
                    )
                
            obs = next_obs
            
        rewards_history.append(total_reward)
        
        # Log episode stats
        stats_recorder.log_episode(
            episode=episode,
            reward=total_reward,
            winner=None,
            epsilon=0.0,
            steps=0
        )
        
        # Update agent periodically
        if episode % update_frequency == 0:
            loss = agent.update()
            loss_history.append(loss)
            
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
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
    plt.title("PPO Training Loss")
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("stats/ppo_loss.png")
    plt.close()
    print("Loss plot saved to stats/ppo_loss.png")

if __name__ == "__main__":
    train_ppo(num_episodes=5000, update_frequency=10)
