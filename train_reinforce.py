
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
    lr: float = 3e-4,  # Reduced from 1e-3 for stability
    rows: int = 3,
    cols: int = 3,
    model_path: str = "models/reinforce_agent.pth",
    hidden_dim: int = 128,
    entropy_coef: float = 0.01
):
    env = SkystonesEnv(render_mode=None, capture_reward=2.0, rows=rows, cols=cols)
    
    # Calculate input dimension
    # ownership (rows*cols) + board_stats (rows*cols*4) + hand_stats (2*max_slots*4)
    input_dim = (rows * cols) + (rows * cols * 4) + (2 * env.max_slots * 4)
    
    agent = ReinforceAgent(
        action_space=env.action_space,
        input_dim=input_dim,
        gamma=gamma,
        lr=lr,
        hidden_dim=hidden_dim,
        entropy_coef=entropy_coef
    )
    
    # Load existing model if available
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        try:
            agent.load(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Starting training from scratch.")
    
    '''
    opponent = ReinforceAgent(
        action_space=env.action_space,
        input_dim=input_dim,
        gamma=gamma,
        lr=lr,
        hidden_dim=hidden_dim
    )
    opponent.load(model_path)
    '''

    opponent = MixAgent(env.action_space, epsilon=0.3)
    # Initialize stats recorder
    stats_recorder = StatsRecorder(save_dir="stats", model="reinforce_agent")
    
    rewards_history = []
    loss_history = []
    
    print(f"Starting REINFORCE training for {num_episodes} episodes...")
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0
        if episode % 2000 == 0:
            stats_recorder.save_plots()
        
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
                
            obs = next_obs
            
            if current_player == agent_player_idx:
                total_reward += reward

        # After the while loop, before agent.update()
        last_actor = env.current_player_idx  # player who just moved

        # 'reward' here is from the last step in the loop
        if last_actor != agent_player_idx:
            # Opponent made the final move and got 'reward' from their perspective.
            # From the agent's perspective it's -reward.
            if agent.rewards:
                agent.rewards[-1] += -reward  # add terminal outcome to last agent reward
            else:
                # Edge case: agent never moved (very short game)
                agent.store_reward(-reward)


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
    train_reinforce(num_episodes=100000, hidden_dim=2*512, rows=3, cols=3, model_path="models/reinforce_agent_3x3.pth") 
