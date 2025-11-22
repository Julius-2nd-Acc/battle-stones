import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any

class StatsRecorder:
    def __init__(self, save_dir: str = "stats", filename: str = "training_stats.csv", model = "ai"):
        self.save_dir = save_dir
        self.filename = filename
        self.model = model
        self.filepath = os.path.join(save_dir, filename)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Initialize CSV with headers if it doesn't exist
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward", "winner", "epsilon", "steps"])
        
        self.rewards: List[float] = []
        self.win_rates: List[float] = []
        self.episodes: List[int] = []

    def log_episode(self, episode: int, reward: float, winner: str | None, epsilon: float, steps: int):
        """Log a single episode's stats to CSV and memory."""
        
        # Append to CSV
        #with open(self.filepath, mode='a', newline='') as f:
        #    writer = csv.writer(f)
        #    writer.writerow([episode, reward, winner, epsilon, steps])
            
        self.rewards.append(reward)
        self.episodes.append(episode)

    def save_plots(self):
        """Generate and save training plots."""
        if not self.rewards:
            return
            
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Rolling Average Reward
        plt.subplot(1, 2, 1)
        window_size = min(100, len(self.rewards))
        if window_size > 0:
            rolling_mean = np.convolve(self.rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(self.episodes[window_size-1:], rolling_mean, label=f'Rolling Avg ({window_size})')
        plt.plot(self.episodes, self.rewards, alpha=0.3, label='Raw Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Reward over Time')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Win Rate (approximate based on positive rewards)
        # Assuming reward > 0 means win, < 0 means loss, 0 means draw/incomplete
        plt.subplot(1, 2, 2)
        wins = [1 if r > 0 else 0 for r in self.rewards]
        if window_size > 0:
            rolling_win_rate = np.convolve(wins, np.ones(window_size)/window_size, mode='valid')
            plt.plot(self.episodes[window_size-1:], rolling_win_rate, color='green')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate')
        plt.title(f'Rolling Win Rate (Window {window_size})')
        plt.grid(True)
        
        plot_path = os.path.join(self.save_dir, f"{self.model}_training_plots.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Plots saved to {plot_path}")
