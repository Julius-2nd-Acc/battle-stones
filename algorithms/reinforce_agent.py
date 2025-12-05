
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Any, List, Tuple

from algorithms.agent_interface import Agent
from services.compact_state import CompactStateBuilder

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (Policy)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (Value)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # Actor logits
        action_logits = self.actor(features)
        
        # Value estimate
        state_value = self.critic(features)
        
        return action_logits, state_value

class ReinforceAgent(Agent):
    def __init__(
        self, 
        action_space, 
        input_dim: int,
        gamma: float = 0.99, 
        lr: float = 1e-3,
        hidden_dim: int = 128,
        entropy_coef: float = 0.01
    ):
        self.action_space = action_space
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = PolicyNetwork(input_dim, action_space.n, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Storage for current episode
        self.saved_log_probs = []
        self.saved_entropies = []
        self.saved_values = []
        self.rewards = []
        
    def _preprocess_obs(self, obs: Any) -> torch.Tensor:
        """
        Convert observation dict to a flat tensor.
        Always uses absolute perspective (like DQN) - no perspective flipping.
        """
        ownership = obs["ownership"].flatten()
        board_stats = obs["board_stats"].flatten()
        hand_stats = obs["hand_stats"].flatten()
        
        # Concatenate all features
        flat_obs = np.concatenate([ownership, board_stats, hand_stats])
        return torch.FloatTensor(flat_obs).to(self.device)

    def choose_action(self, observation: Any, legal_actions: List[int] | None = None) -> int:
        state = self._preprocess_obs(observation)
        
        # Forward pass
        action_logits, state_value = self.policy_net(state)
        
        # Mask illegal actions
        if legal_actions is not None:
            mask = torch.full_like(action_logits, -1e9)
            mask[legal_actions] = 0
            action_logits = action_logits + mask
            
        probs = torch.softmax(action_logits, dim=-1)
        
        # Create distribution
        dist = Categorical(probs)
        
        # Sample action
        action = dist.sample()
        
        # Save log prob, entropy, and value for update
        self.saved_log_probs.append(dist.log_prob(action))
        self.saved_entropies.append(dist.entropy())
        self.saved_values.append(state_value)
        
        return action.item()
    
    def store_reward(self, reward: float):
        self.rewards.append(reward)
        
    def update(self):
        """
        Perform policy gradient update using collected trajectory.
        """
        R = 0
        returns = []
        
        # Calculate returns (G_t)
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        
        # REMOVED: Return normalization - it destroys the reward signal in two-player games
        # where absolute values matter for distinguishing wins from losses
            
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for log_prob, entropy, value, R in zip(self.saved_log_probs, self.saved_entropies, self.saved_values, returns):
            # CRITICAL FIX: Keep value as tensor to maintain gradients
            advantage = R - value.squeeze()
            
            # Policy loss: -log_prob * advantage (detach advantage for policy gradient)
            policy_losses.append(-log_prob * advantage.detach())
            
            # Value loss: MSE(value, R)
            value_losses.append(nn.functional.smooth_l1_loss(value.squeeze(), R))
            
            # Entropy bonus for exploration
            entropy_losses.append(-entropy)
            
        # Sum losses
        policy_loss = torch.stack(policy_losses).sum()
        value_loss = torch.stack(value_losses).sum()
        entropy_loss = torch.stack(entropy_losses).sum()
        
        loss = policy_loss + value_loss + self.entropy_coef * entropy_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        # Clear memory
        self.saved_log_probs = []
        self.saved_entropies = []
        self.saved_values = []
        self.rewards = []
        
        return loss.item()

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path))
