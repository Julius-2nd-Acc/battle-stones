
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
        hidden_dim: int = 128
    ):
        self.action_space = action_space
        self.gamma = gamma
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = PolicyNetwork(input_dim, action_space.n, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Storage for current episode
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
        
    def _preprocess_obs(self, obs: Any) -> torch.Tensor:
        """
        Convert observation dict to a flat tensor.
        """
        ownership = obs["ownership"].flatten()

        to_move = obs["to_move"]
        if to_move == 1:
            # Swap 1 and 2
            ownership = np.where(ownership == 1, 2, np.where(ownership == 2, 1, ownership))
            
        board_stats = obs["board_stats"].flatten()
        hand_stats = obs["hand_stats"].flatten()
        
        # Concatenate
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
        
        # Save log prob and value for update
        self.saved_log_probs.append(dist.log_prob(action))
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
            
        returns = torch.tensor(returns).to(self.device)
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
        policy_losses = []
        value_losses = []
        
        for log_prob, value, R in zip(self.saved_log_probs, self.saved_values, returns):
            advantage = R - value.item()
            
            # Policy loss: -log_prob * advantage
            policy_losses.append(-log_prob * advantage)
            
            # Value loss: MSE(value, R)
            # We use smooth_l1_loss (Huber loss) which is often more stable
            value_losses.append(nn.functional.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))
            
        # Sum losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear memory
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
        
        return loss.item()

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path))
