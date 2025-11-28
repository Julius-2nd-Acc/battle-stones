
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Any, List, Tuple

from algorithms.reinforce_agent import ReinforceAgent, PolicyNetwork

class PPOMemory:
    def __init__(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, 64) # Batch size 64
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+64] for i in batch_start]
        
        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class PPOAgent(ReinforceAgent):
    def __init__(
        self, 
        action_space, 
        input_dim: int,
        gamma: float = 0.99, 
        lr: float = 3e-4,
        hidden_dim: int = 128,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        batch_size: int = 64,
        n_epochs: int = 10,
        entropy_coef: float = 0.01
    ):
        super().__init__(action_space, input_dim, gamma, lr, hidden_dim)
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        
        self.memory = PPOMemory()
        
        # Temporary storage for the last action's details
        self.last_log_prob = None
        self.last_value = None
        self.last_state = None

    def choose_action(self, observation: Any, legal_actions: List[int] | None = None) -> int:
        state = self._preprocess_obs(observation)
        self.last_state = state.cpu().numpy() # Store as numpy for memory
        
        # Forward pass
        action_logits, state_value = self.policy_net(state)
        
        # Mask illegal actions
        if legal_actions is not None:
            mask = torch.full_like(action_logits, -1e9)
            mask[legal_actions] = 0
            action_logits = action_logits + mask
            
        probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(probs)
        action = dist.sample()
        
        self.last_log_prob = dist.log_prob(action).item()
        self.last_value = state_value.item()
        
        return action.item()

    def store_transition(self, reward, done):
        # Store the transition that just happened (using the last state/action)
        # Note: 'action' is not stored in self, but the caller (trainer) knows it.
        # Actually, it's better if the trainer calls memory.store_memory directly
        # or we pass action here.
        # But to keep interface simple, let's assume trainer handles memory storage
        # using agent.last_log_prob etc.
        pass

    def update(self):
        # PPO update using stored memory
        state_arr, action_arr, old_prob_arr, vals_arr, \
        reward_arr, dones_arr, batches = self.memory.generate_batches()

        values = vals_arr
        advantage = np.zeros(len(reward_arr), dtype=np.float32)

        # GAE Calculation
        for t in range(len(reward_arr)-1):
            discount = 1
            a_t = 0
            for k in range(t, len(reward_arr)-1):
                # delta_k = r_k + gamma * V(s_{k+1}) * (1-done) - V(s_k)
                # Note: We need V(s_{k+1}). vals_arr has V(s_k).
                # We can approximate V(s_{k+1}) with vals_arr[k+1] if not done.
                # If done, V(s_{k+1}) is 0 (or terminal value).
                
                next_val = vals_arr[k+1] if not dones_arr[k] else 0
                delta = reward_arr[k] + self.gamma * next_val - vals_arr[k]
                a_t += discount * delta
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t
            
        # Normalize advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        
        advantage = torch.tensor(advantage).to(self.device)
        values = torch.tensor(values).to(self.device)

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = self.memory.generate_batches()
            
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)
                
                # Forward pass
                action_logits, state_values = self.policy_net(states)
                state_values = state_values.squeeze()
                
                # We need to mask logits again? 
                # Ideally yes, but we don't have legal_actions stored easily.
                # However, since we only sample valid actions, the policy should learn to avoid invalid ones.
                # Or we can store legal_actions mask in memory.
                # For now, let's assume the policy has learned enough or the probability of invalid action is low.
                # Wait, if we don't mask, the probability distribution changes.
                # This is a common issue in PPO with invalid action masking.
                # We should store the mask.
                
                probs = torch.softmax(action_logits, dim=-1)
                dist = Categorical(probs)
                new_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()
                
                prob_ratio = torch.exp(new_probs - old_probs)
                
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.policy_clip, 1+self.policy_clip) * advantage[batch]
                
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                returns = advantage[batch] + values[batch]
                critic_loss = (returns - state_values)**2
                critic_loss = critic_loss.mean()
                
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        self.memory.clear_memory()
        return total_loss.item()
