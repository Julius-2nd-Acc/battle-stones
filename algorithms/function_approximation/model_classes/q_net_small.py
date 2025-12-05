import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallQNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_actions)

    def forward(self, x: torch.Tensor):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        q = self.out(x)
        return q