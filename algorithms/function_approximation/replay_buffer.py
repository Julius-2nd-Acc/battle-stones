from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done, legal_next_actions):
        self.buffer.append((obs, action, reward, next_obs, done, legal_next_actions))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones, legal_nexts = zip(*batch)
        return (
            list(obs),
            list(actions),
            list(rewards),
            list(next_obs),
            list(dones),
            list(legal_nexts),
        )