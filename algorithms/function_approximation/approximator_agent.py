from algorithms.agent_interface import Agent
from algorithms.function_approximation.utils import normalize_vectorize_obs, obs_to_tensor
from algorithms.function_approximation.replay_buffer import ReplayBuffer
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FunctionApproximationAgent(Agent):
    def __init__(
        self,
        env,
        model_cls: nn.Module,
        model_kwargs=None,
        hidden_dim=128,
        gamma=0.8,
        lr=1e-3,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.999,
        device=None,
        replay_capacity: int = 50_000,
        batch_size: int = 64,
        warmup_steps: int = 1_000,
        updates_per_step: int = 1
    ):
        self.env = env
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        obs, _ = env.reset()
        obs_vec = normalize_vectorize_obs(env, obs)
        obs_dim = obs_vec.shape[0]
        n_actions = env.action_space.n
        self.n_actions = n_actions

        if model_kwargs is None:
            model_kwargs = {}

        self.model = model_cls(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            **model_kwargs
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.steps_done = 0

        self.replay_buffer = ReplayBuffer(replay_capacity)
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.updates_per_step = updates_per_step
        self.training_steps = 0

    def policy_action_masked(self, observation, legal_actions: list[int]) -> int:
        """
        ε-greedy policy restricted to legal_actions.
        Guaranteed to return an element of legal_actions.
        """
        if not legal_actions:
            # no legal actions -> let env handle terminal/no-move case
            return self.env.action_space.sample()

        # Exploration: uniform over legal actions
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(legal_actions))

        # Exploitation: greedy over *legal* actions only
        self.model.eval()
        with torch.no_grad():
            obs_t = obs_to_tensor(self.env, observation, self.device).unsqueeze(0) 
            q_vals = self.model(obs_t)[0].cpu().numpy()                            

        # Masked argmax
        best_a = max(legal_actions, key=lambda a: q_vals[a])
        return int(best_a)


    def choose_action(self, observation, legal_actions = None):
        if legal_actions is None and hasattr(self.env, 'get_legal_actions'):
            legal_actions = self.env.get_legal_actions()
        if legal_actions is None:
            legal_actions = list(range(self.n_actions))
        
        return self.policy_action_masked(observation, legal_actions)
    
    def greedy_action_masked(self, observation, legal_actions):
        if not legal_actions:
            return self.env.action_space.sample()
        
        self.model.eval()
        with torch.no_grad():
            obs_t = obs_to_tensor(self.env, observation, self.device).unsqueeze(0)
            q_vals = self.model(obs_t)[0]

        q_legal = q_vals[legal_actions]
        best_idx = int(torch.argmax(q_legal).item())
        return int(legal_actions[best_idx])
    
    def greedy_action(self, observation):
        self.model.eval()
        with torch.no_grad():
            obs_t = obs_to_tensor(self.env, observation, self.device).unsqueeze(0)
            q_vals = self.model(obs_t)[0]
        return int(torch.argmax(q_vals).item())



    def update(
    self,
    obs,
    action: int,
    reward: float,
    next_obs,
    done: bool,
    legal_next_actions: list[int] | None = None,
):
        if legal_next_actions is None:
            if not done and hasattr(self.env, "get_legal_actions"):
                legal_next_actions = self.env.get_legal_actions(player_idx=0)
            else:
                legal_next_actions = []

        self.replay_buffer.add(obs, action, reward, next_obs, done, legal_next_actions)

        # Collect data for the buffer until warmup is over (warmup is usually not smaller than batch size)
        if len(self.replay_buffer) < max(self.batch_size, self.warmup_steps):
            return 0.0

        total_loss = 0.0

        # Take updates_per_step batches from the buffer
        for _ in range(self.updates_per_step):
            (
                obs_batch,
                act_batch,
                rew_batch,
                next_obs_batch,
                done_batch,
                legal_next_batch,
            ) = self.replay_buffer.sample(self.batch_size)

            # Transform the stuff in the batch to tensors
            obs_vecs = np.stack([normalize_vectorize_obs(self.env, o) for o in obs_batch], axis=0)
            next_obs_vecs = np.stack([normalize_vectorize_obs(self.env, o) for o in next_obs_batch], axis=0)

            obs_t = torch.from_numpy(obs_vecs).float().to(self.device)     
            next_obs_t = torch.from_numpy(next_obs_vecs).float().to(self.device)
            actions_t = torch.tensor(act_batch, dtype=torch.long, device=self.device)      
            rewards_t = torch.tensor(rew_batch, dtype=torch.float32, device=self.device)  
            dones_t = torch.tensor(done_batch, dtype=torch.float32, device=self.device)  

            # Use incomprehensible PyTorch indexing to extract the action value of the action taken in each state
            self.model.train()
            q_values = self.model(obs_t)                                    
            q_sa = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)     

            # Q Learning update
            with torch.no_grad():
                q_next_all = self.model(next_obs_t)                     

                max_q_next_list = []
                for i in range(self.batch_size):
                    legal_next = legal_next_batch[i]
                    if legal_next:
                        q_next_legal = q_next_all[i, legal_next]   
                        max_q_next_list.append(q_next_legal.max())
                    else:
                        max_q_next_list.append(torch.tensor(0.0, device=self.device))

                max_q_next_t = torch.stack(max_q_next_list)                  # [B]

                target_t = rewards_t + self.gamma * (1.0 - dones_t) * max_q_next_t

            loss = F.mse_loss(q_sa, target_t)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        self.training_steps += 1
        return total_loss / self.updates_per_step

    def save(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
            },
            filepath,
        )
        print(f'Model checkpoint saved to {filepath}')

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        env, 
        model_cls,
        hidden_dim,
        model_kwargs=None,
        **agent_kwargs
    ):
        filepath = Path(filepath)
        agent = cls(env, model_cls=model_cls, model_kwargs=model_kwargs, hidden_dim=hidden_dim, **agent_kwargs)
        checkpoint = torch.load(filepath, map_location=agent.device)

        agent.model.load_state_dict(checkpoint['model_state'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        agent.epsilon = checkpoint.get('epsilon', agent.epsilon)
        agent.epsilon_min = checkpoint.get('epsilon_min', agent.epsilon_min)
        agent.epsilon_decay = checkpoint.get('epsilon_decay', agent.epsilon_decay)
        print(f'Model checkpoint loaded from {filepath}')
        return agent