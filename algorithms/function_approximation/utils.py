import numpy as np
import torch

def normalize_vectorize_obs(env, obs):
    '''
    Convert env observation into a single vector epresenting a state
    Addition compared to other implementations: normalize everything to range [0,1] for NN
    '''
    board_owner = obs['board_owner'].astype(np.float32).ravel() / 2.0

    board_type = obs['board_type'].astype(np.float32).ravel()
    board_type = (board_type + 1.0) / (env.num_types + 1.0)

    hand = obs['hand_types'].astype(np.float32).ravel()
    hand = (hand + 1.0) / (env.num_types + 1.0)

    to_move = np.array([float(obs['to_move'])], dtype=np.float32)

    # Concatenated state vector
    return np.concatenate([board_owner, board_type, hand, to_move], axis=0)

def obs_to_tensor(env, obs, device='cpu'):
    vectorized_state = normalize_vectorize_obs(env, obs)
    return torch.from_numpy(vectorized_state).float().to(device)