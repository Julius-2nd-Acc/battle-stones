from pathlib import Path
from algorithms.game_env import SkystonesEnv
from algorithms.agent_interface import Agent
from algorithms.stats_recorder import StatsRecorder
import random

class Trainer:
    def __init__(
            self,
            agent: Agent,
            env: SkystonesEnv,
            model_path: str,
            save_interval: int = 5000,
            log_interval: int = 1000
        ):
        self.agent = agent
        self.env = env
        self.model_path = Path(model_path)
        self.save_interval = save_interval
        self.log_interval = log_interval
        
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize Stats Recorder
        self.stats_recorder = StatsRecorder(save_dir="stats", model=self.model_path.stem)

    def train(self, num_episodes: int):
        print(f"Starting training for {num_episodes} episodes...")
        
        for episode_idx in range(1, num_episodes + 1):
            obs, info = self.env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            
            # For MC agent, we need to track the episode
            episode_data = [] 

            while not done:
                player_to_act = self.env.current_player_idx
                legal_actions = self.env.get_legal_actions(player_to_act)
                
                if player_to_act == 0:
                    # Agent controls Player 0
                    # Use policy_action_masked if available for exploration (epsilon-greedy)
                    if hasattr(self.agent, 'policy_action_masked'):
                        action = self.agent.policy_action_masked(obs, legal_actions)
                    else:
                        action = self.agent.choose_action(obs, legal_actions)
                else:
                    # Random opponent for Player 1
                    if legal_actions:
                        action = random.choice(legal_actions)
                    else:
                        action = self.env.action_space.sample()

                # DEBUG: make sure we never step with an illegal action according to our own mask
                if self.env.current_player_idx == 0:  # only for the agent player
                    assert action in legal_actions, f"Chosen action {action} not in legal_actions {legal_actions[:20]}"

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                '''
                if done:
                    print(
                        f"Episode {episode_idx} done. "
                        f"terminated={terminated}, truncated={truncated}, "
                        f"illegal_move={info.get('illegal_move', False)}, reward={reward}"
                    )
                '''
                
                # Track data for MC
                if hasattr(self.agent, 'get_state_key'):
                     state_key = self.agent.get_state_key(obs)
                     episode_data.append((state_key, action, reward))

                # Q-Learning Update (Online)
                if player_to_act == 0 and hasattr(self.agent, 'update'):
                    # we care about what P0 could do next, not P1
                    legal_next = self.env.get_legal_actions(player_idx=0) if not done else []
                    self.agent.update(obs, action, reward, next_obs, done, legal_next_actions=legal_next)

                total_reward += reward
                obs = next_obs
                steps += 1

            # MC Update (Offline / End of Episode)
            if hasattr(self.agent, 'update_from_episode'):
                self.agent.update_from_episode(episode_data)

            # Decay Epsilon if applicable
            if hasattr(self.agent, 'epsilon') and hasattr(self.agent, 'epsilon_decay'):
                 # This assumes the agent manages its own decay if we just touch the attribute, 
                 # or we can do it manually here if the agent exposes it.
                 # The previous scripts did it manually:
                 epsilon_min = getattr(self.agent, 'epsilon_min', 0.01)
                 decay = getattr(self.agent, 'epsilon_decay', 0.999)
                 self.agent.epsilon = max(epsilon_min, self.agent.epsilon * decay)

            # Log stats
            eps = getattr(self.agent, 'epsilon', 0.0)
            winner = info.get('winner')
            self.stats_recorder.log_episode(episode_idx, total_reward, winner, eps, steps)

            # Logging
            if episode_idx % self.log_interval == 0:
                print(f"[Episode {episode_idx}/{num_episodes}] total_reward={total_reward:.3f} epsilon={eps:.4f}")
                

            # Saving
            if episode_idx % self.save_interval == 0:
                self.save_agent()
                self.stats_recorder.save_plots()

        self.save_agent()
        self.stats_recorder.save_plots() # Final plot
        print(f"Training finished. Agent saved to: {self.model_path}")

    def save_agent(self):
        if hasattr(self.agent, 'save'):
            tmp_path = self.model_path.with_suffix(".tmp.pkl")
            self.agent.save(tmp_path)
            tmp_path.replace(self.model_path)
            print(f"Saved checkpoint to: {self.model_path}")
