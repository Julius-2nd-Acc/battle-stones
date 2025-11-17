"""Small training harness to exercise EpsilonGreedyMCAgent with GameEnv."""
from algorithms.GreedyMC import EpsilonGreedyMCAgent
from algorithms.game_env import GameEnv
import pickle
import os


def main():
    env = GameEnv(rows=3, cols=3, max_stones=4, train_player_idx=0)
    agent = EpsilonGreedyMCAgent(n_actions=env.n_actions, gamma=1.0, epsilon_start=0.3, epsilon_min=0.05, epsilon_decay=0.995)

    # Attempt to load an existing Q-table so training continues from previous state
    out = os.path.join(os.path.dirname(__file__), "..", "q_table_mc.pkl")
    if os.path.exists(out):
        try:
            with open(out, "rb") as fh:
                loaded_q = pickle.load(fh)
            if hasattr(agent, "set_Q_table"):
                agent.set_Q_table(loaded_q)
                print("Loaded Q-table into agent via set_Q_table(). Continuing training...")
            elif hasattr(agent, "Q"):
                agent.Q = loaded_q
                print("Loaded Q-table into agent.Q. Continuing training...")
            elif hasattr(agent, "Q_table"):
                agent.Q_table = loaded_q
                print("Loaded Q-table into agent.Q_table. Continuing training...")
            else:
                print("Loaded Q-table from disk but could not assign it to the agent (no known setter/attribute).")
        except Exception as e:
            print("Failed to load Q-table:", e)
    else:
        print("No existing Q-table found. Training from scratch...")

    print("Starting short training run (smoke test)...")
    agent.train(env, num_episodes=1000000, max_steps_per_episode=1000, verbose=True)

    q = agent.get_Q_table()
    with open(out, "wb") as fh:
        pickle.dump(q, fh)

    print("Training finished, Q saved to", out)


if __name__ == '__main__':
    main()
