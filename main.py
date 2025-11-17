from obj.board import Board
from services.game_instance import GameInstance

import pickle
from algorithms.GreedyMC import EpsilonGreedyMCAgent
from algorithms.policy import make_inference_policy


def main():
    game_instance = GameInstance()
    game_instance.setup_game(col=5, row=5)

    # Try to load a trained Q-table and inject a policy into player 0 for inference.
    try:
        # compute action-space parameters from the current game setup
        rows = game_instance.board.rows
        cols = game_instance.board.cols
        max_slots = len(next(iter(game_instance.initial_slots.values())))
        n_actions = max_slots * rows * cols

        agent = EpsilonGreedyMCAgent(n_actions=n_actions)
        with open(r"c:\Users\jplk2\Documents\Battle_stones\q_table_smoke.pkl", "rb") as fh:
            q = pickle.load(fh)
            agent.load_Q_table(q)

        policy = make_inference_policy(agent, game_instance, player_idx=0, rows=rows, cols=cols, max_stones=max_slots)
        game_instance.players[0].set_policy(policy)
        try:
            game_instance.players[0].set_name("MonteCarlo")
        except Exception:
            game_instance.players[0].name = "MonteCarlo"
        print("Loaded trained policy into Player 0")
    except Exception:
        print("No trained policy found or failed to load; running with default/random players")

    game_instance.start_game()


if __name__ == "__main__":
    main()
