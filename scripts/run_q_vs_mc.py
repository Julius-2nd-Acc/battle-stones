"""Run a single game between a Q-learning agent (player 0) and the Monte Carlo agent (player 1).

This script expects previously-saved Q-tables in the repository root. It will try the
following filenames (in order):
 - q_table_qlearn.pkl  (Q-learning Q)
 - q_table_smoke.pkl   (MC agent Q saved by MC trainer)

Either or both may be missing; missing policies will act randomly.
"""
import os
import pickle

from services.game_instance import GameInstance
from algorithms.QLearning import EpsilonGreedyQLearningAgent
from algorithms.GreedyMC import EpsilonGreedyMCAgent
from algorithms.policy import make_inference_policy


def try_load_q(agent, candidates):
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "rb") as fh:
                    q = pickle.load(fh)
                agent.load_Q_table(q)
                print(f"Loaded Q-table from: {p}")
                return True
            except Exception as e:
                print(f"Failed to load Q from {p}: {e}")
    return False


def main():
    game = GameInstance()
    game.setup_game()

    rows = game.board.rows
    cols = game.board.cols
    max_slots = len(next(iter(game.initial_slots.values())))
    n_actions = max_slots * rows * cols

    # Q-learning agent -> player 0
    q_agent = EpsilonGreedyQLearningAgent(n_actions=n_actions)
    q_candidates = [
        os.path.join(os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".."), "q_table_qlearn.pkl"),
        os.path.join(os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".."), "q_table_smoke.pkl"),
        os.path.join(os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".."), "q_table.pkl"),
    ]
    q_loaded = try_load_q(q_agent, q_candidates)

    # MC agent -> player 1
    mc_agent = EpsilonGreedyMCAgent(n_actions=n_actions)
    mc_candidates = [
        os.path.join(os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".."), "q_table_smoke.pkl"),
        os.path.join(os.path.abspath(os.path.dirname(__file__) + os.path.sep + ".."), "q_table.pkl"),
    ]
    mc_loaded = try_load_q(mc_agent, mc_candidates)

    # Create inference policies (they validate/fallback internally)
    if q_loaded:
        q_policy = make_inference_policy(q_agent, game, player_idx=0, rows=rows, cols=cols, max_stones=max_slots)
        game.players[0].set_policy(q_policy)
        # label the player to reflect the algorithm
        try:
            game.players[0].set_name("Q-Learning")
        except Exception:
            game.players[0].name = "Q-Learning"
    else:
        print("No Q-table loaded for Q-agent; player 0 will act randomly.")
        try:
            game.players[0].set_name("Random (Q)")
        except Exception:
            game.players[0].name = "Random (Q)"

    if mc_loaded:
        mc_policy = make_inference_policy(mc_agent, game, player_idx=1, rows=rows, cols=cols, max_stones=max_slots)
        game.players[1].set_policy(mc_policy)
        try:
            game.players[1].set_name("MonteCarlo")
        except Exception:
            game.players[1].name = "MonteCarlo"
    else:
        print("No Q-table loaded for MC agent; player 1 will act randomly.")
        try:
            game.players[1].set_name("Random (MC)")
        except Exception:
            game.players[1].name = "Random (MC)"

    print("Starting single Q vs MC game...")
    game.start_game()

    # After the game, print final counts
    counts = game.board.get_current_stone_count()
    print("Final stone counts:")
    for p in game.players:
        print(f"  {p.name}: {counts.get(p, 0)}")


if __name__ == '__main__':
    main()
