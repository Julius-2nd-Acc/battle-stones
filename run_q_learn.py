# scripts/run_q_agent.py
import pickle
from algorithms.QLearning import EpsilonGreedyQLearningAgent
from algorithms.policy import make_inference_policy
from services.game_instance import GameInstance

# create game instance and setup
game = GameInstance()
game.setup_game()

# build agent with same n_actions structure as used for training
rows, cols = game.board.rows, game.board.cols
max_slots = len(next(iter(game.initial_slots.values())))
n_actions = max_slots * rows * cols
agent = EpsilonGreedyQLearningAgent(n_actions=n_actions)

# load Q-table
with open(r"c:\Users\jplk2\Documents\Battle_stones\q_table_qlearn.pkl", "rb") as fh:
    qdict = pickle.load(fh)
agent.load_Q_table(qdict)

# make policy and inject to player 0
policy = make_inference_policy(agent, game, player_idx=0, rows=rows, cols=cols, max_stones=max_slots)
game.players[0].set_policy(policy)
try:
    game.players[0].set_name("Q-Learning")
except Exception:
    game.players[0].name = "Q-Learning"

# run the game (same as running main.py)
game.start_game()