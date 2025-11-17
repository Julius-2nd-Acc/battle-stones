"""Smoke test for inference policy wrapper: load Q, inject into player, and run one game move."""
import pickle
from algorithms.GreedyMC import EpsilonGreedyMCAgent
from algorithms.policy import make_inference_policy
from services.game_instance import GameInstance


def main():
    # create game and setup
    game = GameInstance()
    game.setup_game()

    # create agent and load Q
    agent = EpsilonGreedyMCAgent(n_actions=4 * 3 * 3)
    try:
        with open("c:\\Users\\jplk2\\Documents\\Battle_stones\\q_table_smoke.pkl", "rb") as fh:
            q = pickle.load(fh)
            agent.load_Q_table(q)
    except FileNotFoundError:
        print("No Q table found; run scripts/train_mc.py first")
        return

    # make policy for player 0 and inject
    policy = make_inference_policy(agent, game, player_idx=0, rows=3, cols=3, max_stones=4)
    player0 = game.players[0]
    player0.set_policy(policy)

    # get state (not used by our policy) and choose an action
    state = ""
    action = player0.choose_action(state)
    print("Chosen action:", action)

    # decode and perform the action using existing game method
    # reuse game_env's decoding logic: stone_slot, r, c
    slot = action // (3 * 3)
    cell_idx = action % (3 * 3)
    r = cell_idx // 3
    c = cell_idx % 3
    print(f"Decoded -> slot={slot}, r={r}, c={c}")

    # attempt to place stone
    slot_name = [s.name for s in player0.stones]
    if slot < len(slot_name):
        name = slot_name[slot]
        # find the stone object with that name
        chosen = None
        for s in player0.stones:
            if s.name == name:
                chosen = s
                break
        if chosen is not None and game.board.isValidMove((r, c)):
            game.place_stone(player0, (r, c), chosen)
            print("Move applied")
        else:
            print("Move invalid at apply time")
    else:
        print("Slot out of range")


if __name__ == '__main__':
    main()
