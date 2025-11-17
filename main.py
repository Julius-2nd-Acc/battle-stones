import random
from pathlib import Path

from algorithms.greedy_mc import MCAgent
from algorithms.q_learning import QLearningAgent
from algorithms.game_env import SkystonesEnv


MC_MODEL_PATH = Path("models/mc_agent_skystones.pkl")
Q_MODEL_PATH = Path("models/q_agent_skystones.pkl")


def sample_random_action(env: SkystonesEnv, player_idx: int | None = None) -> int:
    """
    Sample a random *legal* action for the given player in the env.

    Uses env.get_legal_actions if available. If no legal actions exist, falls
    back to any action (env will handle terminal / no-move situations).
    """
    if player_idx is None:
        player_idx = env.current_player_idx

    # If you implemented get_legal_actions(player_idx=None), prefer that:
    if hasattr(env, "get_legal_actions"):
        legal_actions = env.get_legal_actions(player_idx=player_idx)
        if legal_actions:
            return random.choice(legal_actions)
        return env.action_space.sample()

    # Fallback to manual enumeration (old behavior)
    legal_actions = []
    cells = env.rows * env.cols

    for slot in range(env.max_slots):
        stone = env._get_stone_for_slot(player_idx, slot)
        if stone is None:
            continue  # no stone in this slot

        for r in range(env.rows):
            for c in range(env.cols):
                if env.game.board.isValidMove((r, c)):
                    a = slot * cells + (r * env.cols + c)
                    legal_actions.append(a)

    if not legal_actions:
        return env.action_space.sample()

    return random.choice(legal_actions)


def _choose_action_for_player(agent, env: SkystonesEnv, obs, player_idx: int) -> int:
    """
    Helper: choose an action for the given player using the right API.

    - If agent has greedy_action_masked and env.get_legal_actions → use masked greedy.
    - Else if agent has greedy_action → use that.
    - Else → random legal action.
    """
    if agent is None:
        return sample_random_action(env, player_idx=player_idx)

    legal_actions = None
    if hasattr(env, "get_legal_actions"):
        legal_actions = env.get_legal_actions(player_idx=player_idx)

    # Prefer masked greedy if both agent and env support it
    if legal_actions is not None and hasattr(agent, "greedy_action_masked"):
        if not legal_actions:
            return env.action_space.sample()
        return agent.greedy_action_masked(obs, legal_actions)

    # Fallback to unmasked greedy (might propose illegal moves)
    if hasattr(agent, "greedy_action"):
        return agent.greedy_action(obs)

    # Last resort: random legal
    return sample_random_action(env, player_idx=player_idx)


def _describe_action(env: SkystonesEnv, action: int, player_idx: int, controller_label: str) -> str:
    """
    Describe an action using the *current* game state (before step).
    """
    slot, row, col = env._decode_action(action)
    stone = env._get_stone_for_slot(player_idx, slot)
    

    if stone is not None and hasattr(stone, "get_representation"):
        stone_repr = stone.get_representation()
    elif stone is not None and hasattr(stone, "name"):
        stone_repr = stone.name
    else:
        stone_repr = "None (empty slot)"
    
    

    return (
        f"Player {player_idx} [{controller_label}] plays "
        f"slot {slot}, stone {stone_repr}, to (row={row}, col={col})"
    )


def play_one_episode(env, agent_p0=None, agent_p1=None, render=False):
    """
    Play a single game.

    - agent_p0: agent controlling Player 0 (QLearningAgent or MCAgent), or None for random
    - agent_p1: agent controlling Player 1, or None for random

    Behavior on illegal move:
      - The agent that made the illegal move keeps the turn.
      - The illegal move does NOT end the game, and its reward is ignored.

    When render=True, prints each move with a description and then shows the board.

    Returns:
        winner: 0 (Player 0), 1 (Player 1), or None for draw
        total_reward_p0: sum of rewards from Player 0 perspective over the episode
    """
    obs, info = env.reset()
    done = False
    total_reward_p0 = 0.0
    move_number = 1

    while not done:
        current_player = env.current_player_idx
        

        if current_player == 0:
            controller = agent_p0
            controller_label = type(agent_p0).__name__ if agent_p0 is not None else "Random"
            action = _choose_action_for_player(agent_p0, env, obs, player_idx=0)
        else:
            controller = agent_p1
            controller_label = type(agent_p1).__name__ if agent_p1 is not None else "Random"
            action = _choose_action_for_player(agent_p1, env, obs, player_idx=1)

        # Build description BEFORE step mutates the state
        if render:
            move_desc = _describe_action(env, action, current_player, controller_label)
    

        next_obs, reward, terminated, truncated, info = env.step(action)

        if info.get("illegal_move", False):
            if render:
                print(f"\nMove {move_number}:")
                print(move_desc)
                print(f"→ ILLEGAL move by Player {current_player}, retrying...")
                env.render()
            obs = next_obs
            move_number += 1
            continue

        if render:
            print(f"\nMove {move_number}:")
            print(move_desc)
            env.render()
    

        obs = next_obs
        total_reward_p0 += reward
        done = terminated or truncated or env.game.board.get_total_stone_count() == 8
        move_number += 1


    # Determine winner from final board state
    owner_counts = env.game.board.get_current_stone_count()
    if not owner_counts:
        winner = None  # no stones on board
    else:
        max_count = max(owner_counts.values())
        winners = [p for p, c in owner_counts.items() if c == max_count]
        if len(winners) != 1:
            winner = None
        else:
            winner = 0 if winners[0] == env.game.players[0] else 1

    return winner, total_reward_p0


def run_mc_vs_random(num_episodes=50, render=False):
    env = SkystonesEnv(render_mode="human" if render else None, capture_reward=1.0)
    mc_agent = MCAgent.load(MC_MODEL_PATH, env.action_space)

    wins_p0 = 0
    wins_p1 = 0
    draws = 0

    for ep in range(1, num_episodes + 1):
        winner, total_reward_p0 = play_one_episode(
            env, agent_p0=mc_agent, agent_p1=None, render=render
        )

        if winner == 0:
            wins_p0 += 1
        elif winner == 1:
            wins_p1 += 1
        else:
            draws += 1

        print(
            f"[MC vs Random] Episode {ep}/{num_episodes} "
            f"winner={winner}  total_reward_p0={total_reward_p0:.3f}"
        )

    env.close()
    print(
        f"MC vs Random summary: P0(MC) wins={wins_p0}, P1(Random) wins={wins_p1}, draws={draws}"
    )


def run_q_vs_random(num_episodes=50, render=False):
    env = SkystonesEnv(render_mode="human" if render else None, capture_reward=1.0)
    q_agent = QLearningAgent.load(Q_MODEL_PATH, env.action_space)

    wins_p0 = 0
    wins_p1 = 0
    draws = 0

    for ep in range(1, num_episodes + 1):
        winner, total_reward_p0 = play_one_episode(
            env, agent_p0=q_agent, agent_p1=None, render=render
        )

        if winner == 0:
            wins_p0 += 1
        elif winner == 1:
            wins_p1 += 1
        else:
            draws += 1

        print(
            f"[Q vs Random] Episode {ep}/{num_episodes} "
            f"winner={winner}  total_reward_p0={total_reward_p0:.3f}"
        )

    env.close()
    print(
        f"Q vs Random summary: P0(Q) wins={wins_p0}, P1(Random) wins={wins_p1}, draws={draws}"
    )


def run_mc_vs_q(num_episodes=50, render=False):
    env = SkystonesEnv(render_mode="human" if render else None, capture_reward=1.0)
    mc_agent = MCAgent.load(MC_MODEL_PATH, env.action_space)
    q_agent = QLearningAgent.load(Q_MODEL_PATH, env.action_space)

    wins_p0 = 0
    wins_p1 = 0
    draws = 0

    for ep in range(1, num_episodes + 1):
        winner, total_reward_p0 = play_one_episode(
            env, agent_p0=q_agent, agent_p1=mc_agent, render=render
        )

        if winner == 0:
            wins_p0 += 1
        elif winner == 1:
            wins_p1 += 1
        else:
            draws += 1

        print(
            f"[Q(P0) vs MC(P1)] Episode {ep}/{num_episodes} "
            f"winner={winner}  total_reward_p0={total_reward_p0:.3f}"
        )

    env.close()
    print(
        f"Q(P0) vs MC(P1) summary: "
        f"P0(Q) wins={wins_p0}, P1(MC) wins={wins_p1}, draws={draws}"
    )


if __name__ == "__main__":
    # Choose which matchup you want to run.
    # Set render=True to print the board and move descriptions.
    run_q_vs_random(num_episodes=5, render=True)
    # run_mc_vs_random(num_episodes=5, render=True)
    # run_mc_vs_q(num_episodes=5, render=True)
