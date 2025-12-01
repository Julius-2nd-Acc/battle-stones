from services.game_controller import GameController

def test_reinforce_in_controller():
    gc = GameController()
    
    print("Creating game with REINFORCE agent vs Random...")
    # Use 'reinforce' for player 0, 'random' for player 1
    game_id = gc.create_game_with_players(rows=3, cols=3, player0="reinforce", player1="random", autoplay=True)
    
    print(f"Game created with ID: {game_id}")
    print("Game should be running in autoplay mode...")
    print("Check the game state to see REINFORCE agent in action!")
    
    # Get initial state
    state = gc.get_state(game_id)
    print(f"Game started: {state['started']}")
    print(f"Agents: {state['agents']}")

if __name__ == "__main__":
    test_reinforce_in_controller()
