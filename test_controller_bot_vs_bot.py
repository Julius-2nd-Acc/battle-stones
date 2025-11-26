
import time
from services.game_controller import GameController

def test_controller_bot_vs_bot():
    gc = GameController()
    
    print("Creating Bot vs Bot game (Q vs Q)...")
    # Use 'qlearn' for both players
    game_id = gc.create_game_with_players(rows=3, cols=3, player0="qlearn", player1="qlearn", autoplay=True)
    
    print(f"Game created with ID: {game_id}")
    
    # Monitor game
    max_wait = 30
    start_time = time.time()
    
    while True:
        state = gc.get_state(game_id)
        
        # Print board status
        stones_on_board = 0
        for row in state["board"]:
            for cell in row:
                if cell is not None:
                    stones_on_board += 1
        
        print(f"Stones on board: {stones_on_board}")
        
        if not state["started"]:
            print("Game finished!")
            print(f"Winner: {state['winner']}")
            break
            
        if time.time() - start_time > max_wait:
            print("Timeout waiting for game to finish.")
            break
            
        time.sleep(1.0)
        
    # Check stats
    print("Stats:", gc.stats)

if __name__ == "__main__":
    test_controller_bot_vs_bot()
