from obj.board import Board
from obj.player import Player, PlayerType
from obj.stone import Stone
import random


class GameInstance:
    def __init__(self):
        
        self.players = []

        self.initial_slots = {}
    
    def add_player(self, player: Player):
        self.players.append(player)
        
    def setup_game(self, col=3, row=3):
        self.started = True
        self.setup_board(col=col, rows=row)
        self.setup_players()
        pass
    
    def setup_board(self,col=3,rows=3):
        board = Board(cols=col,rows=rows)
        self.board = board

    def setup_players(self):
        self.add_player(Player("B", PlayerType.HUMAN))
        self.add_player(Player("R", PlayerType.AI))
        for player in self.players:
            self.generate_stones_for_player(player)
        # record initial stone slot ordering and pad to uniform length
        max_slots = max((len(p.stones) for p in self.players), default=0)
        for i, p in enumerate(self.players):
            names = [s.name for s in p.stones]
            if len(names) < max_slots:
                names += [None] * (max_slots - len(names))
            self.initial_slots[i] = names
    
    def generate_stones_for_player(self, player:Player):
        # Use a deterministic set of four stones for all players so training and
        # inference are reproducible. Values chosen arbitrarily.
        fixed_stats = [
            (2, 1, 1, 0),  # Stone A: n=2,s=1,e=1,w=0
            (1, 2, 1, 0),  # Stone B
            (1, 2, 1, 0), # Stone C
            (0, 1, 3, 2),  # Stone D
        ]
        for i, (n, s, e, w) in enumerate(fixed_stats):
            # name includes player name for clarity and uniqueness
            stone_name = f"{player.name} Stone {chr(65 + i)}"
            stone = Stone(name=stone_name, n=n, s=s, e=e, w=w)
            player.add_stone(stone)

    def start_game(self):
        while self.started:
            for player in self.players:
                self.player_turn(player)
                self.board.draw_board()
                self.check_game_over()
                if not self.started:
                    break
            
            
            pass
    
    def player_turn(self, player):
        print(f"It's {player.name}'s turn.")
        if len(player.stones) == 0:
            print(f"{player.name} has no stones left to play.")
            return

        # If player has an injected policy, try to use it first (inference mode)
        player_idx = self.players.index(player)
        if getattr(player, "policy", None) is not None:
            try:
                state = self.get_canonical_state(player_idx)
                action = player.choose_action(state)
                # decode action -> slot, row, col
                max_slots = len(self.initial_slots.get(player_idx, []))
                rows = self.board.rows
                cols = self.board.cols
                slot = action // (rows * cols)
                cell_idx = action % (rows * cols)
                r = cell_idx // cols
                c = cell_idx % cols

                # map slot -> stone name
                slot_name = None
                if 0 <= slot < max_slots:
                    slot_name = self.initial_slots[player_idx][slot]

                chosen_stone = None
                if slot_name is not None:
                    for s in player.stones:
                        if s.name == slot_name:
                            chosen_stone = s
                            break

                if chosen_stone is not None and self.board.isValidMove((r, c)):
                    self.place_stone(player, (r, c), chosen_stone)
                    return
                else:
                    print("Policy suggested invalid move; falling back to random move.")
            except Exception:
                print("Policy failed during decision; falling back to random move.")

        # Player's turn logic goes here
        # For now, just dummy place a stone at a random valid position
        valid_positions = [(r, c) for r in range(self.board.rows) for c in range(self.board.cols) if self.board.isValidMove((r, c))]
        if not valid_positions:
            print(f"No valid positions available for {player.name}.")
            return
        position = random.choice(valid_positions)
        stone = random.choice(player.stones)
        self.place_stone(player, position, stone)
        pass

    def get_canonical_state(self, player_idx: int):
        """Return a canonical JSON state string matching GameEnv and policy format."""
        import json

        board_repr = []
        for r in range(self.board.rows):
            for c in range(self.board.cols):
                cell = self.board.getField(r, c)
                if cell is None or getattr(cell, "player", None) is None:
                    board_repr.append(".")
                else:
                    owner_idx = 0 if self.players[0] == getattr(cell, "player") else 1
                    board_repr.append(f"{owner_idx}:{cell.name}")

        players_stones = []
        for p in self.players:
            players_stones.append([s.name for s in p.stones])

        payload = {"board": board_repr, "players": players_stones, "to_move": player_idx}
        return json.dumps(payload, sort_keys=True)
    

    def place_stone(self, player:Player, position:tuple, stone:Stone):
        if self.board.isValidMove(position):
            # ensure stone knows its owner for board logic
            try:
                # prefer stone API if available
                stone.set_Owner(player)
            except Exception:
                # fallback: attach attribute directly
                setattr(stone, "player", player)

            # board.placeStone expects (row, col, stone)
            row, col = position
            self.board.placeStone(row, col, stone)
            player.remove_stone(stone)
        else:
            print("Invalid move. Try again.")
            
    def placed_stone_count(self):
        count = 0
        for r in range(self.board.rows):
            for c in range(self.board.cols):
                cell = self.board.getField(r, c)
                if cell is not Board.EMPTY:
                    count += 1
        return count
            
    def check_game_over(self):
        no_player_stones = all(len(player.stones) == 0 for player in self.players)
        no_empty_fields = not any(
            self.board.isValidMove((r, c))
            for r in range(self.board.rows)
            for c in range(self.board.cols)
        )

        if no_player_stones or no_empty_fields:
            # Informative messages
            if no_player_stones:
                print("All players are out of stones!")
            if no_empty_fields:
                print("No empty fields left on the board!")

            # Obtain stone counts for all players (dict player -> count)
            owner_counts = self.board.get_current_stone_count()

            if not owner_counts:
                # no stones on board -> draw
                print("The game is a draw!")
            else:
                # find the highest stone count and which players have it
                # determine the highest stone count (max returns the value, not a player)
                max_count = max(owner_counts.values())
                winners = [p for p, c in owner_counts.items() if c == max_count]

                if len(winners) == 1:
                    winner = winners[0]
                    print(f"The winner is {winner.name}!")
                else:
                    # tie between two or more players -> draw
                    tied_names = ", ".join(p.name for p in winners)
                    print(f"The game is a draw between: {tied_names}!")

            self.started = False
            print("Game Over!")
        
      