from obj.board import Board
from obj.player import Player, PlayerType
from obj.stone import Stone
from services.state_builder import StateBuilder
import random
import json


class GameInstance:
    def __init__(self):
        
        self.players = []

        self.initial_slots = {}
        self.started = False
        self.board = None
    
    def add_player(self, player: Player):
        self.players.append(player)
        
    def setup_game(self, col=3, row=3, p1= PlayerType.HUMAN, p2= PlayerType.RANDOM):
        self.started = True
        self.setup_board(col=col, rows=row)
        self.setup_players(p1=p1, p2=p2)
    
    def setup_board(self,col=3,rows=3):
        self.board = Board(cols=col,rows=rows)

    def setup_players(self, p1= PlayerType.HUMAN, p2= PlayerType.RANDOM):
        self.add_player(Player("B", p1))
        self.add_player(Player("R", p2))
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

    def get_canonical_state(self, player_idx: int):
        """Return a canonical JSON state string matching GameEnv and policy format."""
        return StateBuilder.build_canonical_state(self, player_idx)
    

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
        return self.board.get_total_stone_count()
            
    def check_game_over(self):
        no_player_stones = all(len(player.stones) == 0 for player in self.players)
        no_empty_fields = not any(
            self.board.isValidMove((r, c))
            for r in range(self.board.rows)
            for c in range(self.board.cols)
        )

        if no_player_stones or no_empty_fields:
            # Obtain stone counts for all players (dict player -> count)
            owner_counts = self.board.get_current_stone_count()

            if not owner_counts:
                # no stones on board -> draw
                pass
            else:
                # find the highest stone count and which players have it
                max_count = max(owner_counts.values())
                winners = [p for p, c in owner_counts.items() if c == max_count]

            self.started = False