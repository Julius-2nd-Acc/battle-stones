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
        self.stone_amount = (col * row) // 2 if (col * row) % 2 == 0 else (col * row + 1) // 2
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
    
    def generate_stones_for_player(self, player: Player):
        stone_sets = [
            # Set 1: The Order (Balanced)
            [
                ("Squire", 1, 3, 1, 1),
                ("Archer", 1, 2, 2, 1),
                ("Cleric", 1, 1, 2, 2),
                ("Knight", 2, 1, 1, 2),
                ("Hero", 2, 2, 2, 2),
                ("Sentinel", 1, 2, 1, 3),
                ("Paladin", 2, 2, 1, 1),
                ("Mage", 1, 1, 3, 2),
                ("Captain", 2, 1, 2, 1),
                ("Guardian", 1, 3, 2, 1),
            ],
            # Set 2: The Horde (Aggressive)
            [
                ("Goblin", 1, 1, 1, 1),
                ("Orc", 2, 1, 1, 2),
                ("Berserker", 3, 3, 0, 0),
                ("Raider", 0, 0, 3, 3),
                ("Warlord", 2, 2, 2, 0),
                ("Imp", 1, 0, 2, 1),
                ("Savage", 2, 0, 1, 2),
                ("Brute", 3, 1, 0, 1),
                ("Destroyer", 0, 2, 2, 0),
                ("Tyrant", 2, 3, 1, 0),
            ],
            # Set 3: The Guard (Defensive)
            [
                ("Sentry", 1, 1, 1, 1),
                ("Shield", 1, 1, 3, 1),
                ("Wall", 1, 3, 1, 1),
                ("Tower", 3, 1, 1, 1),
                ("Bastion", 1, 2, 2, 1),
                ("Bulwark", 2, 1, 2, 1),
                ("Barricade", 1, 2, 1, 2),
                ("Fortress", 2, 2, 1, 2),
                ("Aegis", 1, 1, 2, 3),
                ("Warden", 1, 3, 1, 2),
            ],
            # Set 4: The Void (Chaos)
            [
                ("Shadow", 0, 0, 4, 2),
                ("Imp", 1, 1, 1, 1),
                ("Ghoul", 1, 1, 1, 1),  # Renamed one Imp for distinction
                ("Vortex", 3, 0, 0, 3),
                ("Dragon", 0, 3, 3, 0),
                ("Spectre", 1, 0, 3, 1),
                ("Abyssal", 2, 0, 1, 3),
                ("Reaper", 3, 0, 2, 0),
                ("Phantom", 0, 2, 0, 2),
                ("Chimera", 2, 1, 2, 1),
            ]
        ]

        selected_set = random.choice(stone_sets)

        num_stones_to_generate = min(self.stone_amount, len(selected_set))

        for name, n, s, e, w in selected_set[:num_stones_to_generate]:
            # name includes player name for clarity and uniqueness
            stone_name = f"{player.name} {name}"
            stone = Stone(name=stone_name, n=n, s=s, e=e, w=w, owner=player.name)
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

    def get_max_stones(self):
        board_size = self.board.rows * self.board.cols
        return (
            board_size if board_size < self.stone_amount * 2 and board_size % 2 == 0
            else board_size - 1 if board_size < self.stone_amount * 2
            else self.stone_amount * 2
        )
            
    def check_game_over(self):
        no_player_stones = all(len(player.stones) == 0 for player in self.players)
        no_empty_fields = not any(
            self.board.isValidMove((r, c))
            for r in range(self.board.rows)
            for c in range(self.board.cols)
        )
        placed_stone_count = self.placed_stone_count()

        if no_player_stones or no_empty_fields or placed_stone_count == self.get_max_stones():
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