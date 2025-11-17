from enum import Enum

class Field(Enum):
    EMPTY = 0

class Board:
    """Minimal Board class: only initializes an empty grid."""

    def __init__(self, rows=3, cols=3):
            self.rows = rows
            self.cols = cols
            # grid holds Field.EMPTY for empty cells or a stone object
            self.grid = [[Field.EMPTY for _ in range(cols)] for _ in range(rows)]

    def placeStone(self, row, col, stone):
            """Place a stone object at (row, col). Raises on out-of-bounds or invalid stone."""
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                raise IndexError("position out of bounds")
            if stone is None:
                raise ValueError("stone must not be None")
            self.grid[row][col] = stone
            self._resolvePlacementEffects(row, col)
            
    def isValidMove(self, position):
            """Check if a move at position (row, col) is valid (within bounds and empty)."""
            row, col = position
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                return False
            return self.grid[row][col] == Field.EMPTY

    def getField(self, row, col):
            """Return the field at (row, col): either Empty.EMPTY or a stone object."""
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                raise IndexError("position out of bounds")
            return self.grid[row][col]
    
    def __repr__(self):
        return f"<Board {self.rows}x{self.cols}>"

    def get_current_stone_count(self):
        """Return a dict mapping each player to their current stone count."""
        owner_counts = {}
        for row in self.grid:
            for cell in row:
                if cell is not Field.EMPTY and hasattr(cell, "player"):
                    owner = cell.player
                    owner_counts[owner] = owner_counts.get(owner, 0) + 1
        return owner_counts
    
    
    def _resolvePlacementEffects(self, row, col):
        """Resolve captures against N/S/E/W neighbours after placing a stone."""
        placed = self.grid[row][col]
        if placed is Field.EMPTY:
            return    

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                continue
            target = self.grid[r][c]
            if target is Field.EMPTY:
                continue

            # both stones must expose a player identity
            if not hasattr(placed, "player") or not hasattr(target, "player"):
                continue

            if placed.player == target.player:
                continue

            # choose directional stats: attacker uses the direction toward the target,
            # defender uses the opposite direction (e.g. attacker.s vs defender.n).
            if dr == -1 and dc == 0:   # target is north of placed
                att_attr, def_attr = "n", "s"
            elif dr == 1 and dc == 0:  # target is south of placed
                att_attr, def_attr = "s", "n"
            elif dr == 0 and dc == -1: # target is west of placed
                att_attr, def_attr = "w", "e"
            elif dr == 0 and dc == 1:  # target is east of placed
                att_attr, def_attr = "e", "w"
            else:
                att_attr, def_attr = "attack", "attack"

            placed_attack = getattr(placed, att_attr, 0)
            target_attack = getattr(target, def_attr, 0)

            if placed_attack > target_attack:
                #print(f"{placed.player.name}'s stone at ({row},{col}) captures {target.player.name}'s stone at ({r},{c})!")
                try:
                    target.set_Owner(placed.player)
                except Exception:
                    print("Failed to change stone owner during capture.")
                    pass
       

    def draw_board(self):
        """Print a uniformly spaced text representation of the board."""
        # build display strings for each cell
        reps = []
        for row in self.grid:
            rep_row = []
            for cell in row:
                if cell is Field.EMPTY:
                    rep_row.append(".")
                else:
                    player = getattr(cell, "player", None)
                    name = getattr(player, "name", None)
                    rep_row.append((name[0].upper() if name else "?"))
            reps.append(rep_row)

        # determine column width and print rows with uniform spacing
        width = max((len(s) for r in reps for s in r), default=1)
        for r in reps:
            print(" ".join(s.center(width) for s in r))