"""Stone DTO: load stone definitions from JSON and create `Stone` instances.

Provides:
- load_stones_from_file(path) -> List[Stone]
- add_stones_to_player(path, player)

JSON format example (array of objects):
[
  {"name": "Stone A", "n": 1, "s": 2, "e": 3, "w": 4, "effect": null},
  {"name": "Stone B", "n": 0, "s": 0, "e": 1, "w": 1}
]

This module purposely keeps a very small surface area and falls back to defaults
for missing attributes.
"""
from typing import List
import json
import os

from .stone import Stone


class StoneDTO:
    @staticmethod
    def _create_from_dict(d: dict) -> Stone:
        name = d.get("name", "Stone")
        effect = d.get("effect", None)
        try:
            n = int(d.get("n", 0))
            s = int(d.get("s", 0))
            e = int(d.get("e", 0))
            w = int(d.get("w", 0))
        except (TypeError, ValueError):
            # fallback to zeros if invalid
            n = s = e = w = 0
        return Stone(name=name, effect=effect, n=n, s=s, e=e, w=w)

    @classmethod
    def load_stones_from_file(cls, path: str) -> List[Stone]:
        """Load stones from a JSON file and return a list of Stone objects.

        Raises FileNotFoundError or ValueError for invalid JSON.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Stone JSON file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, list):
            raise ValueError("Stone JSON must be an array of stone objects")

        stones: List[Stone] = []
        for item in data:
            if not isinstance(item, dict):
                # skip invalid entries
                continue
            stones.append(cls._create_from_dict(item))

        return stones

    @classmethod
    def add_stones_to_player(cls, path: str, player) -> int:
        """Load stones from file and append them to the player's stones list.

        Returns the number of stones added.
        The player object must implement `add_stone(stone)`.
        """
        stones = cls.load_stones_from_file(path)
        for s in stones:
            try:
                player.add_stone(s)
            except Exception:
                # fallback: directly append to list if available
                lst = getattr(player, "stones", None)
                if isinstance(lst, list):
                    lst.append(s)
                else:
                    raise
        return len(stones)
