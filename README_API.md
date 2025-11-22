# Battle Stones - API

This repository includes a small FastAPI-based controller for creating and
monitoring in-memory game instances for development and debugging.

Start the server:

```powershell
python scripts/run_api.py
```

Open the interactive docs at: http://127.0.0.1:8000/docs

Example usage (create a game):

```bash
curl -X POST "http://127.0.0.1:8000/games" -H "Content-Type: application/json" -d '{"rows":3,"cols":3, "player0":"human", "player1":"random"}'
```

Get state:

```bash
curl "http://127.0.0.1:8000/games/1"
```

Step (play action 0 for player 0):

```bash
curl -X POST "http://127.0.0.1:8000/games/1/step" -H "Content-Type: application/json" -d '{"action": 0, "player_idx": 0}'
```

Seed a game for deterministic behavior in the controller RNG:

```bash
curl -X POST "http://127.0.0.1:8000/games/1/seed" -H "Content-Type: application/json" -d '{"seed": 1234}'
```
