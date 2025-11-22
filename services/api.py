from typing import Optional
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from services.game_controller import GameController


class CreateGameRequest(BaseModel):
    rows: int = 3
    cols: int = 3
    player0: str = "human"
    player1: str = "random"


class StepRequest(BaseModel):
    action: int
    player_idx: int = 0


class SeedRequest(BaseModel):
    seed: Optional[int] = None


class AutoplayRequest(BaseModel):
    delay: float = 1.0


app = FastAPI(title="Battle Stones Game Controller")

# allow local browsers/tools to access the API easily during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = GameController()


@app.post("/games")
def create_game(req: CreateGameRequest):
    gid = controller.create_game_with_players(rows=req.rows, cols=req.cols, player0=req.player0, player1=req.player1)
    return {"id": gid}


@app.get("/games")
def list_games():
    return controller.list_games()


@app.get("/games/{game_id}")
def get_game(game_id: str, player_idx: int = 0):
    try:
        result = controller.get_state(game_id, player_idx=player_idx)
    except KeyError:
        raise HTTPException(status_code=404, detail="game not found")
    return result


@app.post("/games/{game_id}/step")
def step_game(game_id: str, req: StepRequest):
    if game_id not in controller.games:
        raise HTTPException(status_code=404, detail="game not found")
    result = controller.step(game_id, req.action, player_idx=req.player_idx)
    return result


@app.post("/games/{game_id}/seed")
def seed_game(game_id: str, req: SeedRequest):
    if game_id not in controller.games:
        raise HTTPException(status_code=404, detail="game not found")
    seed = controller.seed(game_id, req.seed)
    return {"seed": seed}


@app.post("/games/{game_id}/autoplay")
def start_autoplay(game_id: str, req: AutoplayRequest):
    if game_id not in controller.games:
        raise HTTPException(status_code=404, detail="game not found")
    ok = controller.start_autoplay(game_id, delay=req.delay)
    if not ok:
        raise HTTPException(status_code=400, detail="autoplay already running or game not found")
    return {"autoplay": "started", "delay": req.delay}


@app.delete("/games/{game_id}/autoplay")
def stop_autoplay(game_id: str):
    if game_id not in controller.games:
        raise HTTPException(status_code=404, detail="game not found")
    ok = controller.stop_autoplay(game_id)
    if not ok:
        raise HTTPException(status_code=400, detail="autoplay not running")
    return {"autoplay": "stopped"}


@app.get("/games/{game_id}/status")
def game_status(game_id: str):
    if game_id not in controller.games:
        raise HTTPException(status_code=404, detail="game not found")
    return {"autoplay": controller.is_autoplaying(game_id)}


@app.delete("/games/{game_id}")
def delete_game(game_id: str):
    ok = controller.delete_game(game_id)
    if not ok:
        raise HTTPException(status_code=404, detail="game not found")
    return {"deleted": game_id}
