from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from .fight_prediction import FighterStats, predict_fight
from .model import load_model

class FightRequest(BaseModel):
    fighter1: str
    fighter2: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, scaler, features = load_model(model_dir="models")

script_dir = Path(__file__).parent
data_file_path = script_dir / "data" / "complete_ufc_data.csv"
fighter_stats = FighterStats(str(data_file_path))

@app.post("/predict/")
async def predict_fight_post(request: FightRequest):
    try:
        winner, confidence, advantages = predict_fight(model, scaler, fighter_stats, request.fighter1, request.fighter2)
        return {
            "winner": winner,
            "confidence": float(confidence),
            "advantages": advantages
        }
    except ValueError as e:
        return {"error": str(e)}