from main import app
from services.pitwall_replay import router as pitwall_replay_router

app.include_router(pitwall_replay_router)
