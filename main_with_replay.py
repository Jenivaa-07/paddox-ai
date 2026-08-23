import logging
import threading

from main import app, fantasy_predictor_svc, race_predictor_svc
from services.pitwall_replay import router as pitwall_replay_router
from download_models import download_and_verify

logger = logging.getLogger("paddox.predictive_bootstrap")

app.include_router(pitwall_replay_router)


def _bootstrap_predictive_models() -> None:
    """Restore optional predictive artifacts without blocking Render startup."""
    try:
        result = download_and_verify()
        if result.get("rf_ready"):
            fantasy_predictor_svc.load_model()
            logger.info("Fantasy predictor reloaded after background bootstrap.")
        if result.get("lstm_ready"):
            race_predictor_svc.load_model()
            logger.info("Race predictor reloaded after background bootstrap.")
    except BaseException as exc:
        # download_and_verify may raise SystemExit when strict artifact policy is
        # enabled. This background worker must never terminate the API process.
        logger.exception("Background predictive bootstrap failed: %s", exc)


# Uvicorn can bind immediately while missing models are restored independently.
threading.Thread(
    target=_bootstrap_predictive_models,
    name="paddox-predictive-bootstrap",
    daemon=True,
).start()
