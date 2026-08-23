import asyncio
import logging
from contextlib import asynccontextmanager

from main import app, fantasy_predictor_svc, race_predictor_svc
from services.pitwall_replay import router as pitwall_replay_router
from download_models import download_and_verify

logger = logging.getLogger("paddox.predictive_bootstrap")

app.include_router(pitwall_replay_router)

# Preserve main.py's existing lifespan/model setup, but do not block the whole
# service behind artifact download / fantasy retraining. Render can mark /health
# healthy immediately; predictive artifacts finish in a background thread and
# are loaded into the already-created service objects when ready.
_original_lifespan = app.router.lifespan_context


def _bootstrap_predictive_models() -> None:
    try:
        result = download_and_verify()
        if result.get("rf_ready"):
            fantasy_predictor_svc.load_model()
            logger.info("Fantasy predictor reloaded after background bootstrap.")
        if result.get("lstm_ready"):
            race_predictor_svc.load_model()
            logger.info("Race predictor reloaded after background bootstrap.")
    except BaseException as exc:
        # download_and_verify can raise SystemExit when production policy requires
        # unavailable artifacts. Never let that terminate the serving process.
        logger.exception("Background predictive bootstrap failed: %s", exc)


@asynccontextmanager
async def _nonblocking_lifespan(app_instance):
    async with _original_lifespan(app_instance):
        task = asyncio.create_task(asyncio.to_thread(_bootstrap_predictive_models))
        app_instance.state.predictive_bootstrap_task = task
        yield
        if not task.done():
            task.cancel()


app.router.lifespan_context = _nonblocking_lifespan
