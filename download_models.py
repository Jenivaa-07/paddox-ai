import hashlib
import json
import logging
import os
import shutil
import sys
import urllib.request
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OFFICIAL_ARTIFACTS = {
    "artifacts/predictive/lstm_race/run_22f2c2a7/model.pt": os.environ.get("EXPECTED_LSTM_SHA256"),
    "artifacts/predictive/rf_fantasy/run_d6aa63ac/model.joblib": os.environ.get("EXPECTED_RF_SHA256"),
}
RF_CURRENT_MODEL = "artifacts/predictive/rf_fantasy/current_model.json"


def sha256_file(filepath):
    digest = hashlib.sha256()
    with open(filepath, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def verify_checksum(filepath, expected_hash):
    if not expected_hash or not os.path.isfile(filepath):
        return False
    return sha256_file(filepath) == expected_hash.strip().lower()


def _read_current_rf_path():
    try:
        with open(RF_CURRENT_MODEL, "r", encoding="utf-8") as handle:
            run_id = json.load(handle).get("run_id")
        if not run_id:
            return None
        return os.path.join("artifacts", "predictive", "rf_fantasy", run_id, "model.joblib")
    except Exception:
        return None


def _runtime_rf_is_valid():
    model_path = _read_current_rf_path()
    if not model_path or not os.path.isfile(model_path):
        return False

    run_dir = os.path.dirname(model_path)
    provenance_path = os.path.join(run_dir, "provenance.json")
    if not os.path.isfile(provenance_path):
        return False

    try:
        with open(provenance_path, "r", encoding="utf-8") as handle:
            provenance = json.load(handle)
        expected = str(provenance.get("sha256") or "").strip().lower()
        return bool(expected) and sha256_file(model_path) == expected
    except Exception:
        return False


def _download_verified_artifact(bucket, path, expected_hash):
    if not expected_hash:
        logger.warning("No expected SHA-256 configured for %s; skipping remote download.", path)
        return False

    if os.path.isfile(path) and verify_checksum(path, expected_hash):
        logger.info("Verified existing artifact %s.", path)
        return True

    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass

    filename = os.path.basename(path)
    source_url = f"{bucket.rstrip('/')}/{filename}"
    parsed = urlparse(source_url)
    is_test = os.environ.get("PYTEST_CURRENT_TEST") is not None or os.environ.get("TEST_ENV") == "true"

    if parsed.scheme != "https":
        if not (parsed.scheme == "file" and is_test):
            raise RuntimeError(
                f"Invalid ARTIFACT_BUCKET_URL scheme: {parsed.scheme}. Only HTTPS is permitted in production."
            )

    logger.info("Downloading %s from %s...", path, source_url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        if parsed.scheme == "file":
            local_path = urllib.request.url2pathname(parsed.path)
            if parsed.netloc:
                local_path = f"\\\\{parsed.netloc}{local_path}"
            shutil.copy(local_path, path)
        else:
            request = urllib.request.Request(
                source_url,
                headers={"User-Agent": "PADDOX-AI/1.0 artifact-bootstrap"},
            )
            with urllib.request.urlopen(request, timeout=60) as response, open(path, "wb") as target:
                shutil.copyfileobj(response, target)
    except Exception as error:
        logger.error("Failed to download %s: %s", path, error)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        return False

    if not verify_checksum(path, expected_hash):
        logger.error("Checksum mismatch for %s; refusing artifact.", path)
        try:
            os.remove(path)
        except OSError:
            pass
        return False

    logger.info("Verified %s.", path)
    return True


def _bootstrap_runtime_fantasy():
    enabled = os.environ.get("FANTASY_RUNTIME_BOOTSTRAP", "true").strip().lower() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
        return False

    logger.warning(
        "Verified fantasy model artifact is unavailable. "
        "Bootstrapping the same Random Forest architecture from real Jolpica F1 results."
    )
    try:
        from bootstrap_fantasy_model import ensure_runtime_fantasy_model

        result = ensure_runtime_fantasy_model()
        logger.info(
            "Runtime fantasy bootstrap complete: %s (sha256=%s)",
            result.get("run_id"),
            result.get("sha256"),
        )
        return _runtime_rf_is_valid()
    except Exception as error:
        logger.exception("Runtime fantasy bootstrap failed: %s", error)
        return False


def download_and_verify():
    require_predictive_models = os.environ.get(
        "REQUIRE_PREDICTIVE_MODELS", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}

    bucket = str(os.environ.get("ARTIFACT_BUCKET_URL") or "").strip()
    official_lstm_path = "artifacts/predictive/lstm_race/run_22f2c2a7/model.pt"
    official_rf_path = "artifacts/predictive/rf_fantasy/run_d6aa63ac/model.joblib"

    lstm_ready = verify_checksum(official_lstm_path, OFFICIAL_ARTIFACTS[official_lstm_path])
    rf_ready = verify_checksum(official_rf_path, OFFICIAL_ARTIFACTS[official_rf_path])
    if not rf_ready:
        rf_ready = _runtime_rf_is_valid()

    if bucket:
        if not lstm_ready:
            lstm_ready = _download_verified_artifact(
                bucket, official_lstm_path, OFFICIAL_ARTIFACTS[official_lstm_path]
            )
        if not rf_ready:
            rf_ready = _download_verified_artifact(
                bucket, official_rf_path, OFFICIAL_ARTIFACTS[official_rf_path]
            )
            if rf_ready:
                with open(RF_CURRENT_MODEL, "w", encoding="utf-8") as handle:
                    json.dump({"run_id": "run_d6aa63ac"}, handle, indent=2)
    else:
        logger.warning(
            "ARTIFACT_BUCKET_URL is not configured. "
            "Official remote predictive artifacts cannot be downloaded."
        )

    if not rf_ready:
        rf_ready = _bootstrap_runtime_fantasy()

    if require_predictive_models and (not lstm_ready or not rf_ready):
        missing = []
        if not lstm_ready:
            missing.append("LSTM race model")
        if not rf_ready:
            missing.append("Random Forest fantasy model")
        logger.error("Required predictive models unavailable: %s", ", ".join(missing))
        sys.exit(1)

    if not lstm_ready:
        logger.warning("LSTM race artifact is unavailable; race prediction readiness remains degraded.")
    if not rf_ready:
        logger.warning("Fantasy RF artifact is unavailable; /predict-fantasy will report model_not_ready.")
    else:
        logger.info("Fantasy Random Forest model is ready.")

    return {"lstm_ready": lstm_ready, "rf_ready": rf_ready}


if __name__ == "__main__":
    download_and_verify()
