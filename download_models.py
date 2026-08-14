import os
import hashlib
import sys
import logging
import urllib.request
import shutil
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_ARTIFACTS = {
    "artifacts/predictive/lstm_race/run_22f2c2a7/model.pt": os.environ.get("EXPECTED_LSTM_SHA256"),
    "artifacts/predictive/rf_fantasy/run_d6aa63ac/model.joblib": os.environ.get("EXPECTED_RF_SHA256")
}

def verify_checksum(filepath, expected_hash):
    if not expected_hash:
        return False
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest().lower() == expected_hash.lower()

def download_and_verify():
    missing = False
    require_predictive_models = os.environ.get(
        "REQUIRE_PREDICTIVE_MODELS", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    
    bucket = os.environ.get("ARTIFACT_BUCKET_URL")
    if not bucket:
        if require_predictive_models:
            logger.error("ARTIFACT_BUCKET_URL is not set.")
            sys.exit(1)
        logger.warning(
            "Predictive model storage is not configured; skipping optional "
            "race and fantasy artifacts. RAG routes can still start."
        )
        return False
        
    is_test = os.environ.get("PYTEST_CURRENT_TEST") is not None or os.environ.get("TEST_ENV") == "true"
    
    parsed_bucket = urlparse(bucket)
    if parsed_bucket.scheme != "https":
        if parsed_bucket.scheme == "file" and is_test:
            logger.info("Allowing file:// scheme for local testing.")
        else:
            logger.error(f"Invalid ARTIFACT_BUCKET_URL scheme: {parsed_bucket.scheme}. Only HTTPS is permitted in production.")
            sys.exit(1)
    
    for path, expected_hash in EXPECTED_ARTIFACTS.items():
        if not expected_hash:
            logger.error(f"Missing required expected SHA-256 hash for {path}")
            missing = True
            continue
            
        if not os.path.exists(path):
            logger.info(f"Artifact {path} missing. Attempting download from external storage...")
            filename = os.path.basename(path)
            source_url = f"{bucket}/{filename}"
            logger.info(f"Downloading {path} from {source_url}...")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            try:
                parsed = urlparse(source_url)
                if parsed.scheme == 'file':
                    local_path = urllib.request.url2pathname(parsed.path)
                    if parsed.netloc:
                        local_path = f"\\\\{parsed.netloc}{local_path}"
                    shutil.copy(local_path, path)
                else:
                    urllib.request.urlretrieve(source_url, path)
            except Exception as e:
                logger.error(f"Failed to download {path}: {e}")
                missing = True
                continue
                
        if not verify_checksum(path, expected_hash):
            logger.error(f"Checksum mismatch for {path}.")
            missing = True
        else:
            logger.info(f"Verified {path}.")
            
    if missing:
        logger.error("Readiness check failed: One or more critical artifacts are unavailable or corrupted.")
        sys.exit(1)
    else:
        logger.info("All deployment artifacts verified successfully.")
        return True
        
if __name__ == "__main__":
    download_and_verify()
