import os
import hashlib
import sys
import json
import logging
import urllib.request
import shutil
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPECTED_ARTIFACTS = {
    "artifacts/predictive/lstm_race/run_22f2c2a7/model.pt": None,
    "artifacts/predictive/rf_fantasy/run_d6aa63ac/model.joblib": os.environ.get("EXPECTED_RF_SHA256"),
    "artifacts/sentiment/run_4025bd5a_1785255656/model.safetensors": None,
}

def verify_checksum(filepath, expected_hash):
    if not expected_hash:
        return True
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest().lower() == expected_hash.lower()

def download_and_verify():
    missing = False
    for path, expected_hash in EXPECTED_ARTIFACTS.items():
        if not os.path.exists(path):
            logger.info(f"Artifact {path} missing. Attempting download from external storage...")
            bucket = os.environ.get("ARTIFACT_BUCKET_URL")
            if not bucket:
                logger.error(f"Cannot download {path}: ARTIFACT_BUCKET_URL is not set.")
                missing = True
                continue
            
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
        
if __name__ == "__main__":
    download_and_verify()
